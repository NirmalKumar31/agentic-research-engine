"""Hosted demo guardrails.

A public URL backed by a real API key is a spend endpoint for anyone who
finds it. Everything here is server-controlled: the frontend can ask for
whatever it likes, and in demo mode the server substitutes its own limits.

The rule throughout is that a client-supplied value may only ever make a
run *smaller*, never larger.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

from agentic_research.config import LLMMode, Settings


@dataclass(frozen=True)
class DemoLimits:
    """Fixed ceilings applied to anonymous runs.

    Deliberately much tighter than the defaults a local user gets. A demo
    exists to show the system works, not to do someone's research for free.
    """

    max_query_chars: int = 300
    min_query_chars: int = 10
    max_rounds: int = 1
    max_sources: int = 6
    max_sources_per_round: int = 6
    max_search_queries: int = 6
    max_llm_calls: int = 20
    max_runtime_seconds: float = 240.0
    max_concurrent_runs: int = 2
    runs_per_ip_per_hour: int = 2
    global_runs_per_day: int = 1
    """Deliberately tiny, and derived rather than guessed.

    A run's worst case is its provider-request ceiling, not the ~22 one run
    happened to use. On the validated 50-requests-per-day account that
    affords a single run. The earlier default of 60 would have drained the
    quota within three visitors and failed opaquely for everyone after."""
    max_cloud_cost_usd: float = 0.05
    max_search_credits: float = 8.0
    max_provider_requests_per_day: int = 50
    """The account-level ceiling the run caps are derived from. Raise this
    with the plan, not independently."""


def runs_affordable(provider_requests_per_day: int, worst_case_requests_per_run: int = 40) -> int:
    """How many demo runs a provider quota can safely support.

    Derived from the **worst case** a run may emit, not the average it
    happened to emit once. A run reserves up to MAX_CLOUD_CALLS provider
    requests, and structured repairs and compatibility retries each consume
    one, so the observed ~22 is a floor rather than a bound.

    Returns **0** when the quota cannot afford even one safe run. An earlier
    version used ``max(1, ...)``, which promised a run the quota could not
    pay for and turned a predictable refusal into a mid-run 429.
    """
    if worst_case_requests_per_run <= 0:
        return 0
    return provider_requests_per_day // worst_case_requests_per_run


def limits_from_settings(settings: Settings) -> DemoLimits:
    """Build the demo ceilings, honouring the few that are configurable.

    Runtime in particular has to be tunable: 240s is right for a cloud
    model and nowhere near enough for a local one, and hard-coding it would
    make the web path untestable against Ollama.
    """
    return DemoLimits(
        max_runtime_seconds=settings.demo_max_runtime_seconds,
        runs_per_ip_per_hour=settings.demo_runs_per_hour,
        max_concurrent_runs=settings.demo_max_concurrent_runs,
        max_provider_requests_per_day=settings.demo_provider_requests_per_day,
        global_runs_per_day=runs_affordable(
            settings.demo_provider_requests_per_day,
            # The worst case a single run may emit, so capacity is never
            # promised beyond what the quota can actually pay for.
            worst_case_requests_per_run=max(1, settings.max_cloud_calls),
        ),
    )


def apply_demo_limits(settings: Settings, limits: DemoLimits) -> Settings:
    """Clamp settings to the demo ceilings.

    Returns a copy; the process-wide settings object is never mutated, so a
    request cannot widen limits for the next one.
    """
    return settings.model_copy(
        update={
            "max_research_rounds": min(settings.max_research_rounds, limits.max_rounds),
            "max_sources": min(settings.max_sources, limits.max_sources),
            "max_sources_per_round": min(
                settings.max_sources_per_round, limits.max_sources_per_round
            ),
            "max_search_queries": min(settings.max_search_queries, limits.max_search_queries),
            "max_llm_calls": min(settings.max_llm_calls, limits.max_llm_calls),
            "max_cloud_cost_usd": min(
                settings.max_cloud_cost_usd or limits.max_cloud_cost_usd,
                limits.max_cloud_cost_usd,
            ),
            "max_search_credits": min(
                settings.max_search_credits or limits.max_search_credits,
                limits.max_search_credits,
            ),
            # A hosted demo never falls back to a paid provider implicitly.
            "allow_cloud_fallback": False,
            # Free hosting has an ephemeral filesystem; persisting is both
            # pointless and a slow disk write on the request path.
            "persist_runs": False,
            "checkpoint_backend": "memory",
        }
    )


class CapacityError(Exception):
    """The demo is at capacity. Carries an honest, specific reason."""

    def __init__(self, reason: str, retry_after_seconds: int | None = None) -> None:
        self.reason = reason
        self.retry_after_seconds = retry_after_seconds
        super().__init__(reason)


@dataclass
class _Window:
    """Fixed-size timestamp window for one key."""

    stamps: deque[float] = field(default_factory=deque)

    def prune(self, horizon: float, now: float) -> None:
        while self.stamps and now - self.stamps[0] > horizon:
            self.stamps.popleft()


class RateLimiter:
    """Per-client and global run limits, held in memory.

    In-memory is the right scope for a single free-tier instance, and is
    stated as a limitation rather than dressed up: behind multiple replicas
    these counters would diverge and the real ceiling would be the cloud
    spend budget, which is enforced per run in the engine itself.
    """

    def __init__(self, limits: DemoLimits) -> None:
        self._limits = limits
        self._per_client: dict[str, _Window] = {}
        self._global = _Window()
        self._active = 0
        self._lock = asyncio.Lock()

    async def acquire(self, client_key: str) -> None:
        """Claim a run slot, or raise :class:`CapacityError`."""
        now = time.monotonic()
        async with self._lock:
            if self._active >= self._limits.max_concurrent_runs:
                raise CapacityError(
                    "The demo is running at capacity right now. Try again in a minute.",
                    retry_after_seconds=60,
                )

            if self._limits.global_runs_per_day <= 0:
                # The provider quota cannot pay for even one safe run. Say
                # so plainly instead of accepting a run that will 429.
                raise CapacityError(
                    "Live runs are disabled: the configured provider quota "
                    "cannot cover a full research run. The recorded example "
                    "run is still available.",
                    retry_after_seconds=None,
                )

            self._global.prune(86_400, now)
            if len(self._global.stamps) >= self._limits.global_runs_per_day:
                raise CapacityError(
                    "The demo has reached its daily budget. It resets within 24 hours.",
                    retry_after_seconds=3_600,
                )

            window = self._per_client.setdefault(client_key, _Window())
            window.prune(3_600, now)
            if len(window.stamps) >= self._limits.runs_per_ip_per_hour:
                oldest = window.stamps[0]
                wait = int(3_600 - (now - oldest)) + 1
                raise CapacityError(
                    f"You have used your {self._limits.runs_per_ip_per_hour} demo runs "
                    "for this hour.",
                    retry_after_seconds=max(wait, 60),
                )

            window.stamps.append(now)
            self._global.stamps.append(now)
            self._active += 1

    async def release(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)

    async def snapshot(self) -> dict[str, int]:
        now = time.monotonic()
        async with self._lock:
            self._global.prune(86_400, now)
            return {
                "active_runs": self._active,
                "max_concurrent_runs": self._limits.max_concurrent_runs,
                "runs_today": len(self._global.stamps),
                "global_runs_per_day": self._limits.global_runs_per_day,
            }


def validate_query(query: str, limits: DemoLimits) -> str:
    """Check a user-supplied question, returning the cleaned form."""
    cleaned = " ".join((query or "").split())
    if len(cleaned) < limits.min_query_chars:
        raise ValueError(
            f"Please ask a fuller question (at least {limits.min_query_chars} characters)."
        )
    if len(cleaned) > limits.max_query_chars:
        raise ValueError(
            f"Question is too long for the demo (limit {limits.max_query_chars} characters)."
        )
    return cleaned


def demo_mode_summary(settings: Settings, limits: DemoLimits) -> dict[str, object]:
    """What the client is allowed to know about the configuration.

    Model names and ceilings are fine to publish; anything that could
    identify or expose a credential is not, and no key or environment value
    is ever included here.
    """
    return {
        "mode": settings.llm_mode.value,
        "max_query_chars": limits.max_query_chars,
        "max_rounds": limits.max_rounds,
        "max_sources": limits.max_sources,
        "max_runtime_seconds": limits.max_runtime_seconds,
        "runs_per_hour": limits.runs_per_ip_per_hour,
        "local_models_available": settings.llm_mode is not LLMMode.CLOUD,
    }
