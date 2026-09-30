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
    runs_per_ip_per_hour: int = 10
    """Also the hard ceiling configuration may not exceed (``_MAX``).

    Raised from 2 because that was the binding constraint for the
    *operator*, not for a visitor: the daily allowance below is derived
    from real provider spend and is the cap that actually protects the
    account, so a tight hourly limit only stopped the owner from
    exercising the demo. A visitor still cannot exceed the derived
    daily total."""
    global_runs_per_day: int = 1
    """Derived from the configured quota rather than written down.

    A run's worst case is its provider-request ceiling, not whatever a
    given run happens to use, so the cap is computed from the ceiling. A
    fixed larger default would drain a small quota within a few visitors
    and then fail opaquely for everyone after."""
    max_cloud_cost_usd: float = 0.05
    max_search_credits: float = 8.0
    max_provider_requests_per_day: int = 50
    """Conservative default for the account-level quota the run caps are
    derived from. Raise this with the provider plan, not independently."""

    # Paid dimensions the engine reserves against before each request.
    # Every one of these must be clamped, not just the cost: a run bounded
    # only by dollars can still exhaust a request or token quota, and the
    # engine refuses a run whose reservation exceeds any single ceiling.
    max_cloud_calls: int = 20
    max_cloud_input_tokens: int = 240_000
    """Denominated in the byte bound, not in real tokens.

    The reservation is an upper bound now (llm/token_bound.py), and it
    over-reserves by 2x to 6x depending on content. The old 120,000 was
    chosen when the reservation was an estimate of real tokens, so the
    change of denominator quietly made it about six times tighter.

    Sized from the recorded run: 40,222 real input tokens across 2
    rounds and 12 sources, so roughly 20,100 for a public run at half
    that budget. At the worst measured prose ratio of 5.5 bytes/token
    the bound reserves about 110,600 -- 92% of the old ceiling, which
    would have refused a public run only slightly larger than the
    extrapolation. 240,000 restores roughly 2x headroom.

    The dollar reservation is deliberately unchanged. This
    re-denominates a token allowance; it does not widen what a run may
    spend."""
    max_cloud_output_tokens: int = 20_000
    max_provider_requests: int = 30
    """HTTP requests to the *model* provider in one run.

    Search is metered separately in provider credits and never reserves
    against this; the two budgets bound different vendors and conflating
    them would let one silently exhaust the other.

    Sized for one demo run: ``max_cloud_calls`` (20) model requests plus
    headroom for structured repairs, compatibility retries and transport
    retries, each of which is a further request against the same ceiling.
    The engine's default of 120 is a local-development figure and far too
    loose to expose anonymously."""


# A fresh instance carries the public-demo maxima. Operators configure the
# few tunable dimensions downward from here; `limits_from_settings` clamps
# so a configured value can never exceed the ceiling declared above.
_MAX = DemoLimits()


def _at_most(configured: float, ceiling: float) -> float:
    """Clamp a configured budget, reading 0 as the engine reads it.

    Zero means *unlimited* to the budget code, so a plain ``min`` would let
    ``MAX_CLOUD_CALLS=0`` clamp every other run to zero while actually
    disabling the ceiling. Treat it as "unset" and fall back to the demo
    maximum instead.
    """
    return ceiling if configured <= 0 else min(configured, ceiling)


def runs_affordable(provider_requests_per_day: int, worst_case_requests_per_run: int = 40) -> int:
    """How many demo runs a provider quota can safely support.

    Derived from the **worst case** a run may emit, not from an average.

    That worst case is ``max_provider_requests`` -- HTTP requests to the
    model provider -- and not ``max_cloud_calls``. The two are different
    numbers for a reason: one logical call becomes several requests when
    a structured-output repair or a compatibility retry fires, so the
    request ceiling is the larger of the pair. Deriving capacity from
    the smaller one promised more runs than the quota could pay for. At
    the deployed values (50 per day, 20 calls, 30 requests) it allowed
    two runs that could together issue 60 requests against a 50-request
    quota, and the second run would have died mid-flight on a 429 after
    spending OpenAI tokens.

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

    Configuration may only move a ceiling **down**. Without the clamps
    below, ``DEMO_RUNS_PER_HOUR=1000`` in the environment would simply
    become the limit, which makes the whole dataclass decorative -- the
    server has to enforce its own maximum rather than trust its own
    deployment config.
    """
    # The effective per-run ceiling on requests to the model provider,
    # clamped the same way the run itself will be clamped, so the daily
    # allowance is derived from the number each admitted run may actually
    # reach rather than from a smaller, friendlier one.
    provider_requests = int(_at_most(settings.max_provider_requests, _MAX.max_provider_requests))
    return DemoLimits(
        max_runtime_seconds=min(settings.demo_max_runtime_seconds, _MAX.max_runtime_seconds),
        runs_per_ip_per_hour=min(settings.demo_runs_per_hour, _MAX.runs_per_ip_per_hour),
        max_concurrent_runs=min(settings.demo_max_concurrent_runs, _MAX.max_concurrent_runs),
        # Deliberately *not* clamped. This one describes an external fact --
        # what the provider account actually allows per day -- rather than
        # how much of it this demo may spend. Capping it at the default
        # would not make anything safer; it would make the derived run cap
        # wrong for anyone on a larger plan.
        max_provider_requests_per_day=settings.demo_provider_requests_per_day,
        global_runs_per_day=runs_affordable(
            settings.demo_provider_requests_per_day,
            # The worst case a single run may emit *after clamping*, so
            # capacity is never promised beyond what the quota can pay
            # for. Provider requests, not logical calls: a repair or a
            # retry turns one call into two requests, and the quota is
            # spent in requests.
            worst_case_requests_per_run=max(1, provider_requests),
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
            # Every paid dimension the engine reserves against. Clamping
            # cost alone is not enough: the reservation check refuses a run
            # that would breach *any* ceiling, so an unclamped request or
            # token budget is a way to spend past the intended envelope
            # while the dollar figure still looks correct.
            "max_cloud_cost_usd": _at_most(settings.max_cloud_cost_usd, limits.max_cloud_cost_usd),
            "max_search_credits": _at_most(settings.max_search_credits, limits.max_search_credits),
            "max_cloud_calls": int(_at_most(settings.max_cloud_calls, limits.max_cloud_calls)),
            "max_cloud_input_tokens": int(
                _at_most(settings.max_cloud_input_tokens, limits.max_cloud_input_tokens)
            ),
            "max_cloud_output_tokens": int(
                _at_most(settings.max_cloud_output_tokens, limits.max_cloud_output_tokens)
            ),
            "max_provider_requests": int(
                _at_most(settings.max_provider_requests, limits.max_provider_requests)
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
    """Concurrency and per-client rate, held in memory.

    Deliberately *not* the daily admission count. That lives in the
    shared Key Value counter, and keeping a second one here was a
    latent inconsistency rather than a safety net: this one is a rolling
    24-hour window, the shared one is keyed by UTC day, and the two
    disagree in both directions. A web restart empties this one while
    the shared count stands; UTC midnight resets the shared one while
    this one still holds the last 24 hours. Whichever number was
    reported was wrong half the time, and there is no version of "two
    authorities" that is better than one.

    What is left here is genuinely process-scoped and belongs that way:
    how many runs are in flight on this instance, and how often one
    caller may start them. Both bound accidents, not money.
    """

    def __init__(self, limits: DemoLimits) -> None:
        self._limits = limits
        self._per_client: dict[str, _Window] = {}
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
            self._active += 1

    async def release(self, client_key: str | None = None) -> None:
        """Give back what an admission took.

        ``client_key`` is passed when the admission was subsequently
        refused -- by the shared daily quota, which is checked after
        this one. Releasing only the concurrency slot would leave the
        caller charged an hourly allowance for a run that never
        happened, so a refused request costs them nothing.
        """
        async with self._lock:
            self._active = max(0, self._active - 1)
            if client_key is None:
                return
            window = self._per_client.get(client_key)
            if window and window.stamps:
                window.stamps.pop()

    async def snapshot(self) -> dict[str, int]:
        """What this process knows. Deliberately excludes the day count.

        ``runs_today`` used to be reported here from the process-local
        window and published through /api/health, where it read as the
        shared count and was not. It is gone rather than renamed: a
        truthful shared figure needs a non-mutating read of the Key
        Value counter, and an approximate one is worse than none.
        """
        async with self._lock:
            return {
                "active_runs": self._active,
                "max_concurrent_runs": self._limits.max_concurrent_runs,
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

    Ceilings are fine to publish; anything that could identify or expose a
    credential is not, and no key or environment value is ever included.

    ``service_mode`` is what the client should actually reason about.
    ``LLM_MODE`` is boot configuration, not a capability: the hosted
    replay instance runs with ``LLM_MODE=local`` purely because cloud mode
    refuses to start without an API key it will never use. There is no
    Ollama on that host, so reporting "local" as though a local model were
    available would be false.
    """
    live = settings.live_research_enabled
    return {
        "service_mode": "live" if live else "replay",
        "live_research_enabled": live,
        # Only meaningful when something can actually run. While replaying,
        # no model of any kind is reachable from this service.
        "local_models_available": bool(live and settings.llm_mode is not LLMMode.CLOUD),
        "mode": settings.llm_mode.value if live else None,
        "max_query_chars": limits.max_query_chars,
        "max_rounds": limits.max_rounds,
        "max_sources": limits.max_sources,
        "max_runtime_seconds": limits.max_runtime_seconds,
        "runs_per_hour": limits.runs_per_ip_per_hour,
    }
