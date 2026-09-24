"""Search execution: provider selection, retry policy and concurrency limits.

The graph's search workers call :meth:`SearchService.run_query`. Everything
about *how* a search is executed — which vendor, how many at once, what to do
about a 429 — lives here so the graph stays a description of the research
process rather than a pile of transport concerns.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

import httpx

from agentic_research.config import Settings
from agentic_research.models import SearchQuery
from agentic_research.observability import get_logger
from agentic_research.search.base import (
    SearchAuthError,
    SearchError,
    SearchOptions,
    SearchProvider,
    SearchRateLimitError,
    SearchResponse,
)
from agentic_research.search.brave import BraveProvider
from agentic_research.search.tavily import TavilyProvider

log = get_logger(__name__)


class SearchProviderNotConfigured(Exception):
    pass


class SearchBudgetExceeded(Exception):
    """The run would exceed its search credit ceiling.

    Search is metered in provider credits, which are money. Checked before
    dispatch using the credits this specific call would consume, so a
    configured ceiling is never silently passed.
    """


def build_provider(settings: Settings) -> SearchProvider:
    """Instantiate the configured provider.

    A registry rather than an if-chain would be tidier with five providers.
    With two it would be indirection for its own sake.
    """
    name = settings.search_provider.strip().lower()
    if name == "tavily":
        if not settings.tavily_api_key:
            raise SearchProviderNotConfigured(
                "SEARCH_PROVIDER=tavily requires TAVILY_API_KEY. "
                "Get a free key at https://app.tavily.com"
            )
        return TavilyProvider(
            settings.tavily_api_key.get_secret_value(), timeout=settings.search_timeout_seconds
        )
    if name == "brave":
        if not settings.brave_api_key:
            raise SearchProviderNotConfigured("SEARCH_PROVIDER=brave requires BRAVE_API_KEY")
        return BraveProvider(
            settings.brave_api_key.get_secret_value(), timeout=settings.search_timeout_seconds
        )
    raise SearchProviderNotConfigured(
        f"Unknown SEARCH_PROVIDER {settings.search_provider!r}. Supported: tavily, brave"
    )


@dataclass
class SearchStats:
    calls: int = 0
    failures: int = 0
    retries: int = 0
    credits: float = 0.0
    latency_s: float = 0.0
    results: int = 0
    by_error: dict[str, int] = field(default_factory=dict)


class SearchService:
    """Owns the HTTP client, the concurrency limit and the retry policy.

    One instance per run, shared by every parallel worker through the graph's
    runtime context. That placement matters: a module-level semaphore would be
    bound to whichever event loop created it and would leak limits between
    runs, which is a genuinely unpleasant bug to track down in tests.
    """

    def __init__(
        self,
        provider: SearchProvider,
        settings: Settings,
        *,
        max_attempts: int = 3,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.stats = SearchStats()
        self._max_attempts = max_attempts
        self._semaphore = asyncio.Semaphore(settings.max_parallel_searches)
        self._client: httpx.AsyncClient | None = None
        # Credits are reserved before dispatch, not counted after it.
        # Checking `stats.credits + cost <= ceiling` and incrementing on
        # completion leaves the whole request in between: every parallel
        # worker reads the same remaining budget, every one of them passes,
        # and the ceiling is exceeded by however many were in flight.
        self._credit_lock = asyncio.Lock()
        self._reserved_credits = 0.0

    async def __aenter__(self) -> SearchService:
        self._client = httpx.AsyncClient(
            timeout=self.settings.search_timeout_seconds,
            headers={"User-Agent": self.settings.user_agent},
            limits=httpx.Limits(max_connections=self.settings.max_parallel_searches * 2),
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def options_for(self, query: SearchQuery) -> SearchOptions:
        """Translate a planned query into provider options.

        Depth escalation lives here rather than in the graph: rounds after the
        first have already demonstrated that cheap search missed something, so
        paying double for those queries is a targeted spend rather than a
        blanket upgrade.
        """
        depth = query.depth or self.settings.search_depth
        if self.settings.search_escalate_on_followup and query.round_number > 1:
            depth = "advanced"
        return SearchOptions(
            max_results=self.settings.max_search_results,
            depth=depth,
            include_raw_content=True,
        )

    async def run_query(self, query: SearchQuery) -> SearchResponse:
        """Execute one query with bounded concurrency and bounded retries.

        Never raises for an expected failure. A search worker that raised would
        take down its whole parallel round, so failures come back as a
        ``SearchResponse`` carrying an error instead.
        """
        if self._client is None:
            raise RuntimeError("SearchService must be used as an async context manager")

        options = self.options_for(query)
        cost = self.provider.credits_for(options)
        started = time.perf_counter()

        async with self._semaphore:
            last_error = "unknown error"
            for attempt in range(1, self._max_attempts + 1):
                # Reserved per attempt, because a retry is another billed
                # provider call rather than a free continuation of the
                # first one.
                if not await self._reserve_credits(cost):
                    return self._failure(
                        query,
                        f"search credit budget exhausted "
                        f"({self._reserved_credits:g} of "
                        f"{self.settings.max_search_credits:g} reserved); "
                        "raise MAX_SEARCH_CREDITS",
                        started,
                    )
                try:
                    response = await self.provider.search(
                        self._client, query.text, options, query_id=query.id
                    )
                except SearchAuthError as exc:
                    # Credentials and credit exhaustion will not fix themselves.
                    self._record_failure(exc, started)
                    log.error("search_failed", query_id=query.id, error=str(exc), retryable=False)
                    return self._failure(query, str(exc), started)
                except SearchError as exc:
                    last_error = str(exc)
                    if not exc.retryable or attempt == self._max_attempts:
                        self._record_failure(exc, started)
                        log.warning(
                            "search_failed",
                            query_id=query.id,
                            error=last_error,
                            attempts=attempt,
                        )
                        return self._failure(query, last_error, started)
                    self.stats.retries += 1
                    await asyncio.sleep(self._backoff(attempt, exc))
                    continue

                self.stats.calls += 1
                await self._settle_credits(reserved=cost, actual=response.credits)
                self.stats.credits += response.credits
                self.stats.latency_s += response.latency_s
                self.stats.results += len(response.results)
                log.info(
                    "search_completed",
                    query_id=query.id,
                    query=query.text[:80],
                    results=len(response.results),
                    depth=options.depth,
                    latency_s=response.latency_s,
                )
                return response

            return self._failure(query, last_error, started)

    def _backoff(self, attempt: int, exc: SearchError) -> float:
        """Exponential backoff with jitter.

        Jitter is not decoration: a round fans out several queries at once, and
        without it they retry in lockstep and re-trigger the same rate limit.
        """
        if isinstance(exc, SearchRateLimitError) and exc.retry_after:
            return min(exc.retry_after, 30.0)
        return min(0.5 * (2 ** (attempt - 1)), 8.0) * (0.5 + random.random())

    def _record_failure(self, exc: SearchError, started: float) -> None:
        self.stats.failures += 1
        self.stats.latency_s += time.perf_counter() - started
        key = type(exc).__name__
        self.stats.by_error[key] = self.stats.by_error.get(key, 0) + 1

    async def _reserve_credits(self, cost: float) -> bool:
        """Claim this call's credits, or refuse it.

        The whole check-and-claim happens under one lock, so two workers
        cannot both see the last credit as available. Reservation is the
        figure the ceiling is enforced against; ``stats.credits`` remains
        the record of what was actually spent.
        """
        ceiling = self.settings.max_search_credits
        async with self._credit_lock:
            # 0 means unlimited, matching the cloud budget convention.
            if ceiling and self._reserved_credits + cost > ceiling:
                log.warning(
                    "search_budget_exhausted",
                    reserved=self._reserved_credits,
                    ceiling=ceiling,
                )
                return False
            self._reserved_credits += cost
            return True

    async def _settle_credits(self, *, reserved: float, actual: float) -> None:
        """Give back the difference when a call cost less than estimated.

        Only ever releases, never claims: a call that cost more than its
        estimate keeps the larger figure reserved, so the ceiling stays a
        ceiling rather than becoming an average.
        """
        if actual >= reserved:
            return
        async with self._credit_lock:
            self._reserved_credits = max(0.0, self._reserved_credits - (reserved - actual))

    def _failure(self, query: SearchQuery, error: str, started: float) -> SearchResponse:
        return SearchResponse(
            query_id=query.id,
            query_text=query.text,
            provider=self.provider.name,
            results=[],
            latency_s=round(time.perf_counter() - started, 3),
            error=error,
        )
