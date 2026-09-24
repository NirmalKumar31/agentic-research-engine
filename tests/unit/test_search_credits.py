"""Search credits are reserved before dispatch, not counted after it.

The old order was: read `stats.credits`, compare against the ceiling,
dispatch, and increment on completion. The entire HTTP request sat
between the read and the increment, so every parallel worker saw the
same remaining budget, every one of them passed the check, and the
ceiling was exceeded by however many were in flight.

Six searches run concurrently by default, so the overshoot was up to six
calls' worth of a budget that is money.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentic_research.config import Settings
from agentic_research.models import SearchQuery
from agentic_research.search.base import SearchOptions, SearchProvider, SearchResponse
from agentic_research.search.service import SearchService


class CountingProvider(SearchProvider):
    """Records how many calls actually reached the provider."""

    name = "counting"

    def __init__(self, *, credits: float = 1.0, delay: float = 0.02) -> None:
        self.credits = credits
        self.delay = delay
        self.dispatched = 0

    async def search(
        self,
        client: Any,
        query_text: str,
        options: SearchOptions,
        *,
        query_id: str = "",
    ) -> SearchResponse:
        self.dispatched += 1
        # Long enough that every concurrent worker is inside the window
        # where the old code had already passed its budget check.
        await asyncio.sleep(self.delay)
        return SearchResponse(
            query_id=query_id,
            query_text=query_text,
            provider=self.name,
            results=[],
            credits=self.credits,
            latency_s=self.delay,
        )

    def credits_for(self, options: SearchOptions) -> float:
        return self.credits


def settings(ceiling: float, parallel: int = 6) -> Settings:
    return Settings(
        llm_mode="local",
        tavily_api_key="tvly-test-key",
        max_search_credits=ceiling,
        max_parallel_searches=parallel,
        _env_file=None,
    )


def query(n: int) -> SearchQuery:
    return SearchQuery(id=f"Q{n}", sub_question_id="SQ1", text=f"query {n}", round_number=1)


class TestTheCeilingHoldsUnderConcurrency:
    async def test_simultaneous_searches_cannot_overshoot(self) -> None:
        """Ten workers, three credits, one credit each. Three dispatch."""
        provider = CountingProvider(credits=1.0)
        async with SearchService(provider, settings(ceiling=3.0)) as service:
            await asyncio.gather(*(service.run_query(query(i)) for i in range(10)))

        assert provider.dispatched == 3, "the ceiling was exceeded by in-flight calls"
        assert service.stats.credits <= 3.0

    async def test_the_last_credit_goes_to_exactly_one_worker(self) -> None:
        """The narrowest case: several workers compete for one credit."""
        provider = CountingProvider(credits=1.0)
        async with SearchService(provider, settings(ceiling=1.0)) as service:
            await asyncio.gather(*(service.run_query(query(i)) for i in range(6)))

        assert provider.dispatched == 1

    async def test_refused_searches_fail_without_dispatching(self) -> None:
        provider = CountingProvider(credits=1.0)
        async with SearchService(provider, settings(ceiling=2.0)) as service:
            results = await asyncio.gather(*(service.run_query(query(i)) for i in range(5)))

        failed = [r for r in results if r.error]
        assert provider.dispatched == 2
        assert len(failed) == 3
        assert all("credit budget exhausted" in (r.error or "") for r in failed)

    async def test_a_fractional_cost_is_still_bounded(self) -> None:
        provider = CountingProvider(credits=0.5)
        async with SearchService(provider, settings(ceiling=2.0)) as service:
            await asyncio.gather(*(service.run_query(query(i)) for i in range(10)))

        assert provider.dispatched == 4
        assert service.stats.credits <= 2.0

    async def test_zero_means_unlimited(self) -> None:
        """Matching the cloud budget convention."""
        provider = CountingProvider(credits=1.0)
        async with SearchService(provider, settings(ceiling=0.0)) as service:
            await asyncio.gather(*(service.run_query(query(i)) for i in range(8)))

        assert provider.dispatched == 8

    async def test_an_overestimated_call_returns_the_difference(self) -> None:
        """Reservation is the estimate; settling releases what was not
        used, so a cheap call does not permanently hold a costly slot."""
        provider = CountingProvider(credits=0.25)  # estimate matches, then costs less
        async with SearchService(provider, settings(ceiling=1.0)) as service:
            await service.run_query(query(1))
            assert service.stats.credits == pytest.approx(0.25)
            # Three more fit inside the ceiling.
            await asyncio.gather(*(service.run_query(query(i)) for i in range(2, 5)))

        assert provider.dispatched == 4


class TestSearchCreditsAreNotProviderRequests:
    def test_the_two_budgets_are_separate_settings(self) -> None:
        """Search credits are metered by the search provider; provider
        requests bound calls to the model provider. Conflating them would
        let one exhaust the other."""
        s = settings(ceiling=8.0)
        assert s.max_search_credits == 8.0
        assert s.max_provider_requests > 0

    def test_search_does_not_consume_the_model_request_budget(self) -> None:
        """UsageTracker is constructed by the router for model calls; the
        search service never reserves against it."""
        import inspect

        from agentic_research.search import service as service_module

        source = inspect.getsource(service_module)
        assert "UsageTracker" not in source
        assert "reserve_provider_request" not in source
