"""Search provider normalisation and failure handling.

All HTTP is mocked with respx: the point is to prove that *our* translation of
a vendor payload is correct, which does not require spending credits.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from agentic_research.models import SearchQuery
from agentic_research.search import (
    BraveProvider,
    SearchAuthError,
    SearchError,
    SearchOptions,
    SearchService,
    TavilyProvider,
)

TAVILY_URL = "https://api.tavily.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

TAVILY_PAYLOAD = {
    "query": "imbalanced fraud detection",
    "results": [
        {
            "title": "Handling Class Imbalance",
            "url": "https://example.org/imbalance",
            "content": "SMOTE and cost-sensitive learning are common.",
            "score": 0.91,
            "published_date": "2025-04-11T00:00:00Z",
            "raw_content": "# Handling Class Imbalance\nFull page text here.",
        },
        {
            "title": "No date here",
            "url": "https://example.net/b",
            "content": "snippet",
            "score": "0.5",
        },
        {"title": "bad", "url": "ftp://nope/x", "content": "skipped"},
        "not-a-dict",
    ],
    "response_time": 1.2,
}


@pytest.fixture
def options() -> SearchOptions:
    return SearchOptions(max_results=5, depth="basic")


class TestTavilyNormalisation:
    @respx.mock
    async def test_normalises_payload_and_drops_junk(self, options: SearchOptions) -> None:
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json=TAVILY_PAYLOAD))
        provider = TavilyProvider("tvly-x")
        async with httpx.AsyncClient() as client:
            response = await provider.search(client, "q", options, query_id="Q1")

        assert response.ok
        # The ftp URL and the non-dict entry are dropped, not crashed on.
        assert len(response.results) == 2
        first = response.results[0]
        assert first.url == "https://example.org/imbalance"
        assert first.score == 0.91
        assert first.published_date is not None and first.published_date.year == 2025
        assert first.raw_content is not None
        assert first.query_id == "Q1"
        assert first.provider == "tavily"
        # A string score is coerced; a missing date stays None rather than failing.
        assert response.results[1].score == 0.5
        assert response.results[1].published_date is None

    @respx.mock
    async def test_requests_markdown_content_and_correct_depth(
        self, options: SearchOptions
    ) -> None:
        route = respx.post(TAVILY_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        provider = TavilyProvider("tvly-x")
        async with httpx.AsyncClient() as client:
            await provider.search(client, "q", SearchOptions(depth="advanced"), query_id="Q1")

        body = route.calls[0].request.content.decode()
        assert '"search_depth":"advanced"' in body.replace(" ", "")
        assert '"include_raw_content":"markdown"' in body.replace(" ", "")
        assert route.calls[0].request.headers["authorization"] == "Bearer tvly-x"

    @respx.mock
    async def test_missing_results_key_is_empty_not_error(self, options: SearchOptions) -> None:
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={"foo": "bar"}))
        provider = TavilyProvider("tvly-x")
        async with httpx.AsyncClient() as client:
            response = await provider.search(client, "q", options)
        assert response.results == []

    @pytest.mark.parametrize(
        ("status", "expected", "retryable"),
        [
            (401, SearchAuthError, False),
            (432, SearchAuthError, False),
            (429, SearchError, True),
            (500, SearchError, True),
            (400, SearchError, False),
        ],
    )
    @respx.mock
    async def test_status_codes_map_to_right_error(
        self, status: int, expected: type[SearchError], retryable: bool, options: SearchOptions
    ) -> None:
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(status, text="nope"))
        provider = TavilyProvider("tvly-x")
        async with httpx.AsyncClient() as client:
            with pytest.raises(expected) as info:
                await provider.search(client, "q", options)
        assert info.value.retryable is retryable

    def test_credits_reflect_depth(self) -> None:
        provider = TavilyProvider("tvly-x")
        assert provider.credits_for(SearchOptions(depth="basic")) == 1.0
        assert provider.credits_for(SearchOptions(depth="advanced")) == 2.0

    def test_missing_key_rejected_at_construction(self) -> None:
        with pytest.raises(SearchAuthError):
            TavilyProvider("")


class TestBraveNormalisation:
    """Brave exists to prove the interface holds for a differently shaped API."""

    @respx.mock
    async def test_normalises_to_same_shape(self, options: SearchOptions) -> None:
        respx.get(BRAVE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "web": {
                        "results": [
                            {
                                "title": "A",
                                "url": "https://a.example/1",
                                "description": "desc a",
                                "page_age": "2025-01-02T00:00:00Z",
                            },
                            {"title": "B", "url": "https://b.example/2", "description": "desc b"},
                        ]
                    }
                },
            )
        )
        provider = BraveProvider("brave-key")
        async with httpx.AsyncClient() as client:
            response = await provider.search(client, "q", options, query_id="Q9")

        assert [r.url for r in response.results] == [
            "https://a.example/1",
            "https://b.example/2",
        ]
        assert all(r.provider == "brave" for r in response.results)
        # Brave has no score, so rank position is normalised into one.
        assert response.results[0].score > response.results[1].score
        assert response.results[0].query_id == "Q9"

    @respx.mock
    async def test_auth_header_and_missing_web_key(self, options: SearchOptions) -> None:
        route = respx.get(BRAVE_URL).mock(return_value=httpx.Response(200, json={}))
        provider = BraveProvider("brave-key")
        async with httpx.AsyncClient() as client:
            response = await provider.search(client, "q", options)
        assert response.results == []
        assert route.calls[0].request.headers["x-subscription-token"] == "brave-key"


class TestSearchServiceResilience:
    """One bad query must not be able to end a research round."""

    @respx.mock
    async def test_retries_then_succeeds(self, settings) -> None:  # noqa: ANN001
        respx.post(TAVILY_URL).mock(
            side_effect=[
                httpx.Response(500, text="server error"),
                httpx.Response(200, json=TAVILY_PAYLOAD),
            ]
        )
        service = SearchService(TavilyProvider("tvly-x"), settings, max_attempts=3)
        async with service:
            response = await service.run_query(
                SearchQuery(id="Q1", sub_question_id="SQ1", text="q", round_number=1)
            )
        assert response.ok
        assert service.stats.retries == 1
        assert service.stats.calls == 1

    @respx.mock
    async def test_exhausted_retries_return_error_not_raise(self, settings) -> None:  # noqa: ANN001
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(500, text="down"))
        service = SearchService(TavilyProvider("tvly-x"), settings, max_attempts=2)
        async with service:
            response = await service.run_query(
                SearchQuery(id="Q1", sub_question_id="SQ1", text="q", round_number=1)
            )
        assert not response.ok
        assert response.results == []
        assert service.stats.failures == 1

    @respx.mock
    async def test_auth_error_is_not_retried(self, settings) -> None:  # noqa: ANN001
        route = respx.post(TAVILY_URL).mock(return_value=httpx.Response(401, text="bad key"))
        service = SearchService(TavilyProvider("tvly-x"), settings, max_attempts=3)
        async with service:
            response = await service.run_query(
                SearchQuery(id="Q1", sub_question_id="SQ1", text="q", round_number=1)
            )
        assert not response.ok
        assert route.call_count == 1, "auth failures must not burn retries"

    @respx.mock
    async def test_followup_round_escalates_search_depth(self, settings) -> None:  # noqa: ANN001
        route = respx.post(TAVILY_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        service = SearchService(TavilyProvider("tvly-x"), settings)
        async with service:
            await service.run_query(
                SearchQuery(id="Q1", sub_question_id="SQ1", text="q", round_number=1)
            )
            await service.run_query(
                SearchQuery(id="Q2", sub_question_id="SQ1", text="q2", round_number=2)
            )
        bodies = [c.request.content.decode().replace(" ", "") for c in route.calls]
        assert '"search_depth":"basic"' in bodies[0]
        assert '"search_depth":"advanced"' in bodies[1], "round 2 should escalate depth"

    async def test_requires_context_manager(self, settings) -> None:  # noqa: ANN001
        service = SearchService(TavilyProvider("tvly-x"), settings)
        with pytest.raises(RuntimeError, match="context manager"):
            await service.run_query(
                SearchQuery(id="Q1", sub_question_id="SQ1", text="q", round_number=1)
            )
