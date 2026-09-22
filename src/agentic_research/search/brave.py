"""Brave Search provider.

Exists to keep :class:`~agentic_research.search.base.SearchProvider` honest. An
interface with exactly one implementation is a guess about what varies; a
second implementation is where you find out whether the abstraction actually
holds. Adding Brave required no change to the graph, the state or the evidence
pipeline, which is the result the design was aiming for.

Scope note: this is covered by normalisation tests against recorded payloads,
not against the live Brave API. See tests/unit/test_search_providers.py.

API shape: GET https://api.search.brave.com/res/v1/web/search with an
``X-Subscription-Token`` header.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import httpx

from agentic_research.models import SearchResult
from agentic_research.search.base import (
    SearchAuthError,
    SearchError,
    SearchOptions,
    SearchProvider,
    SearchRateLimitError,
    SearchResponse,
)

_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# Brave has no notion of search depth; every call costs the same. The engine's
# escalation logic still runs, it just has no effect here.
_FRESHNESS = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}


class BraveProvider(SearchProvider):
    name = "brave"

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        if not api_key:
            raise SearchAuthError(self.name, "BRAVE_API_KEY is not set")
        self._api_key = api_key
        self._timeout = timeout

    def credits_for(self, options: SearchOptions) -> float:
        return 1.0

    async def search(
        self,
        client: httpx.AsyncClient,
        query_text: str,
        options: SearchOptions,
        *,
        query_id: str = "",
    ) -> SearchResponse:
        started = time.perf_counter()
        params: dict[str, Any] = {
            "q": query_text,
            "count": min(options.max_results, 20),
            "result_filter": "web",
        }
        if options.time_range and options.time_range in _FRESHNESS:
            params["freshness"] = _FRESHNESS[options.time_range]

        try:
            response = await client.get(
                _ENDPOINT,
                params=params,
                headers={
                    "X-Subscription-Token": self._api_key,
                    "Accept": "application/json",
                },
                timeout=self._timeout,
            )
            if response.status_code in (401, 403):
                raise SearchAuthError(self.name, f"rejected credentials ({response.status_code})")
            if response.status_code == 429:
                raise SearchRateLimitError(self.name, "rate limited")
            if response.status_code >= 400:
                raise SearchError(
                    self.name,
                    f"HTTP {response.status_code}: {response.text[:200]}",
                    retryable=response.status_code >= 500,
                )
            payload = response.json()
        except SearchError:
            raise
        except httpx.TimeoutException as exc:
            raise SearchError(
                self.name, f"timed out after {self._timeout}s", retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise SearchError(self.name, f"transport error: {exc}", retryable=True) from exc
        except ValueError as exc:
            raise SearchError(self.name, "response was not valid JSON", retryable=True) from exc

        return SearchResponse(
            query_id=query_id,
            query_text=query_text,
            provider=self.name,
            results=self._normalise(payload, query_id=query_id, query_text=query_text),
            latency_s=round(time.perf_counter() - started, 3),
            credits=1.0,
        )

    def _normalise(
        self, payload: dict[str, Any], *, query_id: str, query_text: str
    ) -> list[SearchResult]:
        web = payload.get("web")
        items = web.get("results") if isinstance(web, dict) else None
        if not isinstance(items, list):
            return []

        results: list[SearchResult] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            # Brave returns no relevance score, so rank position stands in for
            # one. Normalising it to 0..1 keeps it comparable with Tavily's.
            rank_score = 1.0 - (len(results) / max(len(items), 1))
            results.append(
                SearchResult(
                    url=url,
                    title=item.get("title") or url,
                    snippet=item.get("description") or "",
                    score=round(rank_score, 3),
                    published_date=_parse_age(item.get("page_age")),
                    raw_content=None,  # Brave returns snippets only
                    provider=self.name,
                    query_id=query_id,
                    query_text=query_text,
                )
            )
        return results


def _parse_age(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
