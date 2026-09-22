"""Tavily search provider.

Talks to the REST endpoint through ``httpx`` rather than the ``tavily-python``
SDK. Three reasons: the async path stays first-class, the dependency list stays
shorter, and normalising the raw payload here is exactly the boundary this
module exists to own — wrapping an SDK that already hides the payload would
make the abstraction harder to verify, not easier.

API shape confirmed against https://docs.tavily.com (retrieved 2026-09-22):
POST https://api.tavily.com/search, bearer auth, ``basic`` costs 1 credit and
``advanced`` costs 2.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import httpx

from agentic_research.models import SearchResult
from agentic_research.observability import get_logger
from agentic_research.search.base import (
    SearchAuthError,
    SearchError,
    SearchOptions,
    SearchProvider,
    SearchRateLimitError,
    SearchResponse,
)

log = get_logger(__name__)

_ENDPOINT = "https://api.tavily.com/search"
_CREDITS = {"basic": 1.0, "advanced": 2.0}


def _parse_date(value: Any) -> datetime | None:
    """Tavily dates arrive in several shapes and are not worth failing over."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    for parse in (
        datetime.fromisoformat,
        lambda s: datetime.strptime(s, "%a, %d %b %Y %H:%M:%S %z"),
        lambda s: datetime.strptime(s, "%Y-%m-%d"),
    ):
        try:
            parsed = parse(text)
        except (ValueError, TypeError):
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


class TavilyProvider(SearchProvider):
    name = "tavily"

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        if not api_key:
            raise SearchAuthError(self.name, "TAVILY_API_KEY is not set")
        self._api_key = api_key
        self._timeout = timeout

    def credits_for(self, options: SearchOptions) -> float:
        return _CREDITS.get(options.depth, 1.0)

    def _payload(self, query_text: str, options: SearchOptions) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query_text,
            "search_depth": options.depth,
            "max_results": options.max_results,
            "topic": options.topic,
            # Asking Tavily for page text here often removes the need for a
            # separate fetch of the same URL later, which is both faster and
            # politer to the origin server.
            "include_raw_content": "markdown" if options.include_raw_content else False,
            "include_published_date": True,
            "include_answer": False,
            "include_images": False,
        }
        if options.time_range:
            payload["time_range"] = options.time_range
        if options.include_domains:
            payload["include_domains"] = options.include_domains
        if options.exclude_domains:
            payload["exclude_domains"] = options.exclude_domains
        return payload

    async def search(
        self,
        client: httpx.AsyncClient,
        query_text: str,
        options: SearchOptions,
        *,
        query_id: str = "",
    ) -> SearchResponse:
        started = time.perf_counter()
        try:
            response = await client.post(
                _ENDPOINT,
                json=self._payload(query_text, options),
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
            self._raise_for_status(response)
            payload = response.json()
        except SearchError:
            raise
        except httpx.TimeoutException as exc:
            raise SearchError(self.name, f"timed out after {self._timeout}s", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise SearchError(self.name, f"transport error: {exc}", retryable=True) from exc
        except ValueError as exc:
            raise SearchError(self.name, "response was not valid JSON", retryable=True) from exc

        results = self._normalise(payload, query_id=query_id, query_text=query_text)
        elapsed = round(time.perf_counter() - started, 3)
        log.debug(
            "search_provider_ok",
            provider=self.name,
            query_id=query_id,
            results=len(results),
            depth=options.depth,
            latency_s=elapsed,
        )
        return SearchResponse(
            query_id=query_id,
            query_text=query_text,
            provider=self.name,
            results=results,
            latency_s=elapsed,
            credits=self.credits_for(options),
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        detail = response.text[:200]
        if response.status_code in (401, 403):
            raise SearchAuthError(self.name, f"rejected credentials ({response.status_code})")
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise SearchRateLimitError(
                self.name,
                "rate limited",
                retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None,
            )
        if response.status_code == 432:
            # Tavily uses this for plan/credit exhaustion; retrying cannot help.
            raise SearchAuthError(self.name, "account is out of search credits")
        raise SearchError(
            self.name,
            f"HTTP {response.status_code}: {detail}",
            retryable=response.status_code >= 500,
        )

    def _normalise(
        self, payload: dict[str, Any], *, query_id: str, query_text: str
    ) -> list[SearchResult]:
        """Convert Tavily's payload into the engine's own result type.

        Defensive by design: a provider adding or renaming a field should cost
        us one missing attribute, not a crashed research round.
        """
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            return []

        normalised: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            raw_content = item.get("raw_content")
            normalised.append(
                SearchResult(
                    url=url,
                    title=item.get("title") or url,
                    snippet=item.get("content") or "",
                    score=_as_float(item.get("score")),
                    published_date=_parse_date(item.get("published_date")),
                    raw_content=raw_content if isinstance(raw_content, str) else None,
                    provider=self.name,
                    query_id=query_id,
                    query_text=query_text,
                )
            )
        return normalised


def _as_float(value: Any) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
