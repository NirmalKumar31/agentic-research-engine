"""Search provider contract.

The research graph knows about :class:`SearchProvider`, ``SearchOptions`` and
``SearchResult`` — never about a vendor's payload shape. Swapping Tavily for
Brave is then a registry change, not a graph change.

Providers are async and take an ``httpx.AsyncClient`` from the caller. Sharing
one client across the whole run means connection reuse across the dozens of
concurrent searches a single round issues.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx

from agentic_research.models import SearchResult


class SearchError(Exception):
    """A search call failed in a way the caller may want to distinguish."""

    def __init__(self, provider: str, message: str, *, retryable: bool = False) -> None:
        self.provider = provider
        self.retryable = retryable
        super().__init__(f"[{provider}] {message}")


class SearchAuthError(SearchError):
    """Bad or missing API key. Never retryable — retrying just burns time."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(provider, message, retryable=False)


class SearchRateLimitError(SearchError):
    def __init__(self, provider: str, message: str, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(provider, message, retryable=True)


@dataclass(slots=True)
class SearchOptions:
    """Per-call search parameters.

    ``depth`` is the cost lever. Providers charge more for deeper search, so
    the engine starts shallow and escalates only for follow-up rounds, where
    the first pass has already shown the easy results are not enough.
    """

    max_results: int = 8
    depth: str = "basic"
    include_raw_content: bool = True
    time_range: str | None = None
    topic: str = "general"
    include_domains: list[str] = field(default_factory=list)
    exclude_domains: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SearchResponse:
    """Outcome of one query against one provider.

    A failed search is returned rather than raised so that one dead query
    cannot take down a whole parallel round.
    """

    query_id: str
    query_text: str
    provider: str
    results: list[SearchResult] = field(default_factory=list)
    latency_s: float = 0.0
    credits: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class SearchProvider(ABC):
    """Interface every search backend implements."""

    name: str = "unknown"

    @abstractmethod
    async def search(
        self,
        client: httpx.AsyncClient,
        query_text: str,
        options: SearchOptions,
        *,
        query_id: str = "",
    ) -> SearchResponse:
        """Run one query and return normalised results."""

    @abstractmethod
    def credits_for(self, options: SearchOptions) -> float:
        """Provider credits one attempt at these options may consume.

        An **upper bound**, not an estimate. Credits are reserved from
        this figure before the request is dispatched, so a value below
        what the provider actually charges means the ceiling did not bound
        that call. The service reconciles upward and logs a contract
        breach when that happens; it cannot un-spend the credits.
        """
