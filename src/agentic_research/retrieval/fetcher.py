"""Page fetching.

Scope is deliberately narrow: retrieve a handful of already-chosen URLs
politely and give up quickly on anything awkward. This is not a crawler. It
does not discover links, respect crawl delays across runs, or persist a
frontier, and it should not grow into something that does.

Real-world failure modes this handles, because all of them occur within the
first few dozen URLs of an ordinary research run: dead hosts, redirect loops,
slow servers, PDFs, 50 MB pages, pages that are pure JavaScript, and HTML
declaring one encoding while being served as another.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field

import httpx

from agentic_research.config import Settings
from agentic_research.models import ContentOrigin, FetchStatus
from agentic_research.observability import get_logger
from agentic_research.retrieval.parser import extract_main_text
from agentic_research.retrieval.urls import domain_of

log = get_logger(__name__)

_TEXTUAL_TYPES = ("text/html", "application/xhtml", "text/plain", "application/xml", "text/xml")


@dataclass(slots=True)
class FetchResult:
    url: str
    final_url: str = ""
    status: FetchStatus = FetchStatus.OK
    text: str = ""
    http_status: int | None = None
    error: str | None = None
    latency_s: float = 0.0
    bytes_read: int = 0
    content_origin: ContentOrigin = ContentOrigin.HTML_FETCH
    page_offsets: list[int] = field(default_factory=list)
    """For PDFs: character offset in ``text`` where each page begins."""

    @property
    def ok(self) -> bool:
        return self.status is FetchStatus.OK and bool(self.text.strip())


@dataclass
class FetchStats:
    attempted: int = 0
    succeeded: int = 0
    skipped_provider_content: int = 0
    latency_s: float = 0.0
    by_status: dict[str, int] = field(default_factory=dict)


class PageFetcher:
    """Fetches and extracts page text under global and per-host limits.

    The per-host limit is separate from the global one on purpose. A research
    round frequently returns six results from the same documentation site, and
    opening six simultaneous connections to one origin is both rude and a good
    way to get rate limited mid-run.
    """

    def __init__(self, settings: Settings, *, per_host_limit: int = 2) -> None:
        self.settings = settings
        self.stats = FetchStats()
        self._semaphore = asyncio.Semaphore(settings.max_parallel_fetches)
        self._per_host_limit = per_host_limit
        self._host_locks: dict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(per_host_limit)
        )
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> PageFetcher:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.fetch_timeout_seconds, connect=8.0),
            follow_redirects=True,
            max_redirects=5,
            headers={
                "User-Agent": self.settings.user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
                "Accept-Language": "en;q=0.9",
            },
            limits=httpx.Limits(max_connections=self.settings.max_parallel_fetches * 2),
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, url: str) -> FetchResult:
        """Retrieve one URL and extract its readable text.

        Always returns a result. Fetching is the single most failure-prone step
        in the pipeline, and a raised exception here would propagate out of a
        parallel worker and abort an entire research round.
        """
        if self._client is None:
            raise RuntimeError("PageFetcher must be used as an async context manager")

        started = time.perf_counter()
        self.stats.attempted += 1
        host = domain_of(url) or url

        async with self._semaphore, self._host_locks[host]:
            try:
                result = await self._fetch_inner(url)
            except httpx.TimeoutException:
                result = FetchResult(
                    url=url,
                    status=FetchStatus.TIMEOUT,
                    error=f"timed out after {self.settings.fetch_timeout_seconds}s",
                )
            except httpx.TooManyRedirects:
                result = FetchResult(url=url, status=FetchStatus.HTTP_ERROR, error="redirect loop")
            except httpx.HTTPError as exc:
                result = FetchResult(
                    url=url, status=FetchStatus.HTTP_ERROR, error=f"{type(exc).__name__}: {exc}"
                )
            except Exception as exc:
                result = FetchResult(
                    url=url, status=FetchStatus.ERROR, error=f"{type(exc).__name__}: {exc}"
                )

        result.latency_s = round(time.perf_counter() - started, 3)
        self.stats.latency_s += result.latency_s
        key = result.status.value
        self.stats.by_status[key] = self.stats.by_status.get(key, 0) + 1
        if result.ok:
            self.stats.succeeded += 1
            log.debug("source_retrieved", url=url[:120], words=len(result.text.split()))
        else:
            log.debug("source_fetch_failed", url=url[:120], status=key, error=result.error)
        return result

    async def _fetch_inner(self, url: str) -> FetchResult:
        assert self._client is not None
        # Streamed so an oversized body can be abandoned partway rather than
        # buffered in full first.
        async with self._client.stream("GET", url) as response:
            final_url = str(response.url)
            if response.status_code >= 400:
                return FetchResult(
                    url=url,
                    final_url=final_url,
                    status=FetchStatus.HTTP_ERROR,
                    http_status=response.status_code,
                    error=f"HTTP {response.status_code}",
                )

            content_type = response.headers.get("content-type", "").lower()
            if content_type and not any(t in content_type for t in _TEXTUAL_TYPES):
                return FetchResult(
                    url=url,
                    final_url=final_url,
                    status=FetchStatus.UNSUPPORTED_TYPE,
                    http_status=response.status_code,
                    error=f"unsupported content-type: {content_type.split(';')[0]}",
                )

            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > self.settings.max_page_bytes:
                return FetchResult(
                    url=url,
                    final_url=final_url,
                    status=FetchStatus.TOO_LARGE,
                    http_status=response.status_code,
                    error=f"content-length {declared} exceeds cap",
                )

            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > self.settings.max_page_bytes:
                    # Servers lie about or omit content-length, so the cap is
                    # also enforced against bytes actually received.
                    return FetchResult(
                        url=url,
                        final_url=final_url,
                        status=FetchStatus.TOO_LARGE,
                        http_status=response.status_code,
                        bytes_read=total,
                        error=f"body exceeded {self.settings.max_page_bytes} bytes",
                    )
                chunks.append(chunk)

        body = b"".join(chunks)
        encoding = response.encoding or "utf-8"
        try:
            html = body.decode(encoding, errors="replace")
        except (LookupError, UnicodeDecodeError):
            html = body.decode("utf-8", errors="replace")

        text = extract_main_text(html, url=final_url)
        if not text.strip():
            # Usually a JavaScript-rendered page or an interstitial. Headless
            # rendering would fix some of these and is out of scope.
            return FetchResult(
                url=url,
                final_url=final_url,
                status=FetchStatus.EMPTY,
                http_status=response.status_code,
                bytes_read=total,
                error="no extractable text",
            )

        return FetchResult(
            url=url,
            final_url=final_url,
            status=FetchStatus.OK,
            text=text,
            http_status=response.status_code,
            bytes_read=total,
        )
