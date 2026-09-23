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
from agentic_research.retrieval.safety import (
    UnresolvableHostError,
    UnsafeURLError,
    validate_url,
)
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

    def __init__(
        self, settings: Settings, *, per_host_limit: int = 2, max_redirects: int = 5
    ) -> None:
        self.settings = settings
        self._max_redirects = max_redirects
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
            # Redirects are handled in _fetch_inner so each hop is revalidated.
            follow_redirects=False,
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
            except UnresolvableHostError as exc:
                # DNS failure, not a policy refusal. Reported as such so the
                # BLOCKED count stays a meaningful security signal.
                result = FetchResult(url=url, status=FetchStatus.HTTP_ERROR, error=exc.reason)
            except UnsafeURLError as exc:
                # Refused by the outbound policy before any connection.
                log.warning("url_blocked", url=url[:120], reason=exc.reason)
                result = FetchResult(url=url, status=FetchStatus.BLOCKED, error=exc.reason)
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
        """Follow redirects manually, revalidating every hop.

        ``follow_redirects=True`` would let a public URL bounce the client
        into a private address or the cloud metadata endpoint with no second
        check, which is the standard SSRF redirect bypass.
        """
        current = url
        for _ in range(self._max_redirects + 1):
            validate_url(current)
            result, redirect_to = await self._fetch_once(current)
            if redirect_to is None:
                return result
            current = redirect_to
        return FetchResult(
            url=url,
            final_url=current,
            status=FetchStatus.HTTP_ERROR,
            error=f"exceeded {self._max_redirects} redirects",
        )

    async def _fetch_once(self, url: str) -> tuple[FetchResult, str | None]:
        """Issue one request.

        Returns ``(result, next_url)``. ``next_url`` is set only for a
        redirect, and the caller must revalidate it before following.
        """
        assert self._client is not None

        def failure(status: FetchStatus, error: str, **extra: object) -> tuple[FetchResult, None]:
            return FetchResult(url=url, status=status, error=error, **extra), None  # type: ignore[arg-type]

        async with self._client.stream("GET", url) as response:
            final_url = str(response.url)
            code = response.status_code

            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    return failure(
                        FetchStatus.HTTP_ERROR,
                        "redirect without a location header",
                        final_url=final_url,
                        http_status=code,
                    )
                return FetchResult(url=url), str(response.url.join(location))

            if code >= 400:
                return failure(
                    FetchStatus.HTTP_ERROR,
                    f"HTTP {code}",
                    final_url=final_url,
                    http_status=code,
                )

            content_type = response.headers.get("content-type", "").lower()
            if content_type and not any(t in content_type for t in _TEXTUAL_TYPES):
                return failure(
                    FetchStatus.UNSUPPORTED_TYPE,
                    f"unsupported content-type: {content_type.split(';')[0]}",
                    final_url=final_url,
                    http_status=code,
                )

            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > self.settings.max_page_bytes:
                return failure(
                    FetchStatus.TOO_LARGE,
                    f"content-length {declared} exceeds cap",
                    final_url=final_url,
                    http_status=code,
                )

            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > self.settings.max_page_bytes:
                    # Servers lie about or omit content-length, so the cap is
                    # also enforced against bytes actually received.
                    return failure(
                        FetchStatus.TOO_LARGE,
                        f"body exceeded {self.settings.max_page_bytes} bytes",
                        final_url=final_url,
                        http_status=code,
                        bytes_read=total,
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
            return failure(
                FetchStatus.EMPTY,
                "no extractable text",
                final_url=final_url,
                http_status=code,
                bytes_read=total,
            )

        return (
            FetchResult(
                url=url,
                final_url=final_url,
                status=FetchStatus.OK,
                text=text,
                http_status=code,
                bytes_read=total,
                content_origin=ContentOrigin.HTML_FETCH,
            ),
            None,
        )
