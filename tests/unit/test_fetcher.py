"""Page fetching failure modes.

Each case here is something the open web does routinely. The requirement is
identical for all of them: classify the failure and keep going.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from agentic_research.config import Settings
from agentic_research.models import FetchStatus
from agentic_research.retrieval import PageFetcher

GOOD_HTML = (
    "<html><body><article><h1>Imbalanced Data</h1>"
    "<p>Fraud detection datasets are severely imbalanced, with positives often "
    "well under one percent of all transactions recorded by the system.</p>"
    "<p>Resampling and cost-sensitive objectives are the usual responses.</p>"
    "</article></body></html>"
)


class TestFetchSuccess:
    @respx.mock
    async def test_extracts_text_and_drops_markup(self, settings: Settings) -> None:
        respx.get("https://ex.com/a").mock(
            return_value=httpx.Response(
                200, html=GOOD_HTML, headers={"content-type": "text/html; charset=utf-8"}
            )
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://ex.com/a")

        assert result.ok
        assert result.status is FetchStatus.OK
        assert "severely imbalanced" in result.text
        assert "<p>" not in result.text
        assert fetcher.stats.succeeded == 1

    @respx.mock
    async def test_follows_redirect_and_records_final_url(self, settings: Settings) -> None:
        respx.get("https://ex.com/old").mock(
            return_value=httpx.Response(301, headers={"location": "https://ex.com/new"})
        )
        respx.get("https://ex.com/new").mock(
            return_value=httpx.Response(200, html=GOOD_HTML, headers={"content-type": "text/html"})
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://ex.com/old")

        assert result.ok
        assert result.final_url == "https://ex.com/new"


class TestFetchFailureModes:
    @respx.mock
    async def test_http_error_is_classified_not_raised(self, settings: Settings) -> None:
        respx.get("https://ex.com/404").mock(return_value=httpx.Response(404))
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://ex.com/404")
        assert result.status is FetchStatus.HTTP_ERROR
        assert result.http_status == 404
        assert not result.ok

    @respx.mock
    async def test_timeout(self, settings: Settings) -> None:
        respx.get("https://ex.com/slow").mock(side_effect=httpx.ConnectTimeout("too slow"))
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://ex.com/slow")
        assert result.status is FetchStatus.TIMEOUT

    @respx.mock
    async def test_redirect_loop(self, settings: Settings) -> None:
        respx.get("https://ex.com/loop").mock(side_effect=httpx.TooManyRedirects("loop"))
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://ex.com/loop")
        assert result.status is FetchStatus.HTTP_ERROR
        assert result.error == "redirect loop"

    @respx.mock
    async def test_non_text_binary_is_unsupported(self, settings: Settings) -> None:
        respx.get("https://ex.com/i.png").mock(
            return_value=httpx.Response(
                200, content=b"\x89PNG\r\n", headers={"content-type": "image/png"}
            )
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://ex.com/i.png")
        assert result.status is FetchStatus.UNSUPPORTED_TYPE
        assert result.text == ""

    @respx.mock
    async def test_oversized_declared_length_rejected(self, settings: Settings) -> None:
        respx.get("https://ex.com/big").mock(
            return_value=httpx.Response(
                200,
                html="<html/>",
                headers={"content-type": "text/html", "content-length": "99999999"},
            )
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://ex.com/big")
        assert result.status is FetchStatus.TOO_LARGE

    @respx.mock
    async def test_oversized_body_rejected_when_length_header_lies(
        self, settings: Settings
    ) -> None:
        settings.max_page_bytes = 10_000
        respx.get("https://ex.com/liar").mock(
            return_value=httpx.Response(
                200,
                content=b"<html><body>" + b"x" * 50_000 + b"</body></html>",
                headers={"content-type": "text/html"},  # no content-length
            )
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://ex.com/liar")
        assert result.status is FetchStatus.TOO_LARGE

    @respx.mock
    async def test_javascript_shell_yields_empty_not_success(self, settings: Settings) -> None:
        respx.get("https://ex.com/spa").mock(
            return_value=httpx.Response(
                200,
                html="<html><body><div id='root'></div><script>app()</script></body></html>",
                headers={"content-type": "text/html"},
            )
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://ex.com/spa")
        assert result.status is FetchStatus.EMPTY
        assert not result.ok

    @respx.mock
    async def test_connection_error(self, settings: Settings) -> None:
        respx.get("https://nope.invalid/x").mock(side_effect=httpx.ConnectError("no dns"))
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://nope.invalid/x")
        assert result.status is FetchStatus.HTTP_ERROR
        assert not result.ok

    @respx.mock
    async def test_mismatched_encoding_does_not_crash(self, settings: Settings) -> None:
        body = "<html><body><article><p>café serves fraud analytics daily to many people here</p></article></body></html>".encode()
        respx.get("https://ex.com/enc").mock(
            return_value=httpx.Response(
                200, content=body, headers={"content-type": "text/html; charset=ascii"}
            )
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://ex.com/enc")
        assert result.status in (FetchStatus.OK, FetchStatus.EMPTY)


class TestConcurrencyLimits:
    @respx.mock
    async def test_per_host_limit_is_enforced(self, settings: Settings) -> None:
        settings.max_parallel_fetches = 10
        live = 0
        peak = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.05)
            live -= 1
            return httpx.Response(200, html=GOOD_HTML, headers={"content-type": "text/html"})

        respx.get(url__startswith="https://same.com/").mock(side_effect=handler)
        async with PageFetcher(settings, per_host_limit=2) as fetcher:
            await asyncio.gather(*(fetcher.fetch(f"https://same.com/{i}") for i in range(6)))
        assert peak <= 2, f"per-host limit exceeded: {peak} concurrent"

    async def test_requires_context_manager(self, settings: Settings) -> None:
        with pytest.raises(RuntimeError, match="context manager"):
            await PageFetcher(settings).fetch("https://x.com")
