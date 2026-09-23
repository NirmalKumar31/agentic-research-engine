"""Outbound URL policy (SSRF).

The fetcher follows URLs chosen by a search provider and then by whatever
those pages redirect to. None of that is trusted input, and a hosted
deployment without this policy will fetch the cloud metadata endpoint on
request.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from agentic_research.config import Settings
from agentic_research.models import FetchStatus
from agentic_research.retrieval import PageFetcher
from agentic_research.retrieval.safety import (
    UnresolvableHostError,
    UnsafeURLError,
    is_safe_url,
    validate_url,
)


class FakeDNS:
    """Deterministic stand-in for getaddrinfo, so no test needs a network."""

    def __init__(self) -> None:
        self._records: dict[str, list[str]] = {}

    def set(self, host: str, addresses: list[str]) -> None:
        self._records[host.lower()] = addresses

    def getaddrinfo(self, host: str, *_args: object, **_kwargs: object) -> list:
        import socket as _socket

        addresses = self._records.get((host or "").lower())
        if not addresses:
            raise _socket.gaierror(f"no fake record for {host}")
        return [(0, 0, 0, "", (address, 0)) for address in addresses]


@pytest.fixture
def fake_dns(monkeypatch: pytest.MonkeyPatch) -> FakeDNS:
    import agentic_research.retrieval.safety as safety

    dns = FakeDNS()
    monkeypatch.setattr(safety.socket, "getaddrinfo", dns.getaddrinfo)
    return dns


GOOD_HTML = (
    "<html><body><article><p>Imbalanced fraud datasets contain very few "
    "positive cases, which distorts accuracy as a metric for them.</p>"
    "</article></body></html>"
)


class TestBlockedDestinations:
    @pytest.mark.parametrize(
        ("url", "expected_reason"),
        [
            ("http://127.0.0.1/x", "loopback"),
            ("http://127.1.2.3/x", "loopback"),
            ("http://[::1]/x", "loopback"),
            ("http://10.0.0.5/x", "private"),
            ("http://10.255.255.254/x", "private"),
            ("http://172.16.0.1/x", "private"),
            ("http://172.31.255.254/x", "private"),
            ("http://192.168.1.1/x", "private"),
            ("http://[fd00::1]/x", "private"),
            ("http://[fc00::1]/x", "private"),
            ("http://169.254.1.1/x", "link-local"),
            ("http://[fe80::1]/x", "link-local"),
            ("http://224.0.0.1/x", "multicast"),
            ("http://[ff02::1]/x", "multicast"),
            ("http://0.0.0.0/x", "private"),
        ],
    )
    def test_private_and_local_ranges_are_refused(self, url: str, expected_reason: str) -> None:
        with pytest.raises(UnsafeURLError) as info:
            validate_url(url, resolve=False)
        assert expected_reason in info.value.reason

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://metadata/computeMetadata/v1/",
        ],
    )
    def test_cloud_metadata_endpoints_are_refused(self, url: str) -> None:
        """The specific thing SSRF is usually aimed at."""
        with pytest.raises(UnsafeURLError):
            validate_url(url, resolve=False)

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/x",
            "http://LOCALHOST/x",
            "http://foo.localhost/x",
            "http://db.internal/admin",
            "http://printer.local/x",
        ],
    )
    def test_local_hostnames_are_refused_without_needing_dns(self, url: str) -> None:
        """Blocked syntactically too, so the check does not depend on the
        resolver behaving."""
        with pytest.raises(UnsafeURLError, match="local"):
            validate_url(url, resolve=False)

    @pytest.mark.parametrize("url", ["http://[::ffff:10.0.0.1]/x", "http://[::ffff:127.0.0.1]/x"])
    def test_ipv6_mapped_ipv4_cannot_smuggle_a_private_address(self, url: str) -> None:
        with pytest.raises(UnsafeURLError):
            validate_url(url, resolve=False)


class TestSchemesPortsAndJunk:
    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://example.com/a",
            "gopher://example.com/",
            "data:text/html,<script>",
            "javascript:alert(1)",
        ],
    )
    def test_only_http_and_https_are_allowed(self, url: str) -> None:
        with pytest.raises(UnsafeURLError, match="scheme"):
            validate_url(url, resolve=False)

    @pytest.mark.parametrize("port", [22, 25, 3306, 5432, 6379, 11211])
    def test_non_web_ports_are_refused(self, port: int) -> None:
        with pytest.raises(UnsafeURLError, match="port"):
            validate_url(f"http://example.com:{port}/x", resolve=False)

    @pytest.mark.parametrize("url", ["", "   ", "not a url", "http://", "https:///path"])
    def test_malformed_urls_are_refused_not_crashed_on(self, url: str) -> None:
        with pytest.raises(UnsafeURLError):
            validate_url(url, resolve=False)

    def test_ordinary_public_urls_pass(self) -> None:
        target = validate_url("https://example.com/article?id=3", resolve=False)
        assert target.host == "example.com"
        assert target.port == 443

    def test_is_safe_url_is_a_boolean_wrapper(self) -> None:
        assert is_safe_url("https://example.com/", resolve=False)
        assert not is_safe_url("http://127.0.0.1/", resolve=False)


class TestResolution:
    """Resolution is stubbed so these stay hermetic; the policy logic under
    test is ours, not the resolver's."""

    def test_a_name_resolving_to_loopback_is_refused(self, fake_dns: FakeDNS) -> None:
        """A public-looking hostname pointing at 127.0.0.1 is the classic
        bypass for policies that only inspect the literal string."""
        fake_dns.set("sneaky.example", ["127.0.0.1"])
        with pytest.raises(UnsafeURLError, match="loopback"):
            validate_url("http://sneaky.example/x")

    def test_all_resolved_addresses_must_be_safe(self, fake_dns: FakeDNS) -> None:
        """One public and one private A record must not pass: which address
        the client picks is not ours to control."""
        fake_dns.set("mixed.example", ["93.184.216.34", "10.1.2.3"])
        with pytest.raises(UnsafeURLError, match="private"):
            validate_url("http://mixed.example/x")

    def test_a_fully_public_name_passes(self, fake_dns: FakeDNS) -> None:
        fake_dns.set("public.example", ["93.184.216.34"])
        assert validate_url("http://public.example/x").addresses == ("93.184.216.34",)

    def test_unresolvable_host_is_distinguished_from_a_policy_block(
        self, fake_dns: FakeDNS
    ) -> None:
        """DNS failure is a network problem; conflating the two would make
        the blocked count useless as a security signal."""
        with pytest.raises(UnresolvableHostError):
            validate_url("http://nowhere.example/x")


class TestFetcherEnforcement:
    async def test_fetcher_refuses_a_private_target(self, settings: Settings) -> None:
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("http://169.254.169.254/latest/meta-data/")
        assert result.status is FetchStatus.BLOCKED
        assert not result.ok
        assert result.text == ""

    @respx.mock
    async def test_a_public_url_redirecting_to_a_private_one_is_blocked(
        self, settings: Settings, fake_dns: FakeDNS
    ) -> None:
        """The redirect bypass. follow_redirects=True would have taken it."""
        fake_dns.set("public.example", ["93.184.216.34"])
        respx.get(path="/start").mock(
            return_value=httpx.Response(
                302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
            )
        )
        metadata = respx.get(path="/latest/meta-data/").mock(
            return_value=httpx.Response(200, text="SECRET-CREDENTIALS")
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://public.example/start")

        assert result.status is FetchStatus.BLOCKED
        assert metadata.call_count == 0, "the private target must never be contacted"
        assert "SECRET" not in result.text

    @respx.mock
    async def test_a_public_redirect_chain_still_works(
        self, settings: Settings, fake_dns: FakeDNS
    ) -> None:
        fake_dns.set("public.example", ["93.184.216.34"])
        respx.get(path="/a").mock(
            return_value=httpx.Response(301, headers={"location": "https://public.example/b"})
        )
        respx.get(path="/b").mock(
            return_value=httpx.Response(200, html=GOOD_HTML, headers={"content-type": "text/html"})
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://public.example/a")
        assert result.ok
        assert "positive cases" in result.text

    @respx.mock
    async def test_redirect_loops_terminate(self, settings: Settings, fake_dns: FakeDNS) -> None:
        fake_dns.set("public.example", ["93.184.216.34"])
        respx.get(path="/loop").mock(
            return_value=httpx.Response(302, headers={"location": "https://public.example/loop"})
        )
        async with PageFetcher(settings, max_redirects=3) as fetcher:
            result = await fetcher.fetch("https://public.example/loop")
        assert result.status is FetchStatus.HTTP_ERROR
        assert "redirect" in (result.error or "")

    async def test_unsupported_scheme_is_blocked_by_the_fetcher(self, settings: Settings) -> None:
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("file:///etc/passwd")
        assert result.status is FetchStatus.BLOCKED
