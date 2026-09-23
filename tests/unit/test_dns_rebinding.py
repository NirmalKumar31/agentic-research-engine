"""DNS rebinding: the validated address is pinned into the connection.

Rebinding works by answering the validation lookup with a public address
and the connection lookup with a private one. Validating a *name* and then
connecting by *name* leaves that window open. Connecting to the address
that was actually checked closes it, which is what these tests pin down.

The hostname still has to survive into the `Host` header and the TLS SNI,
or pinning would quietly break virtual hosting and certificate validation.
"""

from __future__ import annotations

import socket

import httpx
import pytest
import respx

from agentic_research.config import Settings
from agentic_research.models import FetchStatus
from agentic_research.retrieval import PageFetcher
from agentic_research.retrieval.safety import (
    SafeTarget,
    UnpinnedTargetError,
    UnsafeURLError,
    validate_url,
)

GOOD_HTML = (
    "<html><body><article><p>Imbalanced fraud datasets contain very few "
    "positive cases, which distorts accuracy as a metric for them.</p>"
    "</article></body></html>"
)

PUBLIC = "93.184.216.34"
METADATA = "169.254.169.254"


def _answers(addresses: list[str]) -> list:
    return [(socket.AF_INET, 0, 0, "", (address, 0)) for address in addresses]


@pytest.fixture
def resolver(monkeypatch: pytest.MonkeyPatch):
    """Install a scripted resolver and report how often it was consulted."""
    import agentic_research.retrieval.safety as safety

    state: dict[str, object] = {"lookups": 0}

    def install(sequence: list[list[str]]) -> dict[str, object]:
        answers = iter(sequence)
        last = sequence[-1]

        def fake(host: str, *args: object, **kwargs: object) -> list:
            state["lookups"] = int(state["lookups"]) + 1  # type: ignore[arg-type]
            return _answers(next(answers, last))

        monkeypatch.setattr(safety.socket, "getaddrinfo", fake)
        return state

    return install


class TestPinning:
    def test_pinned_url_targets_the_validated_address(self, resolver) -> None:
        resolver([[PUBLIC]])
        url, headers, extensions = validate_url(
            "https://target.example/article?id=3"
        ).pinned_request()

        assert url == f"https://{PUBLIC}/article?id=3"
        # Pinning must not weaken TLS: the certificate is still checked
        # against the real hostname, not against the address.
        assert headers["Host"] == "target.example"
        assert extensions["sni_hostname"] == "target.example"

    def test_a_second_lookup_cannot_move_the_connection(self, resolver) -> None:
        """The attack itself, deterministically.

        First lookup answers public, any later lookup answers the cloud
        metadata address. The pinned target must still be the public one.
        """
        resolver([[PUBLIC], [METADATA]])
        pinned_url, _, _ = validate_url("https://rebind.example/x").pinned_request()

        assert PUBLIC in pinned_url
        assert METADATA not in pinned_url, "connection followed the rebind"

    def test_ipv6_is_bracketed(self, resolver) -> None:
        import agentic_research.retrieval.safety as safety

        def fake(host: str, *args: object, **kwargs: object) -> list:
            return [(socket.AF_INET6, 0, 0, "", ("2606:4700::1111", 0, 0, 0))]

        original = safety.socket.getaddrinfo
        safety.socket.getaddrinfo = fake  # type: ignore[assignment]
        try:
            url, _, _ = validate_url("https://v6.example/p").pinned_request()
        finally:
            safety.socket.getaddrinfo = original  # type: ignore[assignment]
        assert url.startswith("https://[2606:4700::1111]/")

    def test_a_non_default_port_survives(self, resolver) -> None:
        resolver([[PUBLIC]])
        url, headers, _ = validate_url("https://alt.example:8443/p").pinned_request()
        assert url == f"https://{PUBLIC}:8443/p"
        assert headers["Host"] == "alt.example:8443"

    def test_an_ip_literal_pins_to_itself_without_dns(self, resolver) -> None:
        state = resolver([[PUBLIC]])
        url, headers, _ = validate_url(f"https://{PUBLIC}/p").pinned_request()
        assert url == f"https://{PUBLIC}/p"
        assert headers["Host"] == PUBLIC
        assert state["lookups"] == 0, "a literal address needs no lookup"


class TestFailsClosed:
    """A target with no validated address must not be contacted.

    The tempting fallback -- connect by hostname when pinning is not
    possible -- is precisely the rebinding window pinning removes, so it
    has to raise instead.
    """

    def test_no_validated_address_raises_rather_than_falling_back(self) -> None:
        target = SafeTarget(
            url="https://unpinned.example/x",
            host="unpinned.example",
            port=443,
            addresses=(),
            scheme="https",
        )
        with pytest.raises(UnpinnedTargetError, match="refusing to connect by hostname"):
            target.pinned_request()

    def test_the_error_is_a_policy_refusal(self) -> None:
        """Subclassing UnsafeURLError means the fetcher's existing handler
        marks it BLOCKED rather than treating it as a network blip."""
        assert issubclass(UnpinnedTargetError, UnsafeURLError)

    def test_an_index_outside_the_validated_set_is_refused(self) -> None:
        """Failover must not be able to reach an address that was never
        validated, including by walking off the end of the list."""
        target = SafeTarget(
            url="https://two.example/x",
            host="two.example",
            port=443,
            addresses=(PUBLIC, "93.184.216.35"),
            scheme="https",
        )
        with pytest.raises(UnpinnedTargetError, match="outside the validated set"):
            target.pinned_request(2)
        with pytest.raises(UnpinnedTargetError):
            target.pinned_request(-1)

    async def test_the_fetcher_blocks_an_unpinnable_target(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end: no connection is attempted at all."""
        import agentic_research.retrieval.fetcher as fetcher_module

        def unpinnable(url: str, **kwargs: object) -> SafeTarget:
            from urllib.parse import urlsplit

            parts = urlsplit(url)
            return SafeTarget(
                url=url,
                host=parts.hostname or "",
                port=443,
                addresses=(),
                scheme="https",
            )

        monkeypatch.setattr(fetcher_module, "validate_url", unpinnable)
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://unpinned.example/x")

        assert result.status is FetchStatus.BLOCKED
        assert not result.ok
        assert result.text == ""


class TestFailoverStaysInsideTheValidatedSet:
    """Multiple public records are common; a dead first address should not
    lose the source. Failover must never re-resolve."""

    @respx.mock
    async def test_a_dead_first_address_falls_over_to_the_second(
        self, settings: Settings, resolver
    ) -> None:
        second = "93.184.216.35"
        resolver([[PUBLIC, second]])
        attempted: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempted.append(request.url.host)
            if request.url.host == PUBLIC:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(200, html=GOOD_HTML, headers={"content-type": "text/html"})

        respx.get(path="/doc").mock(side_effect=handler)
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://multi.example/doc")

        assert result.ok
        assert attempted == [PUBLIC, second], "failover did not try the next address"

    @respx.mock
    async def test_failover_never_re_resolves(self, settings: Settings, resolver) -> None:
        """A second lookup during failover would hand the attacker exactly
        the window pinning closes."""
        second = "93.184.216.35"
        state = resolver([[PUBLIC, second]])

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == PUBLIC:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(200, html=GOOD_HTML, headers={"content-type": "text/html"})

        respx.get(path="/doc").mock(side_effect=handler)
        async with PageFetcher(settings) as fetcher:
            await fetcher.fetch("https://multi.example/doc")

        assert state["lookups"] == 1, "failover performed an extra DNS lookup"

    @respx.mock
    async def test_an_http_error_does_not_trigger_failover(
        self, settings: Settings, resolver
    ) -> None:
        """A 500 is a real answer from the right server. Retrying it on
        another address would just hammer the origin."""
        resolver([[PUBLIC, "93.184.216.35"]])
        route = respx.get(path="/doc").mock(return_value=httpx.Response(500))

        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://multi.example/doc")

        assert result.status is FetchStatus.HTTP_ERROR
        assert route.call_count == 1

    @respx.mock
    async def test_all_addresses_failing_surfaces_the_error(
        self, settings: Settings, resolver
    ) -> None:
        resolver([[PUBLIC, "93.184.216.35"]])

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        respx.get(path="/doc").mock(side_effect=handler)
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://multi.example/doc")

        assert not result.ok
        assert result.status is FetchStatus.HTTP_ERROR


class TestFetcherEndToEnd:
    @respx.mock
    async def test_the_socket_target_is_the_checked_address(
        self, settings: Settings, resolver
    ) -> None:
        resolver([[PUBLIC]])
        route = respx.get(path="/doc").mock(
            return_value=httpx.Response(200, html=GOOD_HTML, headers={"content-type": "text/html"})
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://pinned.example/doc")

        assert result.ok
        request = route.calls[0].request
        assert request.url.host == PUBLIC, "did not connect to the pinned address"
        assert request.headers["host"] == "pinned.example"
        # A citation should show the logical URL; the address is transport detail.
        assert result.final_url == "https://pinned.example/doc"

    @respx.mock
    async def test_a_rebinding_host_is_never_contacted_privately(
        self, settings: Settings, resolver
    ) -> None:
        """Even if every lookup after the first answers private, the
        request goes to the address that passed validation."""
        resolver([[PUBLIC], [METADATA], [METADATA]])
        good = respx.get(path="/doc").mock(
            return_value=httpx.Response(200, html=GOOD_HTML, headers={"content-type": "text/html"})
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://rebind.example/doc")

        assert result.ok
        assert good.calls[0].request.url.host == PUBLIC

    @respx.mock
    async def test_each_redirect_hop_is_revalidated_and_repinned(
        self, settings: Settings, resolver
    ) -> None:
        resolver([[PUBLIC]])
        respx.get(path="/one").mock(
            return_value=httpx.Response(302, headers={"location": "https://second.example/two"})
        )
        second = respx.get(path="/two").mock(
            return_value=httpx.Response(200, html=GOOD_HTML, headers={"content-type": "text/html"})
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://first.example/one")

        assert result.ok
        # The second hop was pinned too, and carries its own hostname.
        assert second.calls[0].request.url.host == PUBLIC
        assert second.calls[0].request.headers["host"] == "second.example"

    @respx.mock
    async def test_a_relative_redirect_resolves_against_the_logical_url(
        self, settings: Settings, resolver
    ) -> None:
        """A relative Location must not inherit the pinned IP as its host,
        or the next hop would be validated against an address instead of
        the name the origin meant."""
        resolver([[PUBLIC]])
        respx.get(path="/start").mock(
            return_value=httpx.Response(302, headers={"location": "/landing"})
        )
        landing = respx.get(path="/landing").mock(
            return_value=httpx.Response(200, html=GOOD_HTML, headers={"content-type": "text/html"})
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://site.example/start")

        assert result.ok
        assert landing.calls[0].request.headers["host"] == "site.example"
