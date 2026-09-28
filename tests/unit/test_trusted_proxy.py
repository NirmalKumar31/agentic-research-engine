"""Which address the per-client limiter believes.

X-Forwarded-For is appended to by each proxy, so its leftmost entry is
whatever the caller typed. The previous implementation read that entry,
which meant a caller could mint a fresh per-client allowance on every
request by changing a header -- and TRUSTED_PROXY_HOPS, configured in
the deployment blueprint, was read by nothing at all.

The per-client limit is abuse mitigation, not the financial boundary;
the durable global quota is that. It should still cost something to
defeat.
"""

from __future__ import annotations

import pytest

from agentic_research.web.api import _client_key

PEER = "203.0.113.9"


class Req:
    """Just the two things _client_key reads."""

    def __init__(self, forwarded: str | None = None, peer: str | None = PEER) -> None:
        self.headers: dict[str, str] = {}
        if forwarded is not None:
            self.headers["x-forwarded-for"] = forwarded
        self.client = type("C", (), {"host": peer})() if peer else None


def key(forwarded: str | None, hops: int, peer: str | None = PEER) -> str:
    return _client_key(Req(forwarded, peer), hops)  # type: ignore[arg-type]


class TestSpoofingIsIneffective:
    def test_a_spoofed_prefix_does_not_change_the_client(self) -> None:
        """Render appends the address it saw, so a forged entry is
        pushed left and the real address stays rightmost."""
        assert key("198.51.100.7, 192.0.2.44", hops=1) == "192.0.2.44"

    def test_many_forged_entries_still_resolve_to_the_real_client(self) -> None:
        forged = ", ".join(f"10.0.0.{n}" for n in range(1, 20))
        assert key(f"{forged}, 192.0.2.44", hops=1) == "192.0.2.44"

    def test_each_forged_value_does_not_mint_a_new_allowance(self) -> None:
        first = key("1.1.1.1, 192.0.2.44", hops=1)
        second = key("2.2.2.2, 192.0.2.44", hops=1)
        assert first == second == "192.0.2.44"


class TestHopCounting:
    def test_one_hop_reads_the_last_entry(self) -> None:
        assert key("192.0.2.44", hops=1) == "192.0.2.44"

    def test_two_hops_step_back_one_further(self) -> None:
        """Two trusted proxies contribute the last two entries, so the
        caller is the one before them."""
        assert key("192.0.2.44, 10.1.1.1", hops=2) == "192.0.2.44"

    def test_zero_hops_trusts_no_header_at_all(self) -> None:
        """The correct default. A service reached directly has no proxy
        to vouch for anything in that header."""
        assert key("192.0.2.44", hops=0) == PEER

    def test_fewer_entries_than_hops_falls_back_to_the_peer(self) -> None:
        """The request did not arrive through the expected path, so the
        header cannot be interpreted and the peer is the only fact."""
        assert key("192.0.2.44", hops=3) == PEER


class TestMalformedInput:
    @pytest.mark.parametrize(
        "value",
        [
            "not-an-ip",
            "",
            "   ",
            "192.0.2.999",
            "<script>alert(1)</script>",
            "192.0.2.44:8080",
            "example.com",
        ],
    )
    def test_a_non_address_falls_back_to_the_peer(self, value: str) -> None:
        assert key(value, hops=1) == PEER

    def test_no_header_falls_back_to_the_peer(self) -> None:
        assert key(None, hops=1) == PEER

    def test_no_peer_and_no_header_is_still_a_usable_key(self) -> None:
        assert key(None, hops=1, peer=None) == "unknown"

    def test_empty_entries_are_ignored_not_counted(self) -> None:
        """A caller padding the header with commas must not shift which
        entry is treated as trusted."""
        assert key(",, 192.0.2.44 ,", hops=1) == "192.0.2.44"


class TestAddressFamilies:
    def test_ipv4(self) -> None:
        assert key("198.51.100.7, 192.0.2.44", hops=1) == "192.0.2.44"

    def test_ipv6(self) -> None:
        assert key("2001:db8::1, 2001:db8::2", hops=1) == "2001:db8::2"

    def test_ipv6_is_normalised(self) -> None:
        """2001:0db8::0001 and 2001:db8::1 are one client, and must not
        hold two separate allowances."""
        assert key("2001:0db8:0000::0001", hops=1) == key("2001:db8::1", hops=1)

    def test_leading_zero_octets_are_refused_not_normalised(self) -> None:
        """Python's ipaddress rejects 192.000.002.044 rather than
        parsing it, because leading zeros are read as octal by some
        stacks and as decimal by others -- the ambiguity is itself the
        spoofing vector. Falling back to the peer is the safe answer."""
        assert key("192.000.002.044", hops=1) == PEER


class TestTheSettingIsActuallyRead:
    def test_the_route_passes_the_configured_hops(self) -> None:
        """The defect was not a wrong rule, it was a setting nothing
        consulted. This asserts the call site, not the helper."""
        import inspect

        from agentic_research.web import api

        source = inspect.getsource(api.create_app)
        assert "_client_key(request, state.settings.trusted_proxy_hops)" in source
