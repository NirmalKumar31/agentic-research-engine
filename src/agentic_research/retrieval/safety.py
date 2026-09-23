"""Outbound URL safety policy.

The fetcher follows URLs chosen by a search provider and, transitively, by
whatever those pages redirect to. None of that is trustworthy input. Without
a policy, a hosted deployment will happily fetch ``http://169.254.169.254/``
and hand cloud instance credentials to a language model.

The policy is deny-by-default on address ranges rather than on hostnames,
because a hostname can resolve anywhere. Every candidate is resolved and
every resolved address is checked, and the same check is reapplied to each
redirect hop rather than trusting ``follow_redirects``.

DNS rebinding is closed by pinning: the connection is made to the address
that was validated, not to whatever the name resolves to a moment later.
Rebinding works by answering the validation lookup with a public address
and the connection lookup with a private one; connecting to the address
that was actually checked removes that window.
The hostname is preserved in the ``Host`` header and in the TLS SNI, so
virtual hosting and certificate validation behave normally.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Cloud instance metadata services. These resolve inside link-local space and
# are already covered by the range checks; they are named separately so a
# blocked attempt says why, and so the intent survives a refactor of the
# range logic.
_METADATA_HOSTS = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
        "metadata",
    }
)

# Names that mean "this machine" regardless of what DNS says. Blocked
# syntactically as well as by resolution, so the check does not depend on
# the resolver behaving.
_LOCAL_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})
_LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain")

# Ports outside these are refused. Fetching http on port 22 or 6379 is never
# research; it is a port probe.
_ALLOWED_PORTS = frozenset({80, 443, 8080, 8443})


class UnsafeURLError(Exception):
    """A URL was rejected before any connection was attempted."""

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        self.reason = reason
        super().__init__(f"refused {url}: {reason}")


class UnresolvableHostError(UnsafeURLError):
    """DNS gave us nothing.

    A subclass of UnsafeURLError because we still must not connect, but
    distinguished so callers can report it as the network failure it is
    rather than as a policy refusal.
    """


class UnpinnedTargetError(UnsafeURLError):
    """A connection was requested for a target with no validated address.

    Fails closed. The obvious alternative -- fall back to connecting by
    hostname -- is exactly the DNS-rebinding window pinning exists to
    remove, so a target that cannot be pinned must not be contacted at all.
    Reachable when a caller validated with ``resolve=False`` and then tried
    to connect, which is a programming error rather than a network one.
    """


@dataclass(frozen=True)
class SafeTarget:
    """A URL that passed validation, with the addresses it resolved to."""

    url: str
    host: str
    port: int
    addresses: tuple[str, ...]
    scheme: str = "https"

    def pinned_request(self, address_index: int = 0) -> tuple[str, dict[str, str], dict[str, str]]:
        """Return ``(url, headers, extensions)`` that connect to a checked IP.

        The URL's host is replaced with a validated address so the socket
        cannot be pointed somewhere else by a second DNS lookup. The
        original hostname is carried in ``Host`` for virtual hosting and in
        ``sni_hostname`` so TLS still presents and verifies the real name --
        pinning the address must not weaken certificate validation.

        ``address_index`` selects among the addresses that were validated in
        this same resolution. It exists for failover and deliberately cannot
        reach anything outside that set: re-resolving to find an alternative
        would hand the attacker the second lookup pinning removes.

        Raises :class:`UnpinnedTargetError` when there is no validated
        address. Failing closed matters here -- returning the hostname URL
        instead would silently restore the rebinding window.
        """
        if not self.addresses:
            raise UnpinnedTargetError(
                self.url,
                "no validated address to pin to; refusing to connect by hostname",
            )
        if not 0 <= address_index < len(self.addresses):
            raise UnpinnedTargetError(
                self.url,
                f"address index {address_index} outside the validated set of {len(self.addresses)}",
            )

        address = self.addresses[address_index]
        literal = f"[{address}]" if ":" in address else address
        default_port = 443 if self.scheme == "https" else 80
        netloc = literal if self.port == default_port else f"{literal}:{self.port}"

        parts = urlsplit(self.url)
        pinned = urlunsplit((self.scheme, netloc, parts.path or "/", parts.query, ""))
        host_header = self.host if self.port == default_port else f"{self.host}:{self.port}"
        headers = {"Host": host_header}
        extensions = {"sni_hostname": self.host} if self.scheme == "https" else {}
        return pinned, headers, extensions


def _address_is_blocked(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Return a reason string when an address must not be contacted."""
    if address.is_loopback:
        return "loopback address"
    if address.is_link_local:
        # Covers 169.254.0.0/16 and fe80::/10, i.e. cloud metadata.
        return "link-local address (cloud metadata range)"
    if address.is_private:
        return "private address"
    if address.is_multicast:
        return "multicast address"
    if address.is_reserved:
        return "reserved address"
    if address.is_unspecified:
        return "unspecified address"
    # IPv4-mapped and 6to4 forms can smuggle a private v4 address through a
    # v6 literal, so unwrap and re-check rather than trusting the v6 flags.
    if isinstance(address, ipaddress.IPv6Address):
        mapped = address.ipv4_mapped or getattr(address, "sixtofour", None)
        if mapped is not None:
            nested = _address_is_blocked(mapped)
            if nested:
                return f"IPv6-wrapped {nested}"
    return None


def _resolve(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError) as exc:
        raise UnresolvableHostError(host, "hostname does not resolve") from exc
    # sockaddr[0] is the address for both AF_INET and AF_INET6; the tuple is
    # typed loosely, so narrow explicitly rather than silencing.
    return sorted({str(info[4][0]) for info in infos})


def validate_url(url: str, *, resolve: bool = True) -> SafeTarget:
    """Validate a URL for outbound fetching, or raise :class:`UnsafeURLError`.

    ``resolve=False`` skips DNS, which is only for unit tests that assert the
    syntactic checks without touching the network.
    """
    raw = (url or "").strip()
    if not raw:
        raise UnsafeURLError(raw, "empty URL")

    try:
        parts = urlsplit(raw)
    except ValueError as exc:
        raise UnsafeURLError(raw, f"malformed URL ({exc})") from exc

    scheme = (parts.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise UnsafeURLError(raw, f"scheme {scheme or '(none)'!r} is not allowed")

    try:
        host = parts.hostname
    except ValueError as exc:
        raise UnsafeURLError(raw, f"malformed host ({exc})") from exc
    if not host:
        raise UnsafeURLError(raw, "no host")
    host = host.rstrip(".").lower()

    try:
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise UnsafeURLError(raw, f"invalid port ({exc})") from exc
    if port not in _ALLOWED_PORTS:
        raise UnsafeURLError(raw, f"port {port} is not allowed")

    if host in _METADATA_HOSTS:
        raise UnsafeURLError(raw, "cloud metadata endpoint")
    if host in _LOCAL_HOSTNAMES or host.endswith(_LOCAL_SUFFIXES):
        raise UnsafeURLError(raw, "local hostname")

    # A bare IP literal never needs DNS, and must be checked directly so a
    # literal cannot bypass resolution.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        reason = _address_is_blocked(literal)
        if reason:
            raise UnsafeURLError(raw, reason)
        return SafeTarget(url=raw, host=host, port=port, addresses=(str(literal),), scheme=scheme)

    if not resolve:
        return SafeTarget(url=raw, host=host, port=port, addresses=(), scheme=scheme)

    addresses = _resolve(host)
    if not addresses:
        raise UnresolvableHostError(raw, "hostname resolved to no addresses")
    for candidate in addresses:
        try:
            parsed = ipaddress.ip_address(candidate)
        except ValueError:
            raise UnsafeURLError(raw, f"unparseable resolved address {candidate}") from None
        reason = _address_is_blocked(parsed)
        if reason:
            # Every resolved address must be safe, not merely one of them:
            # otherwise a name with one public and one private A record slips
            # through depending on which the client picks.
            raise UnsafeURLError(raw, f"resolves to {reason} ({candidate})")

    return SafeTarget(url=raw, host=host, port=port, addresses=tuple(addresses), scheme=scheme)


def is_safe_url(url: str, *, resolve: bool = True) -> bool:
    try:
        validate_url(url, resolve=resolve)
    except UnsafeURLError:
        return False
    return True
