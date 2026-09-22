"""URL canonicalisation.

Deduplication is only as good as this module. The same article routinely
arrives as four different strings across a research round::

    https://www.Example.com/post?utm_source=x&id=7#intro
    https://example.com/post/?id=7
    http://example.com:80/post?id=7
    https://example.com/post?id=7&fbclid=abc

All four should collapse to one source, one fetch and one extraction call.

The rules stay conservative: canonicalisation that is too aggressive silently
merges genuinely different pages, which is a much worse failure than fetching
one page twice.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

# Parameters that identify a campaign or a referrer, never a document.
_TRACKING_PREFIXES = ("utm_", "pk_", "mc_", "ga_", "hsa_", "_hs")
_TRACKING_EXACT = frozenset(
    {
        "fbclid",
        "gclid",
        "dclid",
        "msclkid",
        "igshid",
        "mkt_tok",
        "yclid",
        "ref",
        "referrer",
        "source",
        "src",
        "campaign",
        "spm",
        "scid",
        "cmpid",
        "ncid",
        "sr_share",
        "at_medium",
        "at_campaign",
        "trk",
        "trkcampaign",
        "__twitter_impression",
        "guccounter",
        "ved",
        "usg",
    }
)

_DEFAULT_PORTS = {"http": "80", "https": "443"}
_INDEX_SUFFIXES = ("/index.html", "/index.htm", "/index.php", "/default.aspx")


def _is_tracking(key: str) -> bool:
    lowered = key.lower()
    return lowered in _TRACKING_EXACT or lowered.startswith(_TRACKING_PREFIXES)


def canonicalize(url: str) -> str:
    """Return a stable, comparable form of ``url``.

    Returns the input unchanged if it cannot be parsed; an unparseable URL is
    still a usable dictionary key, and raising here would mean one malformed
    search result could fail a whole round.
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    if not parts.scheme or not parts.netloc:
        return raw

    scheme = parts.scheme.lower()
    host = parts.hostname or ""
    host = host.lower().rstrip(".")
    # www is a hosting convention, not a different document. Other
    # subdomains (docs., blog., api.) genuinely can be, so only www goes.
    if host.startswith("www."):
        host = host[4:]

    netloc = host
    if parts.port and str(parts.port) != _DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{parts.port}"

    path = unquote(parts.path or "/")
    for suffix in _INDEX_SUFFIXES:
        if path.endswith(suffix):
            path = path[: -len(suffix)] or "/"
            break
    # AMP variants serve the same article at a different path.
    if path.endswith("/amp"):
        path = path[:-4] or "/"
    if len(path) > 1:
        path = path.rstrip("/")
    if not path:
        path = "/"

    kept = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking(k)
    ]
    # Sorted so that ?a=1&b=2 and ?b=2&a=1 compare equal.
    query = urlencode(sorted(kept), doseq=True)

    # Fragments address a position within a document, not a document.
    return urlunsplit((scheme, netloc, path, query, ""))


def domain_of(url: str) -> str:
    """Registrable-ish host for diversity metrics, with ``www.`` removed."""
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return ""
    host = host.lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def same_document(a: str, b: str) -> bool:
    return bool(a) and canonicalize(a) == canonicalize(b)
