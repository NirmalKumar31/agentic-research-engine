"""Source quality heuristics.

What this is: a cheap, transparent prior for *ordering* sources, deciding
which pages are worth an extraction call when the budget will not cover them
all, and warning when a conclusion rests on one blog post.

What this is not: a truth detector. Every signal here is about the *kind* of
document, not about whether its contents are correct. A vendor whitepaper can
be right and a standards document can be out of date. The score is reported
alongside the report rather than hidden, precisely so a reader can disagree
with it.

Every component is stated and weighted explicitly below. Nothing is learned,
and no model is asked to produce a number here — an LLM-generated "quality:
0.87" would carry no defined meaning.
"""

from __future__ import annotations

from datetime import UTC, datetime

from agentic_research.models import SourceType

# Generic suffix fallback, consulted *last*. A suffix says who operates a
# host, not what kind of document it publishes, so an explicit mapping
# always wins over one of these.
_SUFFIX_TYPES: tuple[tuple[str, SourceType], ...] = (
    (".gov", SourceType.GOVERNMENT),
    (".mil", SourceType.GOVERNMENT),
    (".edu", SourceType.ACADEMIC),
    (".ac.uk", SourceType.ACADEMIC),
)

_EXACT_DOMAIN_TYPES: dict[str, SourceType] = {
    "arxiv.org": SourceType.ACADEMIC,
    "aclanthology.org": SourceType.ACADEMIC,
    "openreview.net": SourceType.ACADEMIC,
    "pubmed.ncbi.nlm.nih.gov": SourceType.ACADEMIC,
    "pmc.ncbi.nlm.nih.gov": SourceType.ACADEMIC,
    "dl.acm.org": SourceType.ACADEMIC,
    "ieeexplore.ieee.org": SourceType.ACADEMIC,
    "nature.com": SourceType.ACADEMIC,
    "sciencedirect.com": SourceType.ACADEMIC,
    "scholar.google.com": SourceType.ACADEMIC,
    "semanticscholar.org": SourceType.ACADEMIC,
    "ietf.org": SourceType.STANDARDS_BODY,
    "w3.org": SourceType.STANDARDS_BODY,
    "iso.org": SourceType.STANDARDS_BODY,
    "nist.gov": SourceType.STANDARDS_BODY,
    "owasp.org": SourceType.STANDARDS_BODY,
    "stackoverflow.com": SourceType.FORUM,
    "stackexchange.com": SourceType.FORUM,
    "reddit.com": SourceType.FORUM,
    "news.ycombinator.com": SourceType.FORUM,
    "quora.com": SourceType.FORUM,
    "medium.com": SourceType.BLOG,
    "substack.com": SourceType.BLOG,
    "dev.to": SourceType.BLOG,
    "hashnode.dev": SourceType.BLOG,
    "towardsdatascience.com": SourceType.BLOG,
    "wikipedia.org": SourceType.OTHER,
    "reuters.com": SourceType.NEWS,
    "apnews.com": SourceType.NEWS,
    "bbc.com": SourceType.NEWS,
    "nytimes.com": SourceType.NEWS,
    "wsj.com": SourceType.NEWS,
    "theguardian.com": SourceType.NEWS,
    "bloomberg.com": SourceType.NEWS,
    "ft.com": SourceType.NEWS,
    "techcrunch.com": SourceType.NEWS,
    "arstechnica.com": SourceType.NEWS,
    "theverge.com": SourceType.NEWS,
}

# Host prefixes that indicate first-party documentation: docs.stripe.com is
# Stripe's own documentation, and the subdomain is the evidence of that.
_DOCS_PREFIXES = ("docs.", "developer.", "developers.", "api.", "learn.", "platform.")

# A /docs/ path says the publisher documents *its own* product. It says
# nothing about authority over the subject being researched: a vendor page
# at example.com/docs/nist-ai-rmf is commentary on NIST, not a NIST
# publication. Such pages classify as VENDOR, which scores below a
# standards body or an academic source.
_DOCS_PATH_HINTS = ("/docs/", "/documentation/", "/reference/", "/api/", "/guide/")

_BLOG_PATH_HINTS = ("/blog/", "/posts/", "/news/")

# Base score by document kind. Primary and first-party sources rank above
# commentary about them; forums rank lowest because they are unreviewed, not
# because they are wrong.
_TYPE_BASE: dict[SourceType, float] = {
    SourceType.STANDARDS_BODY: 0.90,
    SourceType.OFFICIAL_DOCS: 0.85,
    SourceType.ACADEMIC: 0.85,
    # Authoritative as a publisher, but not a standards authority.
    SourceType.GOVERNMENT: 0.80,
    SourceType.NEWS: 0.60,
    SourceType.VENDOR: 0.55,
    SourceType.BLOG: 0.45,
    SourceType.FORUM: 0.35,
    SourceType.OTHER: 0.50,
}


def classify_source(url: str, domain: str) -> SourceType:
    """Best-effort guess at what kind of document a URL points to."""
    host = domain.lower()
    path = url.lower()

    # Order is the whole correctness of this function, and it was wrong.
    # The suffix table ran first, so pubmed.ncbi.nlm.nih.gov matched
    # ".gov" and was labelled a standards body -- its explicit ACADEMIC
    # entry was unreachable. Specific knowledge must beat a guess derived
    # from a TLD.
    #
    #   1. exact or parent-domain mappings
    #   2. first-party documentation patterns
    #   3. generic suffix fallback
    for known, source_type in _EXACT_DOMAIN_TYPES.items():
        if host == known or host.endswith("." + known):
            return source_type

    if host.startswith(_DOCS_PREFIXES):
        return SourceType.OFFICIAL_DOCS

    for suffix, source_type in _SUFFIX_TYPES:
        if host.endswith(suffix):
            return source_type

    if any(hint in path for hint in _DOCS_PATH_HINTS):
        return SourceType.VENDOR
    if any(hint in path for hint in _BLOG_PATH_HINTS):
        return SourceType.BLOG
    return SourceType.OTHER


def _recency_adjustment(
    published: datetime | None, horizon_months: int | None
) -> tuple[float, str | None]:
    """Reward or penalise age, but only when the question is time sensitive.

    For a question about a stable algorithm, a 2016 paper is not worse than a
    2026 blog post, so recency is deliberately inert unless asked for.
    """
    if horizon_months is None or published is None:
        return 0.0, None
    now = datetime.now(UTC)
    when = published if published.tzinfo else published.replace(tzinfo=UTC)
    age_months = max(0.0, (now - when).days / 30.44)
    if age_months <= horizon_months:
        return 0.08, f"published within the {horizon_months}-month recency window"
    if age_months <= horizon_months * 2:
        return -0.05, f"slightly older than the {horizon_months}-month window"
    return -0.15, f"much older than the {horizon_months}-month recency window"


def score_source(
    *,
    url: str,
    domain: str,
    source_type: SourceType,
    search_score: float | None,
    word_count: int,
    published: datetime | None = None,
    recency_horizon_months: int | None = None,
) -> tuple[float, list[str]]:
    """Return a 0..1 quality prior and the reasons behind it.

    Reasons are returned with the score so that the number is auditable rather
    than oracular.
    """
    reasons: list[str] = []
    score = _TYPE_BASE.get(source_type, 0.5)
    reasons.append(f"{source_type.value} baseline {score:.2f}")

    # The search provider's own relevance judgement, damped so it can shift
    # the ranking without overwhelming document type.
    if search_score is not None:
        delta = (min(max(search_score, 0.0), 1.0) - 0.5) * 0.20
        score += delta
        reasons.append(f"search relevance {search_score:.2f} -> {delta:+.3f}")

    # Very short extractions are usually landing pages, paywall stubs or
    # cookie walls rather than substantive documents.
    if word_count < 120:
        score -= 0.20
        reasons.append(f"thin content ({word_count} words) -> -0.200")
    elif word_count > 600:
        score += 0.05
        reasons.append(f"substantial content ({word_count} words) -> +0.050")

    delta, reason = _recency_adjustment(published, recency_horizon_months)
    if reason:
        score += delta
        reasons.append(f"{reason} -> {delta:+.3f}")

    return round(min(max(score, 0.0), 1.0), 3), reasons


def domain_concentration(domains: list[str]) -> float:
    """Share of sources coming from the single most common domain.

    High concentration means the research leaned on one publisher. That is a
    warning about independence, not about correctness.
    """
    if not domains:
        return 0.0
    counts: dict[str, int] = {}
    for domain in domains:
        if domain:
            counts[domain] = counts.get(domain, 0) + 1
    if not counts:
        return 0.0
    return round(max(counts.values()) / len(domains), 4)
