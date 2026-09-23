"""Source deduplication.

Runs in two passes, at two different points in the pipeline, because the two
kinds of duplicate become detectable at different times:

* **Before fetching** — URL-level. Several sub-questions routinely surface the
  same page. Collapsing them here is what makes each unique page cost one
  fetch and one extraction call instead of one per query that found it.
* **After fetching** — content-level. Syndicated and mirrored articles have
  different URLs and identical bodies, which only becomes visible once the
  text is in hand.

Merging is not discarding: when two results collapse, their provenance is
unioned, so a page found by three sub-questions still records all three.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from agentic_research.models import DiscoveryRef, SearchResult, SourceDocument
from agentic_research.retrieval.urls import canonicalize, domain_of

# Near-duplicate titles on the same domain, e.g. a page reachable at both
# /guide and /guide-v2. Set high because false merges lose real evidence.
_TITLE_SIMILARITY_THRESHOLD = 0.92

# Titles shorter than this are too easy to match by accident.
_MIN_TITLE_LENGTH_FOR_MATCH = 20

_DIGITS = re.compile(r"\d+")


@dataclass
class DedupStats:
    """Counts used for the run's duplicate-retrieval metric."""

    results_in: int = 0
    unique_out: int = 0
    duplicate_urls: int = 0
    duplicate_content: int = 0
    duplicate_titles: int = 0
    already_known: int = 0

    @property
    def fetches_avoided(self) -> int:
        """Page fetches (and extraction calls) skipped by deduplication."""
        return self.duplicate_urls + self.already_known

    @property
    def duplicate_rate(self) -> float:
        if self.results_in == 0:
            return 0.0
        total = self.duplicate_urls + self.already_known + self.duplicate_content
        return round(total / self.results_in, 4)


@dataclass
class Candidate:
    """A unique URL awaiting fetch, with every query that surfaced it."""

    canonical_url: str
    url: str
    title: str
    snippet: str
    domain: str
    best_score: float | None = None
    provider_content: str | None = None
    published_date: object | None = None
    discovered_by: list[DiscoveryRef] = field(default_factory=list)
    """Every (query, sub-question) path that surfaced this URL. Kept as pairs
    rather than two parallel lists so the relationship survives the merge."""

    @property
    def found_by_queries(self) -> list[str]:
        return list(dict.fromkeys(d.query_id for d in self.discovered_by))

    @property
    def sub_question_ids(self) -> list[str]:
        return list(dict.fromkeys(d.sub_question_id for d in self.discovered_by))


def _normalise_title(title: str) -> str:
    return " ".join(title.lower().split())


def _titles_match(a: str, b: str) -> bool:
    """Whether two same-domain titles denote the same document.

    String similarity alone is not enough. "Part 1" and "Part 2", "Python 2"
    and "Python 3", or the 2024 and 2025 editions of one report all score
    above any useful threshold while being genuinely different documents, so
    differing numbers veto a merge outright.
    """
    if len(a) < _MIN_TITLE_LENGTH_FOR_MATCH or len(b) < _MIN_TITLE_LENGTH_FOR_MATCH:
        return False
    if _DIGITS.findall(a) != _DIGITS.findall(b):
        return False
    return SequenceMatcher(None, a, b).ratio() >= _TITLE_SIMILARITY_THRESHOLD


def dedupe_search_results(
    results: list[SearchResult],
    query_to_sub_question: dict[str, str],
    known_canonical_urls: set[str] | None = None,
) -> tuple[list[Candidate], DedupStats]:
    """Collapse this round's search results into unique fetch candidates.

    ``known_canonical_urls`` carries sources already retrieved in earlier
    rounds, so the iterative loop cannot re-fetch what it already has.
    """
    known = known_canonical_urls or set()
    stats = DedupStats(results_in=len(results))
    by_url: dict[str, Candidate] = {}

    for result in results:
        canonical = canonicalize(result.url)
        if not canonical:
            continue

        sub_question_id = query_to_sub_question.get(result.query_id, "")

        if canonical in known:
            stats.already_known += 1
            continue

        existing = by_url.get(canonical)
        if existing is not None:
            stats.duplicate_urls += 1
            _merge_into(existing, result, sub_question_id)
            continue

        by_url[canonical] = Candidate(
            canonical_url=canonical,
            url=result.url,
            title=result.title,
            snippet=result.snippet,
            domain=domain_of(result.url),
            best_score=result.score,
            provider_content=result.raw_content,
            published_date=result.published_date,
            discovered_by=(
                [DiscoveryRef(query_id=result.query_id, sub_question_id=sub_question_id)]
                if result.query_id and sub_question_id
                else []
            ),
        )

    candidates = _collapse_similar_titles(list(by_url.values()), stats)
    stats.unique_out = len(candidates)
    return candidates, stats


def _merge_into(candidate: Candidate, result: SearchResult, sub_question_id: str) -> None:
    """Fold a duplicate result into the candidate that already holds its URL."""
    if result.query_id and sub_question_id:
        ref = DiscoveryRef(query_id=result.query_id, sub_question_id=sub_question_id)
        if ref not in candidate.discovered_by:
            candidate.discovered_by.append(ref)
    # Keep the strongest relevance signal any query produced for this page.
    if result.score is not None and (
        candidate.best_score is None or result.score > candidate.best_score
    ):
        candidate.best_score = result.score
    # Keep whichever body text we managed to obtain.
    if not candidate.provider_content and result.raw_content:
        candidate.provider_content = result.raw_content
    if candidate.published_date is None and result.published_date is not None:
        candidate.published_date = result.published_date


def _collapse_similar_titles(candidates: list[Candidate], stats: DedupStats) -> list[Candidate]:
    """Merge same-domain candidates whose titles are near-identical.

    Restricted to a single domain: two unrelated sites publishing an article
    called "Getting Started" are not the same document, and merging across
    domains would also destroy the independent-corroboration signal that
    source diversity depends on.
    """
    kept: list[Candidate] = []
    by_domain: dict[str, list[Candidate]] = {}

    for candidate in candidates:
        peers = by_domain.setdefault(candidate.domain, [])
        title = _normalise_title(candidate.title)
        match = None
        if title:
            match = next(
                (p for p in peers if _titles_match(title, _normalise_title(p.title))), None
            )
        if match is not None:
            stats.duplicate_titles += 1
            for ref in candidate.discovered_by:
                if ref not in match.discovered_by:
                    match.discovered_by.append(ref)
            continue
        peers.append(candidate)
        kept.append(candidate)

    return kept


def dedupe_by_content(sources: list[SourceDocument]) -> tuple[list[SourceDocument], int]:
    """Second pass, after fetching: collapse identical bodies.

    Catches syndication, where the same article is republished at several
    domains. The duplicate is retained in the returned list but marked with
    ``duplicate_of`` rather than deleted, so a citation that already points at
    it can still be resolved.
    """
    unique: list[SourceDocument] = []
    by_hash: dict[str, SourceDocument] = {}
    duplicates = 0

    for source in sources:
        if not source.is_usable or not source.content_hash:
            unique.append(source)
            continue
        original = by_hash.get(source.content_hash)
        if original is None:
            by_hash[source.content_hash] = source
            unique.append(source)
            continue
        duplicates += 1
        for ref in source.discovered_by:
            if ref not in original.discovered_by:
                original.discovered_by.append(ref)
        unique.append(source.model_copy(update={"duplicate_of": original.id}))

    return unique, duplicates
