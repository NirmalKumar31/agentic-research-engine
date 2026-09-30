"""Which candidates get fetched, and why the rest did not.

Selection is the last point at which source quality can be influenced.
The engine reads six pages; evidence ranking, entailment and relevance
all operate on what those six contain, so a primary source that was
never fetched cannot be recovered downstream by any later preference.

The defect this replaces, from a live run on "What are the main causes
of overfitting in machine learning?": 47 candidates were ranked by

    (round(provider_score, 1), authority_rank, canonical_url)

and the top six taken. Two consequences, both observed:

* **Authority never applied.** Banding to tenths means authority only
  broke ties *within* a band. A tweet the provider scored 0.87 sat in
  the 0.9 band and a primary source at 0.84 sat in 0.8, so the tweet
  won on relevance alone and the authority term was never consulted.
* **One sub-question took every slot.** A global top-N spent all six
  slots on the dimension with the highest-scoring candidates. Four of
  five research dimensions received no source at all, coverage came
  back 1/5, and nothing published.

So: allocate across sub-questions first, then rank within each
allocation by relevance *and* accountability. Relevance still leads --
an authoritative page about the wrong subject is worse than a good page
about the right one -- but a 0.03 relevance difference can no longer
outweigh tweet-versus-paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_research.citations.guards import SourceAuthority, authority_of
from agentic_research.evidence.dedup import Candidate
from agentic_research.evidence.quality import classify_source
from agentic_research.models import SourceType

# How much accountability is worth, expressed on the provider's own
# relevance scale so the two are commensurable.
#
# The bonus is capped at 0.15 and the penalty at 0.20, which fixes a
# property worth stating: a relevance gap wider than 0.35 can never be
# overturned by source class. Authority adjusts the ordering of pages
# the provider rated comparably; it is not a veto.
_AUTHORITY_BONUS: dict[SourceAuthority, float] = {
    SourceAuthority.PRIMARY: 0.15,
    SourceAuthority.SECONDARY: 0.08,
    SourceAuthority.AGGREGATOR: 0.0,
    SourceAuthority.UNKNOWN: 0.0,
}

# Applied only for questions whose answer is stable and documented.
# A question about current discussion or public opinion wants exactly
# the sources these penalise, which is why the caller decides.
_FIRST_PASS_PENALTY: dict[SourceType, float] = {
    SourceType.SOCIAL: -0.20,
    SourceType.BLOG: -0.08,
    SourceType.FORUM: -0.08,
}

# Bucket key for a candidate no query attributed to a sub-question.
_UNATTRIBUTED = "_unattributed"

_MAX_BONUS = max(_AUTHORITY_BONUS.values())
_MAX_PENALTY = -min(_FIRST_PASS_PENALTY.values())
DECIDING_RELEVANCE_GAP = _MAX_BONUS + _MAX_PENALTY
"""Relevance difference beyond which source class cannot change the order."""


@dataclass(frozen=True)
class DroppedCandidate:
    """A candidate that was not fetched, with the reason."""

    url: str
    domain: str
    source_type: str
    relevance: float
    reason: str


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[Candidate, ...] = ()
    dropped: tuple[DroppedCandidate, ...] = ()
    by_sub_question: dict[str, list[str]] = field(default_factory=dict)
    """Sub-question id -> the canonical URLs chosen for it.

    The diagnostic that would have made the overfitting run legible:
    four sub-questions mapped to nothing."""

    @property
    def sub_questions_with_no_source(self) -> tuple[str, ...]:
        return tuple(sq for sq, urls in self.by_sub_question.items() if not urls)


# Shapes whose answer is stable and documented, so the best source is
# an accountable one. TEMPORAL and RECOMMENDATION are excluded on
# purpose: "what is the current..." and "should we adopt..." are
# precisely the questions where recent discussion and practitioner
# opinion are the material, and penalising those sources would be
# answering a different question from the one asked.
_ACCOUNTABLE_SHAPES = frozenset(
    {"definition", "comparison", "list", "causal", "procedural", "numeric", "synthesis"}
)


def prefers_accountable_sources(contract: object | None) -> bool:
    """Whether this question's answer should come from accountable sources.

    Defaults to ``True`` when there is no usable contract. That is the
    safe direction: preferring a reviewed source for a question whose
    shape is unknown costs little, whereas defaulting to "anything
    goes" reproduces the run that read a tweet.
    """
    question_type = getattr(contract, "question_type", None)
    if question_type is None:
        return True
    return str(question_type) in _ACCOUNTABLE_SHAPES


def ranking_score(candidate: Candidate, *, explanatory: bool) -> float:
    """Relevance adjusted by how accountable the source is."""
    source_type = classify_source(candidate.url, candidate.domain)
    score = float(candidate.best_score or 0.0)
    score += _AUTHORITY_BONUS.get(authority_of(source_type.value), 0.0)
    if explanatory:
        score += _FIRST_PASS_PENALTY.get(source_type, 0.0)
    return score


def _sort_key(candidate: Candidate, *, explanatory: bool) -> tuple[float, str]:
    # Canonical URL last so an identical pool always selects identically.
    return (ranking_score(candidate, explanatory=explanatory), candidate.canonical_url)


def select_with_diagnostics(
    candidates: list[Candidate] | tuple[Candidate, ...],
    *,
    limit: int,
    explanatory: bool = True,
    sub_question_order: list[str] | tuple[str, ...] = (),
) -> SelectionResult:
    """Allocate ``limit`` fetches across sub-questions, then rank.

    ``sub_question_order`` lets the caller pass planner priority. Any
    sub-question a candidate mentions but the caller did not list is
    still served, appended in first-seen order, so a missing priority
    list degrades to "cover everything" rather than to "cover nothing".

    A candidate surfaced by several sub-questions is **selected once and
    credited to all of them**. An earlier version of this function put
    such a candidate in every bucket and then recorded it against
    whichever bucket happened to claim it first, which produced two
    wrong outcomes: a sub-question whose query returned the selected
    page was reported as having no source at all, so critique published
    a "no suitable source found" gap that was false; and the bucket that
    lost the race went on to spend another fetch on a sub-question
    already represented by that same page.

    Allocation is about *fetching*, not about answering. Crediting a
    shared candidate to two sub-questions says the page was retrieved
    for both, and nothing more -- whether its extracted evidence
    actually addresses either is decided later by topicality and
    coverage, which look at the text rather than at the query that
    surfaced the URL.
    """
    pool = list(candidates)
    if limit <= 0 or not pool:
        return SelectionResult(
            dropped=tuple(_dropped(c, "no fetch budget remained", explanatory) for c in pool),
            by_sub_question={sq: [] for sq in sub_question_order},
        )

    # Bucket by sub-question. A candidate surfaced for several
    # sub-questions appears in each bucket; selection deduplicates.
    buckets: dict[str, list[Candidate]] = {sq: [] for sq in sub_question_order}
    for candidate in pool:
        for sq in candidate.sub_question_ids or [_UNATTRIBUTED]:
            buckets.setdefault(sq, []).append(candidate)
    for sq in buckets:
        buckets[sq].sort(key=lambda c: _sort_key(c, explanatory=explanatory), reverse=True)

    selected: list[Candidate] = []
    chosen_urls: set[str] = set()
    by_sub_question: dict[str, list[str]] = {sq: [] for sq in buckets}

    def claim(candidate: Candidate) -> None:
        """Select once, credit every sub-question the candidate serves."""
        selected.append(candidate)
        chosen_urls.add(candidate.canonical_url)
        for sq in candidate.sub_question_ids or [_UNATTRIBUTED]:
            urls = by_sub_question.setdefault(sq, [])
            if candidate.canonical_url not in urls:
                urls.append(candidate.canonical_url)

    # Breadth first: every sub-question gets *representation* before any
    # gets a second source. A sub-question already credited by a shared
    # candidate is skipped rather than given a duplicate, which is what
    # frees the slot for a sub-question that has none.
    for sq, bucket in buckets.items():
        if len(selected) >= limit:
            break
        if by_sub_question.get(sq):
            continue
        for candidate in bucket:
            if candidate.canonical_url not in chosen_urls:
                claim(candidate)
                break

    # Then depth, round-robin, so a remaining budget is spread rather
    # than spent entirely on the first sub-question.
    progressed = True
    while len(selected) < limit and progressed:
        progressed = False
        for bucket in buckets.values():
            if len(selected) >= limit:
                break
            for candidate in bucket:
                if candidate.canonical_url in chosen_urls:
                    continue
                claim(candidate)
                progressed = True
                break

    dropped = tuple(
        _dropped(c, _drop_reason(c, explanatory), explanatory)
        for c in sorted(pool, key=lambda c: _sort_key(c, explanatory=explanatory), reverse=True)
        if c.canonical_url not in chosen_urls
    )
    return SelectionResult(
        selected=tuple(selected), dropped=dropped, by_sub_question=by_sub_question
    )


def _drop_reason(candidate: Candidate, explanatory: bool) -> str:
    source_type = classify_source(candidate.url, candidate.domain)
    penalty = _FIRST_PASS_PENALTY.get(source_type, 0.0) if explanatory else 0.0
    if penalty:
        return (
            f"deprioritised as {source_type.value} for a question with a stable, "
            f"documented answer (authority-adjusted below the fetch budget)"
        )
    return "ranked below the fetch budget after allocation across sub-questions"


def _dropped(candidate: Candidate, reason: str, explanatory: bool) -> DroppedCandidate:
    source_type = classify_source(candidate.url, candidate.domain)
    return DroppedCandidate(
        url=candidate.url,
        domain=candidate.domain,
        source_type=source_type.value,
        relevance=float(candidate.best_score or 0.0),
        reason=reason,
    )


def select_candidates(
    candidates: list[Candidate] | tuple[Candidate, ...],
    *,
    limit: int,
    explanatory: bool = True,
    sub_question_order: list[str] | tuple[str, ...] = (),
) -> tuple[Candidate, ...]:
    """The selected candidates alone, for callers that need no diagnostics."""
    return select_with_diagnostics(
        candidates,
        limit=limit,
        explanatory=explanatory,
        sub_question_order=sub_question_order,
    ).selected
