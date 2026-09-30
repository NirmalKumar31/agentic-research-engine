"""A cheap negative prefilter, and the vocabulary to say so.

`quote_verified` is a provenance property -- the quote appears verbatim
in the source that cites it -- and coverage treated it as sufficient:
two exact-match items across two sources marked a sub-question covered.
A live run on the causes of overfitting retrieved papers on double
descent, extracted exact quotes from them, filed them under
sub-questions about training-data size, and counted those answered.

What this module adds is a **lower bound, not a relevance judgement**.
Two shared content words do not show that a source answers a question,
and the earlier name for this -- ``is_topical`` -- invited exactly that
reading. Four distinct things get confused if they share one word, so
they have four names here and nothing collapses them:

``lexically_plausible``
    This module. Deterministic, free, and weak: shared salient terms
    between a sub-question and an extracted quote. A negative is
    informative ("nothing retrieved even discusses these terms"); a
    positive means only "not obviously about something else".

``evidence_relevant``
    An evidence item admitted toward a sub-question: quote verified
    *and* lexically plausible. Still not a semantic judgement.

``claim_relevant``
    A published claim judged to help answer the question, by
    ``citations/relevance.py`` -- structural checks plus an independent
    judgement, against the answer contract.

``slot_satisfied``
    A contract slot discharged, by ``answer_coverage``. The only level
    that licenses saying the question was answered.

The report and the payload use these names. "Topical" is not shown to a
reader as though semantic relevance had been established from two
shared words.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum


class RelevanceLevel(StrEnum):
    """How strongly something has been shown to bear on a question.

    Ordered weakest to strongest. Named so that a report can state
    which level it reached instead of implying the strongest.
    """

    LEXICALLY_PLAUSIBLE = "lexically_plausible"
    EVIDENCE_RELEVANT = "evidence_relevant"
    CLAIM_RELEVANT = "claim_relevant"
    SLOT_SATISFIED = "slot_satisfied"


# Words that carry no subject matter. Kept small on purpose: a long
# stoplist starts removing terms that do distinguish one sub-question
# from another ("training", "data", "model" are the subject here).
_STOPWORDS = frozenset(
    {
        "about",
        "all",
        "and",
        "but",
        "few",
        "for",
        "her",
        "him",
        "his",
        "nor",
        "per",
        "she",
        "too",
        "via",
        "yet",
        "you",
        "after",
        "against",
        "also",
        "another",
        "any",
        "are",
        "because",
        "been",
        "being",
        "between",
        "both",
        "can",
        "could",
        "did",
        "does",
        "doing",
        "during",
        "each",
        "from",
        "further",
        "had",
        "has",
        "have",
        "how",
        "into",
        "its",
        "itself",
        "just",
        "may",
        "might",
        "more",
        "most",
        "much",
        "must",
        "not",
        "now",
        "only",
        "other",
        "our",
        "out",
        "over",
        "own",
        "same",
        "should",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "until",
        "very",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "would",
        "your",
    }
)

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")

# Two shared terms rather than one. One is met by coincidence -- almost
# any machine-learning text shares "data" with almost any other -- and
# that coincidence is what let double-descent papers count as evidence
# about training-set size.
_REQUIRED_OVERLAP = 2

# A single shared term this long is not a coincidence. It is what lets
# "self-attention" and "scaled dot-product attention" reach each other
# on "attention", and "overfitting" and "poor generalization" on
# "generalization", without a hand-written synonym universe.
_DISTINCTIVE_LENGTH = 8

# An acronym worth generating: two to six capitals.
_ACRONYM = re.compile(r"\b([A-Z]{2,6})s?\b")
# "retrieval-augmented generation (RAG)" states its own alias.
_PARENTHETICAL = re.compile(r"([A-Za-z][A-Za-z0-9\s-]{3,60}?)\s*\(([A-Z]{2,6})\)")


def salient_terms(text: str) -> frozenset[str]:
    """Lower-cased content words of three characters or more.

    A hyphenated compound yields its parts as well as the whole:
    "self-attention" gives ``self-attention``, ``self`` and
    ``attention``. Without this, a question about self-attention shares
    nothing with evidence about "scaled dot-product attention" -- the
    compound and the bare word are different tokens, and the correct
    evidence was rejected for a punctuation difference. Purely
    mechanical, so it adds no vocabulary the text did not contain.
    """
    out: set[str] = set()
    for token in _TOKEN.findall(text or ""):
        lowered = token.lower()
        if lowered not in _STOPWORDS:
            out.add(lowered)
        if "-" in lowered:
            for part in lowered.split("-"):
                if len(part) >= 3 and part not in _STOPWORDS:
                    out.add(part)
    return frozenset(out)


def _acronym_of(phrase: str) -> str:
    words = [w for w in re.split(r"[^A-Za-z0-9]+", phrase) if w]
    if len(words) < 2:
        return ""
    return "".join(w[0] for w in words).lower()


def alias_groups(question: str = "", entities: Sequence[str] = ()) -> tuple[frozenset[str], ...]:
    """Equivalent-term groups, grounded in this question's own text.

    Bounded on purpose. There is no global synonym table here: every
    group is derived from the question or the analysis's entity list, so
    the aliases available to a run are the ones that run actually
    names. A hand-written universe of technical equivalences would be
    unbounded, unauditable, and wrong the moment the field moved.

    Three sources, all local:

    * a parenthetical gloss in the question -- "retrieval-augmented
      generation (RAG)" declares its own alias;
    * the acronym of any multi-word entity, so an entity of "large
      language model" reaches evidence that says "LLM";
    * an entity that is itself an acronym of another entity.
    """
    groups: list[set[str]] = []
    named = [e.strip() for e in entities if e and e.strip()]

    for phrase, acronym in _PARENTHETICAL.findall(question or ""):
        group = set(salient_terms(phrase))
        if group:
            group.add(acronym.lower())
            groups.append(group)

    for entity in named:
        acronym = _acronym_of(entity)
        if acronym:
            group = set(salient_terms(entity))
            group.add(acronym)
            groups.append(group)

    # An entity that is an acronym of another entity or of a phrase the
    # question uses.
    acronyms = {e.lower() for e in named if _ACRONYM.fullmatch(e.strip())}
    for entity in named:
        built = _acronym_of(entity)
        if built and built in acronyms:
            groups.append(set(salient_terms(entity)) | {built})

    # Merge groups that share a term, so the relation is transitive.
    merged: list[set[str]] = []
    for group in groups:
        for existing in merged:
            if existing & group:
                existing |= group
                break
        else:
            merged.append(set(group))
    return tuple(frozenset(g) for g in merged if len(g) > 1)


def _expand(terms: frozenset[str], groups: Sequence[frozenset[str]]) -> frozenset[str]:
    out = set(terms)
    for group in groups:
        if out & group:
            out |= group
    return frozenset(out)


def overlap(question: str, evidence: str, *, aliases: Sequence[frozenset[str]] = ()) -> int:
    """How many salient terms the two share, counting declared aliases."""
    left = _expand(salient_terms(question), aliases)
    right = _expand(salient_terms(evidence), aliases)
    return len(left & right)


def shared_terms(
    question: str, evidence: str, *, aliases: Sequence[frozenset[str]] = ()
) -> frozenset[str]:
    left = _expand(salient_terms(question), aliases)
    right = _expand(salient_terms(evidence), aliases)
    return left & right


def lexically_plausible(
    question: str,
    evidence: str,
    *,
    aliases: Sequence[frozenset[str]] = (),
    topic_terms: frozenset[str] = frozenset(),
) -> bool:
    """Whether this evidence could plausibly bear on this sub-question.

    A **negative** is the informative direction: nothing retrieved
    discusses the sub-question's terms at all, which is a real finding
    about retrieval. A **positive** means only that the item is not
    obviously about something else -- it is not evidence that the
    source answers anything, and must never be reported as though it
    were.

    Passes on either two shared salient terms or one *discriminating*
    distinctive term of eight characters or more. ``topic_terms`` are
    the overall question's salient terms and are excluded from that
    shortcut: a term the whole run shares says nothing about which
    sub-question a quote bears on. The second rule is what lets
    "self-attention" and "scaled dot-product attention" reach each
    other on "attention" without a synonym table; a short shared term
    like "data" is met by coincidence and does not count on its own.

    The two empty cases are deliberately asymmetric. A sub-question with
    no salient terms gives this nothing to compare against, so it
    abstains and the item passes: rejecting there would call evidence
    irrelevant without knowing the topic, punishing a run for a
    degenerate question. Evidence with no content words is rejected,
    because that is the one direction where abstaining would reinstate
    the defect for the evidence carrying least.
    """
    terms = salient_terms(question)
    if not terms:
        return True
    if not salient_terms(evidence):
        return False
    shared = shared_terms(question, evidence, aliases=aliases)
    # The distinctive-term shortcut must not fire on the topic itself.
    #
    # Found by the offline evaluation. For "what are the main causes of
    # overfitting", every sub-question and every extracted quote
    # contains "overfitting" -- eleven characters, so the shortcut
    # admitted all ten excerpts for all five sub-questions, including
    # double-descent papers as evidence about data leakage. A term the
    # whole run shares carries no information about *which*
    # sub-question a quote bears on, which is the only thing decided
    # here. Topic terms still count toward the two-term threshold;
    # they just cannot satisfy it alone.
    discriminating = {term for term in shared if term not in topic_terms}
    if any(len(term) >= _DISTINCTIVE_LENGTH for term in discriminating):
        return True
    return len(shared) >= min(_REQUIRED_OVERLAP, len(terms))
