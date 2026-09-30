"""What the question's own wording says it is asking for.

Read from the user's text, deterministically, before any model sees it.

Two live defects motivate this module, and both are cases of a model's
judgement silently replacing something the question stated outright.

**The shape was unstable.** "What are the main causes of hallucination
in large language models?" was classified ``list`` on one run and
``definition`` on a near-identical earlier phrasing, and "What is the
context window size of GPT-4 Turbo?" was classified ``definition`` --
so a figure was checked against a slot asking what the subject *is*.
The contract is derived from that label, so an unstable label means two
identical questions are held to different requirements.

**A context noun became a comparison side.** Asked how
retrieval-augmented generation differs from fine-tuning *for language
models*, the analyst returned three entities, and the coverage rule
required the published claims to speak about all three. Three correct,
cited, relevant claims were published and the report then said it had
not answered the question, because nothing mentioned "language models"
-- which was never one of the things being contrasted.

The scope here is deliberately narrow. This module answers only where
the wording is explicit, and returns ``None`` everywhere else so the
model's own reading stands. Guessing broadly is how a definition became
a list; the fix is not to guess harder in the other direction.
"""

from __future__ import annotations

import re

from agentic_research.answer_contract import QuestionType

# ---------------------------------------------------------------------------
# Comparison sides
# ---------------------------------------------------------------------------

# Connectives that put the things being compared on either side of
# themselves. Ordered longest-first so " vs. " is not matched by " vs ".
_SPLIT_CONNECTIVES: tuple[str, ...] = (
    " differs from ",
    " differ from ",
    " compared with ",
    " compared to ",
    " compare with ",
    " compare to ",
    " versus ",
    " vs. ",
    " vs ",
)

# "the difference between A and B" puts both on one side of the phrase.
_BETWEEN = re.compile(r"\bdifferences?\s+between\s+(?P<rest>.+)", re.IGNORECASE)

# "between A and B" in any phrasing. Looser than _BETWEEN and tried
# after it, because "what types of difference are there between X and
# Y" separates "difference" from "between" and the strict pattern
# misses it. Safe to be loose only because `_accept` rejects sides that
# name nothing.
_LOOSE_BETWEEN = re.compile(r"\bbetween\s+(?P<rest>.+)", re.IGNORECASE)

# "How do A, B and C differ?" -- the subjects precede a trailing verb.
_TRAILING_DIFFER = re.compile(
    r"^(?:how\s+(?:do|does)\s+)?(?P<subjects>.+?)\s+differ\b", re.IGNORECASE
)

# Interrogative openings that are never part of a subject's name.
_LEADING_NOISE = re.compile(
    r"^(?:what(?:'s| is| are)?(?: the)?(?: main| key| principal| practical)?"
    r"(?: differences?| distinctions?)?(?: between)?|how (?:does|do|is|are)|"
    r"in what ways?(?: do| does)?|compare|explain)\s+",
    re.IGNORECASE,
)

# Where a trailing phrase stops naming a subject and starts naming the
# setting it sits in. "fine-tuning for language models" is one subject
# in one domain, not two subjects -- this is the boundary the RAG defect
# crossed.
_CONTEXT_PREPOSITIONS: tuple[str, ...] = (
    " for ",
    " in ",
    " on ",
    " with ",
    " when ",
    " under ",
    " within ",
    " across ",
    " regarding ",
    " at ",
    " during ",
    " as of ",
)

_SUBJECT_SEPARATORS = re.compile(r",\s*and\s+|\s+and\s+|,\s*", re.IGNORECASE)

# A connective can appear in a sentence that compares nothing: "what
# does versus mean in legal citations" splits into "does" and "mean".
# A side has to name something, so a candidate consisting only of
# function words is rejected and the question is not read as a
# comparison at all.
_NOT_A_SUBJECT = frozenset(
    {
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "mean",
        "means",
        "meaning",
        "it",
        "this",
        "that",
        "they",
        "them",
        # Articles are deliberately absent: they are already removed as
        # leading noise, and keeping them here rejected a subject
        # literally named "A". Short names are real -- Go, R, C -- and a
        # stoplist that eats them is worse than one that lets a bare
        # article through, which pair-building and the relevance judge
        # would reject anyway.
        "one",
        "ones",
        "thing",
        "things",
        "what",
        "which",
        "who",
        "how",
        "why",
        "when",
        "where",
        "there",
        "here",
    }
)


def _names_something(text: str) -> bool:
    tokens = [t for t in re.split(r"[^\w-]+", text.lower()) if t]
    return any(t not in _NOT_A_SUBJECT for t in tokens)


def _strip_context(text: str) -> str:
    """Drop the trailing setting, keeping the subject that precedes it."""
    lowered = f" {text.lower()} "
    cut = len(text)
    for preposition in _CONTEXT_PREPOSITIONS:
        found = lowered.find(preposition)
        if found != -1:
            # -1 for the space this function prepended.
            cut = min(cut, max(0, found - 1))
    return text[:cut].strip()


def _clean(text: str) -> str:
    text = text.strip().strip("?.!,;:").strip()
    text = _LEADING_NOISE.sub("", text).strip()
    return text.strip("?.!,;:").strip()


def _split_subjects(text: str) -> list[str]:
    parts = [_clean(p) for p in _SUBJECT_SEPARATORS.split(text)]
    out: list[str] = []
    for part in parts:
        if part and part.lower() not in {p.lower() for p in out}:
            out.append(part)
    return out


def _accept(sides: list[str]) -> tuple[str, ...]:
    """Distinct, subject-like sides, or nothing.

    Deduplicated across both halves of the connective, not only within
    each: "how does RAG differ from RAG" produced the same subject
    twice and read as a two-way comparison.
    """
    distinct: list[str] = []
    for side in sides:
        if side.lower() in {d.lower() for d in distinct}:
            continue
        distinct.append(side)
    if len(distinct) < 2 or not all(_names_something(d) for d in distinct):
        return ()
    return tuple(distinct)


def comparison_sides(question: str) -> tuple[str, ...]:
    """The things the question puts on either side of a comparison.

    Returns an empty tuple when the wording is not an explicit
    comparison, which is the signal to leave the model's entities alone.

    Only the wording is read. A subject the question does not name
    cannot become a side, which is the whole point: "for language
    models" is a setting, and no amount of model confidence should turn
    it into a third thing being contrasted.
    """
    text = question.strip()
    if not text:
        return ()

    lowered = text.lower()

    # 1. Split connectives: the strongest signal, because each side is
    #    delimited by the connective itself.
    for connective in _SPLIT_CONNECTIVES:
        index = lowered.find(connective)
        if index == -1:
            continue
        left = _clean(text[:index])
        right = _strip_context(_clean(text[index + len(connective) :]))
        accepted = _accept(_split_subjects(left) + _split_subjects(right))
        if accepted:
            return accepted

    # 2. "difference between A and B": both sides after one phrase.
    match = _BETWEEN.search(text)
    if match:
        accepted = _accept(_split_subjects(_strip_context(_clean(match.group("rest")))))
        if accepted:
            return accepted

    # 3. "between A and B" anywhere in the question.
    match = _LOOSE_BETWEEN.search(text)
    if match:
        accepted = _accept(_split_subjects(_strip_context(_clean(match.group("rest")))))
        if accepted:
            return accepted

    # 4. "How do A, B and C differ?": subjects before a trailing verb.
    match = _TRAILING_DIFFER.match(text)
    if match:
        accepted = _accept(_split_subjects(_strip_context(_clean(match.group("subjects")))))
        if accepted:
            return accepted

    return ()


def reconcile_comparison_subjects(
    question: str, entities: list[str] | tuple[str, ...]
) -> tuple[str, ...]:
    """The comparison's sides, preferring the analyst's naming.

    The wording decides *how many* sides there are and which concepts
    they are; the analyst's entity list usually names them better
    (expanded acronyms, canonical capitalisation), so where an entity
    corresponds to a side, the entity's spelling is kept.

    An entity matching no side is dropped -- that is the defect this
    exists to stop. An empty result means the wording carried no
    explicit comparison, and the caller should fall back to the
    analyst's entities rather than refuse a run over a parse.
    """
    sides = comparison_sides(question)
    if not sides:
        return ()

    resolved: list[str] = []
    for side in sides:
        side_lower = side.lower()
        best: str | None = None
        for entity in entities:
            entity_lower = entity.strip().lower()
            if not entity_lower:
                continue
            corresponds = entity_lower in side_lower or side_lower in entity_lower
            # The longer spelling is the more complete name.
            if corresponds and (best is None or len(entity) > len(best)):
                best = entity.strip()
        chosen = best or side
        if chosen.lower() not in {r.lower() for r in resolved}:
            resolved.append(chosen)
    return tuple(resolved)


# ---------------------------------------------------------------------------
# Answer shape
# ---------------------------------------------------------------------------

# Checked in order. Earlier patterns are more specific readings of
# wording that a later pattern would also match: "what is the context
# window *size of* X" is a figure, and only looks like a definition
# because it opens with "what is".
_SHAPE_PATTERNS: tuple[tuple[QuestionType, re.Pattern[str]], ...] = (
    (
        QuestionType.NUMERIC,
        re.compile(
            r"\bhow (?:many|much|long|big|fast|often)\b"
            r"|\bwhat (?:percentage|proportion|fraction|share)\b"
            r"|\b(?:size|length|number|count|cost|price|latency|throughput|"
            r"accuracy|limit|capacity|window)\s+of\b",
            re.IGNORECASE,
        ),
    ),
    (
        QuestionType.PROCEDURAL,
        re.compile(
            r"\bhow (?:do|can|would|should)\s+(?:i|we|you|one)\b"
            r"|\bhow to\b|\bsteps?\s+(?:to|for)\b|\bwalk me through\b",
            re.IGNORECASE,
        ),
    ),
    (
        QuestionType.RECOMMENDATION,
        re.compile(
            r"^\s*should\s+(?:i|we|you|one)\b|\bwhich\s+should\b"
            r"|\bis it worth\b|\bshould (?:i|we) (?:use|adopt|choose|pick)\b",
            re.IGNORECASE,
        ),
    ),
    (
        QuestionType.TEMPORAL,
        re.compile(
            r"\bwhen (?:was|were|did|does|will|is)\b"
            r"|\btimeline\b|\bhistory of\b|\bas of\b"
            r"|\b(?:latest|current|most recent)\b",
            re.IGNORECASE,
        ),
    ),
    (
        QuestionType.LIST,
        re.compile(
            r"\b(?:causes?|reasons?|factors?|risks?|examples?|types?|kinds?|"
            r"categories|benefits?|drawbacks?|advantages?|disadvantages?|"
            r"use cases?|components?|stages?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        QuestionType.CAUSAL,
        re.compile(r"^\s*why\b|\bwhy (?:do|does|did|is|are)\b", re.IGNORECASE),
    ),
    (
        QuestionType.DEFINITION,
        re.compile(
            r"^\s*what (?:is|are)\b|\bdefine\b|\bdefinition of\b|\bmeaning of\b",
            re.IGNORECASE,
        ),
    ),
)


def shape_from_wording(question: str) -> QuestionType | None:
    """The shape the question's own wording asks for, or ``None``.

    ``None`` is the common and correct answer for anything that is not
    an explicit question form, and it means "the model's reading
    stands". This does not attempt to classify every question; it
    refuses to let an explicit one be reinterpreted.
    """
    text = (question or "").strip()
    if not text:
        return None

    # A comparison is decided by its sides rather than by a keyword: the
    # word "versus" in passing is not a comparison, and two named
    # subjects either side of a connective is.
    if len(comparison_sides(text)) >= 2:
        return QuestionType.COMPARISON

    for shape, pattern in _SHAPE_PATTERNS:
        if pattern.search(text):
            return shape
    return None
