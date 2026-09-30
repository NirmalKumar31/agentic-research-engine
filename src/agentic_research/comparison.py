"""A comparison represented as verified sides, not as fused prose.

The conflict this resolves. A comparison's core requirement is an
explicit statement of how the subjects differ, and the honest form of
such a statement asserts two things at once:

    Fine-tuning changes a model's behavioural patterns, while
    retrieval-augmented generation changes the information the model can
    access.

The atomicity guard refuses that, correctly: every other guard reasons
about "the sentence that supports this claim", and a fused claim hands
it two sentences' worth of assertion with one quote. Across nine hosted
runs the synthesiser produced exactly one `direct_contrast` claim and
atomicity refused it. A live run then published three supported,
relevant, cited claims about the two subjects and reported that it had
not answered the question.

The wrong fixes, both rejected:

* let compound contrast claims past atomicity -- that is laundering,
  and it would let a claim whose second half no quote supports publish
  on the strength of its first half;
* let repair delete the unsupported half and present the remainder as a
  repaired form of the original -- the reader would see a narrower
  claim than the one the evidence was checked against.

The right one: keep atomicity, verify **one atomic assertion per
side**, and assemble the contrast as structure rather than as a new
sentence. Nothing here is generated. A pair holds claims that already
passed entailment, the deterministic guards and the relevance
judgement; if a side has no such claim, there is no pair, and the
contrast is reported as unfilled.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from agentic_research.answer_contract import AnswerContract, QuestionType
from agentic_research.citations.relevance import _mentions


@dataclass(frozen=True)
class SideClaim:
    """One published, verified claim, and the subject it speaks about."""

    subject: str
    text: str
    answer_slot: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComparisonPair:
    """A contrast assembled from verified side claims.

    ``dimension`` is the answer slot the sides were declared against.
    Where the question named its own dimensions those slots exist by
    name (``latency``, ``cost``); otherwise it is the contract's generic
    ``dimension`` slot, which is the strongest available signal that two
    claims address a common axis and is deliberately not stronger than
    that -- see :func:`build_comparison_pairs`.
    """

    dimension: str
    sides: tuple[SideClaim, ...]

    @property
    def subjects(self) -> tuple[str, ...]:
        seen: list[str] = []
        for side in self.sides:
            if side.subject not in seen:
                seen.append(side.subject)
        return tuple(seen)

    def covers(self, subjects: Sequence[str]) -> bool:
        mine = {s.lower() for s in self.subjects}
        return all(s.lower() in mine for s in subjects)

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "sides": [
                {
                    "subject": s.subject,
                    "text": s.text,
                    "answer_slot": s.answer_slot,
                    "evidence_ids": list(s.evidence_ids),
                }
                for s in self.sides
            ],
        }


# Slots that describe the comparison as a whole rather than one axis of
# it, plus the generic placeholder.
#
# `dimension` is excluded because it names no axis. Two claims declaring
# it may address entirely different properties -- "LLMs predict tokens"
# and "neural networks classify images" are about task and mechanism --
# and counting that as a contrast is the shortcut the audit flagged. A
# pair now requires a slot that names the axis, which exists whenever
# the analysis proposed dimensions for the comparison. A claim
# declaring the generic slot still publishes; it just cannot complete
# the contrast on its own.
_NON_DIMENSION_SLOTS = frozenset({"direct_contrast", "relationship", "dimension"})


class RelationshipKind(StrEnum):
    """What kind of relationship a claim asserts between the subjects.

    Only some kinds explain why a direct contrast is inappropriate.
    Asked how a large language model differs from a neural network, the
    honest answer is that one is a kind of the other -- there is no
    contrast to find, and demanding one makes the engine wrong about
    itself. "Both are used in industry" explains nothing and must not
    stand in for a comparison.
    """

    SUBTYPE = "subtype"
    EQUIVALENCE = "equivalence"
    COMPONENT = "component"
    DEPENDENCY = "dependency"
    GENERAL = "general"


# Ordered: the first pattern to match decides, so the more specific
# readings are tried before the looser ones.
_RELATIONSHIP_PATTERNS: tuple[tuple[RelationshipKind, re.Pattern[str]], ...] = (
    (
        RelationshipKind.SUBTYPE,
        re.compile(
            r"\b(?:is|are|as)\s+(?:a|an|one)?\s*"
            r"(?:kind|type|form|subset|subclass|subtype|category|instance|example|"
            r"variety|species)\s+of\b"
            r"|\bis a specialis(?:ed|ation)\b|\bfalls under\b|\bbelongs to the\b",
            re.IGNORECASE,
        ),
    ),
    (
        RelationshipKind.EQUIVALENCE,
        re.compile(
            r"\b(?:is|are)\s+(?:the same as|equivalent to|identical to|"
            r"synonymous with|another name for|the same thing as)\b"
            r"|\brefer to the same\b",
            re.IGNORECASE,
        ),
    ),
    (
        RelationshipKind.COMPONENT,
        re.compile(
            r"\b(?:is|are)\s+(?:a|an|one)?\s*"
            r"(?:component|part|member|element|ingredient|stage|step|layer)\s+of\b"
            r"|\bconsists? of\b|\bis composed of\b|\bcontains\b",
            re.IGNORECASE,
        ),
    ),
    (
        RelationshipKind.DEPENDENCY,
        re.compile(
            r"\b(?:depends? on|requires?|relies on|is built on|is based on|"
            r"is implemented (?:with|using)|uses)\b",
            re.IGNORECASE,
        ),
    ),
)

# Kinds that resolve why a direct contrast is inappropriate.
#
# DEPENDENCY is excluded deliberately. "RAG uses a language model" is a
# true relationship and it does not answer how RAG differs from
# fine-tuning -- two things can depend on each other and still need
# contrasting. Only containment and identity remove the contrast.
_DISCHARGING_KINDS = frozenset(
    {RelationshipKind.SUBTYPE, RelationshipKind.EQUIVALENCE, RelationshipKind.COMPONENT}
)


def dynamically_discharging_slots(contract: AnswerContract) -> frozenset[str]:
    """Slots that can discharge a core slot without declaring it.

    The contract cannot express "this slot answers the question when
    the claim asserts a particular kind of thing", so these routes live
    in code. Anything reasoning about which slots the answer turns on
    -- the relevance judge's authority, for one -- has to consult this
    as well as `satisfied_by`, or dropping a static alternative
    silently narrows that reasoning.
    """
    if contract.question_type is QuestionType.COMPARISON:
        return frozenset({"relationship"})
    return frozenset()


def relationship_kind(text: str) -> RelationshipKind:
    """Classify a relationship claim. Unrecognised wording is GENERAL."""
    for kind, pattern in _RELATIONSHIP_PATTERNS:
        if pattern.search(text or ""):
            return kind
    return RelationshipKind.GENERAL


def discharges_contrast(contract: AnswerContract, claims: Sequence[SideClaim]) -> tuple[bool, str]:
    """Whether a relationship claim removes the need for a contrast.

    Requires all three of: the claim declares `relationship`; its kind
    is one that explains away the contrast; and it speaks about every
    comparison subject. A subtype claim about only one of two subjects
    establishes nothing about the pair.
    """
    if contract.question_type is not QuestionType.COMPARISON:
        return False, ""
    subjects = contract.subjects_to_span
    if len(subjects) < 2:
        return False, ""
    for claim in claims:
        if claim.answer_slot != "relationship":
            continue
        kind = relationship_kind(claim.text)
        if kind not in _DISCHARGING_KINDS:
            continue
        if all(_mentions(claim.text, subject) for subject in subjects):
            return True, kind.value
    return False, ""


def build_comparison_pairs(
    contract: AnswerContract,
    claims: Sequence[SideClaim] | Sequence[tuple[str, str]],
) -> tuple[ComparisonPair, ...]:
    """Assemble complete contrasts from published side claims.

    A pair is complete only when, **within a single dimension**, there
    is a verified claim speaking about every comparison subject. Two
    claims that merely mention both subjects from different slots do not
    form a contrast: "RAG retrieves at query time" and "fine-tuning is a
    kind of training" are both true and are not a comparison of
    anything.

    The honest limit, stated rather than implied: where the question
    named no dimensions of its own, "same dimension" means "declared
    against the same generic slot". That is a proxy. Two claims can
    share that slot and still address different axes, and nothing here
    can tell. What stops the original defect -- five supported claims
    about one of two subjects, published as an answer to how they differ
    -- does not depend on the proxy: every subject must be spoken about
    by its own verified claim.
    """
    if contract.question_type is not QuestionType.COMPARISON:
        return ()
    subjects = contract.subjects_to_span
    if len(subjects) < 2:
        return ()

    normalised = [c if isinstance(c, SideClaim) else SideClaim("", c[1], c[0]) for c in claims]

    by_dimension: dict[str, list[SideClaim]] = {}
    for claim in normalised:
        slot = (claim.answer_slot or "").strip()
        if not slot or slot in _NON_DIMENSION_SLOTS or not contract.has_slot(slot):
            continue
        for subject in subjects:
            if _mentions(claim.text, subject):
                by_dimension.setdefault(slot, []).append(
                    SideClaim(
                        subject=subject,
                        text=claim.text,
                        answer_slot=slot,
                        evidence_ids=claim.evidence_ids,
                    )
                )

    pairs: list[ComparisonPair] = []
    for slot, sides in by_dimension.items():
        pair = ComparisonPair(dimension=slot, sides=tuple(sides))
        if pair.covers(subjects):
            pairs.append(pair)
    # Named dimensions before the generic placeholder: a contrast on
    # "latency" is a better answer than one on "dimension".
    pairs.sort(key=lambda p: (p.dimension == "dimension", p.dimension))
    return tuple(pairs)


def render_pairs(pairs: Sequence[ComparisonPair]) -> str:
    """The contrast as a side-by-side table.

    Markdown rather than prose on purpose. Writing "X does this while Y
    does that" would be a new sentence no quote was checked against --
    the laundering this module exists to avoid. A table asserts only
    that these verified claims sit beside each other, which is exactly
    what was established.
    """
    if not pairs:
        return ""
    lines: list[str] = []
    for pair in pairs:
        label = pair.dimension.replace("_", " ")
        lines.append(f"**{label}**\n")
        lines.append("| Subject | What the evidence supports |")
        lines.append("| --- | --- |")
        for side in pair.sides:
            cell = side.text.replace("|", "\\|")
            citations = " ".join(f"[{e.split('-')[0]}]" for e in side.evidence_ids[:1])
            lines.append(f"| {side.subject} | {cell} {citations}".rstrip() + " |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def pairs_from_payload(payload: object) -> tuple[ComparisonPair, ...]:
    """Rehydrate contrasts from a serialised coverage assessment.

    State is checkpointed and serialised, so `answer_coverage` is a plain
    dict by the time a renderer sees it. This reads the pairs back; it does
    not recompute them, so a report cannot disagree with the coverage that
    produced it.

    It lives here rather than beside one renderer because there are two.
    The first version sat in `graph/nodes/reporting.py`, and only that
    renderer was given the pairs -- so `finalize` rendered the contrast
    table and `runner._render`, which produces the `markdown` the web
    result actually carries, did not. A live comparison run found 2
    complete pairs, reported the question answered, and shipped a report
    with no table in it. The test written for that fix asserted the one
    call site it knew about and passed throughout.
    """
    if not isinstance(payload, dict):
        return ()
    raw = payload.get("comparison_pairs") or []
    pairs: list[ComparisonPair] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        sides = tuple(
            SideClaim(
                subject=str(side.get("subject", "")),
                text=str(side.get("text", "")),
                answer_slot=str(side.get("answer_slot", "")),
                evidence_ids=tuple(side.get("evidence_ids") or ()),
            )
            for side in entry.get("sides") or []
            if isinstance(side, dict)
        )
        if sides:
            pairs.append(ComparisonPair(dimension=str(entry.get("dimension", "")), sides=sides))
    return tuple(pairs)
