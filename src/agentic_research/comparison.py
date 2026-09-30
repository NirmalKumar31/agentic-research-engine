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

from collections.abc import Sequence
from dataclasses import dataclass

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
# it. A claim declaring one of these is not a side assertion on a
# dimension, so it cannot combine with another to form a pair.
_NON_DIMENSION_SLOTS = frozenset({"direct_contrast", "relationship"})


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
