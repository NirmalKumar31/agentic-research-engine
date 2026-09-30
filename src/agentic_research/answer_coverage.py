"""Which parts of the question the report actually answered.

A report could finish with five published claims and answer nothing
that was asked, and there was no stage at which anything noticed. The
contract says what an answer requires; this says how much of it
arrived, and turns the rest into limitations that name what is
missing rather than gesturing at it.

The rule that matters: a report satisfying no core slot has not
answered the question, and must say so. Presenting background
material as the answer is the failure this exists to prevent.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from agentic_research.answer_contract import AnswerContract, AnswerSlot, QuestionType
from agentic_research.citations.relevance import _mentions


@dataclass(frozen=True)
class AnswerCoverage:
    contract: AnswerContract
    satisfied: tuple[str, ...]
    duplicates: tuple[str, ...]
    quality_by_slot: dict[str, float]

    @property
    def required(self) -> tuple[AnswerSlot, ...]:
        return self.contract.required_slots

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.required if s.name not in self.satisfied)

    def _discharged(self, slot: AnswerSlot) -> bool:
        """Whether this core slot was filled, directly or by an
        alternative the question also admits."""
        return slot.name in self.satisfied or any(
            name in self.satisfied for name in slot.satisfied_by
        )

    @property
    def missing_core(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.contract.core_slots if not self._discharged(s))

    @property
    def answered(self) -> bool:
        """Every core slot discharged.

        Not "at least one", which an earlier version of this docstring
        claimed while the code required all of them. The code was
        right: a multi-part question makes each part its own core slot,
        and a report answering one part of three has not answered the
        question. Every canonical question type has exactly one core
        slot, so the two readings only diverge for multi-part
        questions -- which is precisely where the strict one matters.

        Not "some claims published" either. A comparison with two
        definitions published has answered nothing.
        """
        return bool(self.contract.core_slots) and not self.missing_core

    @property
    def partially_answered(self) -> bool:
        return bool(self.satisfied) and not self.answered

    def to_dict(self) -> dict[str, object]:
        return {
            "question_type": str(self.contract.question_type),
            "required_slots": [s.name for s in self.required],
            "satisfied_slots": list(self.satisfied),
            "missing_slots": list(self.missing),
            "missing_core_slots": list(self.missing_core),
            "duplicate_slots": list(self.duplicates),
            "answered": self.answered,
            "quality_by_slot": dict(self.quality_by_slot),
        }

    def limitations(self) -> list[str]:
        """What to tell the reader, naming the gap rather than hinting.

        Ordered with the honest headline first: if nothing core was
        answered, that is the first thing a reader needs, before any
        detail about which sub-part is thin.
        """
        out: list[str] = []

        if not self.contract.usable:
            return [
                "The question could not be turned into a checkable set of "
                f"requirements: {self.contract.unusable_reason}. "
                "Nothing below should be read as an answer to it."
            ]

        if not self.satisfied:
            out.append(
                "This research did not answer the question. No claim survived "
                "verification, and the material below is background rather than "
                "an answer."
            )
        elif self.missing_core:
            out.append("This research did not answer the question. " + self._core_sentence())

        for name in self.missing:
            if name in self.missing_core:
                continue
            slot = next(s for s in self.required if s.name == name)
            out.append(f"The evidence did not establish {slot.description.lower()}.")

        for name in self.duplicates:
            out.append(
                f"More than one published claim fills the {name} slot; they may repeat each other."
            )

        return out

    def _core_sentence(self) -> str:
        qtype = self.contract.question_type
        if qtype is QuestionType.COMPARISON:
            subjects = " and ".join(self.contract.entities) or "the subjects"
            return (
                f"Nothing published states how {subjects} differ; what survived "
                "describes them separately."
            )
        if qtype is QuestionType.CAUSAL:
            return (
                "Nothing published establishes a cause; what survived describes "
                "what happened without showing why."
            )
        if qtype is QuestionType.SYNTHESIS:
            missing = ", ".join(
                next(s.description for s in self.required if s.name == name)
                for name in self.missing_core
            )
            return f"These parts were not answered: {missing}."
        missing = ", ".join(self.missing_core)
        return f"The required {missing} was not established."


def _spans_every_entity(contract: AnswerContract, claim_texts: Sequence[str]) -> bool:
    """Whether the published claims, between them, speak about each
    subject the question named.

    Not whether any single claim contrasts them. A contrast asserts two
    things and the atomicity guard refuses compound claims, because
    every other guard reasons about "the sentence that supports this
    claim" and a fused claim hands it two. So a comparison's core slot
    was close to unfillable: nine hosted runs produced exactly one
    `direct_contrast` claim and it was refused on atomicity.

    What a reader can compare is two atomic claims, one about each
    subject. That is what this checks.

    It is deliberately stronger than the defect the contract was built
    to stop. That defect was five supported claims about *one* of two
    subjects, published as an answer to how they differ; covering one
    subject still fails here, because every entity must be spoken
    about. What changes is that covering both, in separate verifiable
    claims, now counts -- where before it was reported as not having
    answered at all.
    """
    if not contract.entities or not claim_texts:
        return False
    blob = "\n".join(claim_texts)
    return all(_mentions(blob, entity) for entity in contract.entities)


def assess_coverage(
    contract: AnswerContract,
    published_slots: list[str] | tuple[str, ...],
    *,
    quality_by_slot: dict[str, float] | None = None,
    claim_texts: Sequence[str] | None = None,
) -> AnswerCoverage:
    """How much of the contract the published claims filled.

    A slot only counts when the contract asked for it: a claim
    declaring something the question never required satisfies nothing.

    ``claim_texts`` lets a comparison's core slot be discharged by
    claims that between them speak about every subject named -- see
    :func:`_spans_every_entity`. Omitting it only loses that route;
    every other rule is unchanged.
    """
    counts = Counter(name for name in published_slots if contract.has_slot(name))
    satisfied = set(counts)

    # A comparison answered by one atomic claim per subject.
    if (
        contract.question_type is QuestionType.COMPARISON
        and "direct_contrast" not in satisfied
        and satisfied
        and _spans_every_entity(contract, claim_texts or ())
    ):
        satisfied.add("direct_contrast")

    return AnswerCoverage(
        contract=contract,
        satisfied=tuple(sorted(satisfied)),
        duplicates=tuple(sorted(name for name, n in counts.items() if n > 1)),
        quality_by_slot=dict(quality_by_slot or {}),
    )
