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
from agentic_research.comparison import ComparisonPair, SideClaim, build_comparison_pairs


@dataclass(frozen=True)
class AnswerCoverage:
    contract: AnswerContract
    satisfied: tuple[str, ...]
    duplicates: tuple[str, ...]
    quality_by_slot: dict[str, float]
    comparison_pairs: tuple[ComparisonPair, ...] = ()
    """Complete contrasts assembled from verified side claims.

    The evidence that a comparison was answered, and the thing the
    report renders side by side. Empty for every non-comparison."""
    absent_entities: tuple[str, ...] = ()
    """Subjects the question named that no retrieved source mentions.

    Reported because "0 of 5 requirements covered" describes the
    engine, and the reader needs to know it describes the *question*.
    A run asked how two named language models differ, retrieved five
    sources, and published nothing -- correctly, because no source
    discussed either name. That is the fail-closed design working, and
    it was indistinguishable on screen from the engine being broken."""

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
            "absent_entities": list(self.absent_entities),
            "comparison_pairs": [p.to_dict() for p in self.comparison_pairs],
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

        # First, because it is the *reason* for everything after it. A
        # reader told only that nothing was answered will conclude the
        # engine failed; a reader told no source mentions the subject
        # can see that the question was the thing that could not be
        # answered.
        if self.absent_entities:
            named = ", ".join(f"\u201c{e}\u201d" for e in self.absent_entities)
            out.append(
                f"No retrieved source mentions {named}. Either no reachable "
                "source covers it, or the question names something that does "
                "not exist -- so nothing below should be read as an answer "
                "about it. This is a refusal to invent one, not a retrieval "
                "failure."
            )

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
            subjects = (
                " and ".join(self.contract.subjects_to_span)
                or " and ".join(self.contract.entities)
                or "the subjects"
            )
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
    # The comparison's declared sides, not every concept the question
    # named. Asked how retrieval-augmented generation differs from
    # fine-tuning *for language models*, this required a published claim
    # to mention "language models" -- a setting, never a side -- and so
    # reported three correct, cited, relevant claims as not having
    # answered the question.
    subjects = contract.subjects_to_span
    if not subjects or not claim_texts:
        return False
    blob = "\n".join(claim_texts)
    return all(_mentions(blob, subject) for subject in subjects)


def _entities_absent_from(contract: AnswerContract, source_texts: Sequence[str]) -> tuple[str, ...]:
    """Subjects the question named that appear in no retrieved source.

    Deliberately checked against the *sources*, not the published
    claims. A claim can be absent for many reasons -- the synthesiser
    wrote badly, a guard refused it, the budget ran out. A subject
    missing from every source it searched is a different and much
    stronger fact: there was nothing to answer from.

    Reuses `_mentions`, so it inherits the same matching the relevance
    check uses. An empty source list returns nothing rather than every
    entity: a run that retrieved no sources at all has a different
    problem, and reporting it as a false premise would be wrong.
    """
    # Sides first where the question has them: a comparison whose
    # subject no source covers cannot be answered at all, whereas a
    # missing context noun is usually harmless.
    subjects = contract.comparison_subjects or contract.entities
    if not subjects or not source_texts:
        return ()
    blob = "\n".join(source_texts)
    return tuple(e for e in subjects if not _mentions(blob, e))


def assess_coverage(
    contract: AnswerContract,
    published_slots: list[str] | tuple[str, ...],
    *,
    quality_by_slot: dict[str, float] | None = None,
    claims: Sequence[SideClaim] | None = None,
    claim_texts: Sequence[str] | None = None,
    source_texts: Sequence[str] | None = None,
) -> AnswerCoverage:
    """How much of the contract the published claims filled.

    A slot only counts when the contract asked for it: a claim
    declaring something the question never required satisfies nothing.

    ``claims`` carries each published claim with the slot it declared,
    so a comparison's core slot can be discharged by verified side
    assertions assembled into a pair. The slot and the text must travel
    together: the two were previously passed as separate lists built
    with *different* filters, which nothing indexed together yet and
    which no longer holds once they are paired.

    ``claim_texts`` remains for callers that only have the texts. It
    cannot discharge a contrast on its own, because a contrast now
    requires knowing which dimension each claim was declared against.
    """
    if claims is None and claim_texts:
        claims = [SideClaim(subject="", text=t, answer_slot="") for t in claim_texts]
    counts = Counter(name for name in published_slots if contract.has_slot(name))
    satisfied = set(counts)

    # A comparison answered by one verified atomic claim per subject,
    # assembled into a structured pair rather than a fused sentence.
    #
    # Stronger than the rule it replaces. `_spans_every_entity` pooled
    # every published claim and asked only whether the subjects were
    # mentioned somewhere between them, so two claims about different
    # things satisfied the contrast. A pair requires a verified claim
    # per subject *within one dimension*.
    pairs = build_comparison_pairs(contract, list(claims or ()))
    if pairs and "direct_contrast" not in satisfied:
        satisfied.add("direct_contrast")

    return AnswerCoverage(
        contract=contract,
        satisfied=tuple(sorted(satisfied)),
        duplicates=tuple(sorted(name for name, n in counts.items() if n > 1)),
        quality_by_slot=dict(quality_by_slot or {}),
        absent_entities=_entities_absent_from(contract, source_texts or ()),
        comparison_pairs=pairs,
    )
