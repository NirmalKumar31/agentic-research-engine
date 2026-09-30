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

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agentic_research.answer_contract import AnswerContract, AnswerSlot, QuestionType
from agentic_research.citations.relevance import _mentions
from agentic_research.comparison import (
    ComparisonPair,
    SideClaim,
    build_comparison_pairs,
    discharges_contrast,
)


@dataclass(frozen=True)
class AnswerCoverage:
    contract: AnswerContract
    satisfied: tuple[str, ...]
    duplicates: tuple[str, ...]
    quality_by_slot: dict[str, float]
    relationship_discharge: str = ""
    """The relationship kind that stood in for a contrast, if one did.

    Recorded because "answered" then rests on a different fact from
    usual -- not that the engine found a contrast, but that it found
    there was none to find -- and a reader is entitled to see which."""
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
            "relationship_discharge": self.relationship_discharge,
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


# Language that reports a relationship without asserting direction or
# production. A claim built only from these may be true, supported and
# relevant, and it still does not answer "does X cause Y".
_ASSOCIATION_ONLY = re.compile(
    r"\b(?:associated with|association between|correlat(?:ed|ion)|linked to|"
    r"link between|related to|relationship between|co-?occur|accompan(?:y|ies|ied)|"
    r"observed (?:alongside|together)|predicts?|predictive of)\b",
    re.IGNORECASE,
)

# Language that asserts one thing produces another.
_STATES_CAUSATION = re.compile(
    r"\b(?:causes?|caused|causing|leads? to|led to|results? in|resulted in|"
    r"produces?|produced|drives?|driven by|because|due to|"
    r"responsible for|gives? rise to|induces?|triggers?)\b",
    re.IGNORECASE,
)


# Sentences that report the absence or uncertainty of a finding. These
# often contain the strongest causal vocabulary in the whole report --
# "it is unclear whether X causes Y" names causation in order to deny
# it -- so they must be checked before the causal patterns rather than
# after.
_UNRESOLVED = re.compile(
    r"\b(?:unclear|unknown|uncertain|not established|does not establish|"
    r"cannot be determined|no evidence|insufficient evidence|"
    r"remains? to be|has not been shown|inconclusive|disputed|"
    r"whether or not)\b",
    re.IGNORECASE,
)


def states_causation(text: str) -> bool:
    """Whether a claim asserts causation rather than association.

    Used to admit claims to the `causal_evidence` slot of a yes/no
    causal question. `causal_guard` already refuses a claim that asserts
    causation its quote does not carry; this is the other direction --
    a claim asserting only association passes every guard, because it
    over-claims nothing, and could still satisfy the core slot of "does
    X cause Y?".

    Deliberately conservative in three ways, each because the failure
    mode is a reader being misled rather than a claim being lost:

    * association language alone is refused even when causal words also
      appear, because "X is associated with Y, which may cause Z" is
      not evidence that X causes Y;
    * a sentence reporting that causation is unclear or unestablished is
      refused, even though it contains the strongest causal vocabulary
      in the report -- it names causation to deny it;
    * anything unrecognised is refused rather than admitted.
    """
    if not text or not text.strip():
        return False
    # Checked first: a sentence denying a causal finding contains the
    # causal vocabulary precisely in order to deny it.
    if _UNRESOLVED.search(text):
        return False
    if _ASSOCIATION_ONLY.search(text):
        return False
    return bool(_STATES_CAUSATION.search(text))


# Slots whose satisfaction needs more than a claim declaring them.
# Keyed by slot name; each test receives the claim text.
_SLOT_ADMISSION: dict[str, Callable[[str], bool]] = {
    "causal_evidence": states_causation,
}


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

    # Some slots need more than a claim pointing at them. A claim
    # declaring `causal_evidence` while asserting only association is
    # true, supported, relevant -- and not an answer to whether X
    # causes Y. Without the texts the admission test cannot run, so the
    # slot keeps the old behaviour rather than being refused for a
    # reason the caller could not supply.
    if claims:
        by_slot: dict[str, list[str]] = {}
        for claim in claims:
            if claim.answer_slot:
                by_slot.setdefault(claim.answer_slot, []).append(claim.text)
        for slot, admits in _SLOT_ADMISSION.items():
            if slot in satisfied and slot in by_slot:
                satisfied = (
                    satisfied - {slot}
                    if not any(admits(text) for text in by_slot[slot])
                    else satisfied
                )

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

    # A relationship may stand in for a contrast, but only when it is
    # the kind that explains why a contrast is inappropriate -- one
    # subject being a kind of, equivalent to, or a component of the
    # other. This used to be a static `satisfied_by` on the slot, which
    # could not express "only for certain claims" and so admitted every
    # relationship claim, including "both are used with language
    # models".
    discharged, kind = discharges_contrast(contract, list(claims or ()))
    if discharged and "direct_contrast" not in satisfied:
        satisfied.add("direct_contrast")
        relationship_discharge = kind
    else:
        relationship_discharge = ""

    return AnswerCoverage(
        contract=contract,
        satisfied=tuple(sorted(satisfied)),
        duplicates=tuple(sorted(name for name, n in counts.items() if n > 1)),
        quality_by_slot=dict(quality_by_slot or {}),
        absent_entities=_entities_absent_from(contract, source_texts or ()),
        comparison_pairs=pairs,
        relationship_discharge=relationship_discharge,
    )
