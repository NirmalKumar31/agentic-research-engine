"""Two causal questions, two contracts, no substitution between them.

The contract used to let `candidate_drivers` discharge `causal_evidence`,
and the comment justifying it admitted the cost: for a genuine causal
test, naming a plausible driver discharged the core slot without
establishing causation, with a relevance model as the only backstop.
That is the association-for-causation substitution the entire
verification layer exists to refuse, and a model is not an acceptable
sole guard for it.

The two forms are now separate shapes:

* "does X cause Y?"  -> CAUSAL, core slot `causal_evidence`
* "what causes X?"   -> CAUSAL_DRIVERS, core slot `candidate_drivers`

There is no route from one to the other. A driver cannot complete a
yes/no causal claim, and evidence of established causation is not
demanded of a question that only asked which factors contribute.
"""

from __future__ import annotations

import pytest

from agentic_research.answer_contract import (
    CANONICAL_SLOTS,
    QuestionType,
    build_contract,
)
from agentic_research.answer_coverage import assess_coverage, states_causation
from agentic_research.comparison import SideClaim
from agentic_research.question_form import shape_from_wording


class TestTheTwoFormsAreDistinguishedFromWording:
    @pytest.mark.parametrize(
        "question",
        [
            "Does smoking cause lung cancer?",
            "Can overfitting cause poor generalization?",
            "Did the regulation lead to lower prices?",
            "Is dropout responsible for better generalization?",
            "Do larger datasets result in better accuracy?",
            "Was the outage caused by the deployment?",
        ],
    )
    def test_a_yes_no_test_reads_as_causal(self, question: str) -> None:
        assert shape_from_wording(question) is QuestionType.CAUSAL

    @pytest.mark.parametrize(
        "question",
        [
            "What causes hallucination in large language models?",
            "What are the main causes of overfitting?",
            "Why do large language models hallucinate?",
            "What factors contribute to overfitting?",
            "What are the main reasons LLMs hallucinate?",
            "What drives up inference cost?",
        ],
    )
    def test_a_driver_question_reads_as_causal_drivers(self, question: str) -> None:
        assert shape_from_wording(question) is QuestionType.CAUSAL_DRIVERS

    def test_the_word_cause_does_not_by_itself_make_a_test(self) -> None:
        """ "What are the main causes of X" contains "cause" and is not
        asking whether one specific thing causes another. Reading it as
        a test is what made the substitution look necessary."""
        assert (
            shape_from_wording("What are the main causes of overfitting?")
            is QuestionType.CAUSAL_DRIVERS
        )


class TestNoRouteBetweenTheContracts:
    def test_causal_evidence_has_no_declared_alternative(self) -> None:
        slot = next(s for s in CANONICAL_SLOTS[QuestionType.CAUSAL] if s.name == "causal_evidence")
        assert slot.satisfied_by == ()

    def test_a_causal_test_contract_has_no_drivers_slot(self) -> None:
        """Structural, not a rule that can be forgotten: a claim cannot
        declare a slot the contract does not contain."""
        names = {s.name for s in CANONICAL_SLOTS[QuestionType.CAUSAL]}
        assert "candidate_drivers" not in names

    def test_a_driver_contract_does_not_demand_established_causation(self) -> None:
        core = [s.name for s in CANONICAL_SLOTS[QuestionType.CAUSAL_DRIVERS] if s.core]
        assert core == ["candidate_drivers"]

    def test_a_driver_contract_still_asks_about_the_limits(self) -> None:
        names = {s.name for s in CANONICAL_SLOTS[QuestionType.CAUSAL_DRIVERS]}
        assert "causation_limits" in names


class TestAssociationCannotDischargeACausalTest:
    """Adversarial: the evidence forms a causal test may be offered."""

    def _coverage(self, text: str):
        contract = build_contract(
            "Does smoking cause lung cancer?",
            QuestionType.CAUSAL,
            entities=("smoking", "lung cancer"),
        )
        return assess_coverage(
            contract,
            ["causal_evidence"],
            claims=[SideClaim(subject="", text=text, answer_slot="causal_evidence")],
        )

    @pytest.mark.parametrize(
        "text",
        [
            "Smoking is associated with lung cancer.",
            "Smoking is correlated with a higher incidence of lung cancer.",
            "There is a strong association between smoking and lung cancer.",
            "Smoking is linked to lung cancer in observational cohorts.",
            "Smoking is predictive of lung cancer diagnosis.",
            "Rates of lung cancer are higher among smokers.",
        ],
    )
    def test_association_does_not_answer_it(self, text: str) -> None:
        coverage = self._coverage(text)
        assert not coverage.answered, text

    @pytest.mark.parametrize(
        "text",
        [
            "Smoking causes lung cancer.",
            "Randomised trials show the intervention leads to lower mortality.",
            "Tobacco smoke produces DNA adducts that result in malignant transformation.",
            "Lung cancer risk falls after cessation because exposure is removed.",
        ],
    )
    def test_stated_causation_does_answer_it(self, text: str) -> None:
        """Non-vacuity: the rule above must not refuse everything."""
        assert self._coverage(text).answered, text

    def test_a_mixed_sentence_is_refused(self) -> None:
        """Conservative on purpose. "X is associated with Y, which may
        cause Z" is not evidence that X causes Y, and a sentence
        carrying both vocabularies is exactly where a reader is most
        likely to be misled."""
        assert not states_causation(
            "Overfitting is linked to small datasets, which may cause poor generalization."
        )

    @pytest.mark.parametrize(
        "text",
        [
            "It remains unclear whether smoking causes lung cancer in this cohort.",
            "The evidence does not establish that smoking causes lung cancer.",
            "Whether smoking causes lung cancer cannot be determined from this data.",
            "There is no evidence that the drug causes the observed effect.",
            "The causal link is disputed.",
        ],
    )
    def test_explicit_uncertainty_does_not_answer_it(self, text: str) -> None:
        """These carry the strongest causal vocabulary in a report and
        name causation in order to deny it. Matching causal words
        without reading the frame counts them as evidence."""
        assert not self._coverage(text).answered, text

    def test_a_plausible_driver_does_not_complete_a_causal_test(self) -> None:
        """The substitution, refused at the structural level: the slot
        a driver would declare does not exist on this contract, so the
        claim is not even eligible to satisfy the core requirement."""
        contract = build_contract(
            "Does smoking cause lung cancer?",
            QuestionType.CAUSAL,
            entities=("smoking", "lung cancer"),
        )
        coverage = assess_coverage(
            contract,
            ["candidate_drivers"],
            claims=[
                SideClaim(
                    subject="",
                    text="Smoking is one of several factors in lung cancer incidence.",
                    answer_slot="candidate_drivers",
                )
            ],
        )
        assert not coverage.answered
        assert "candidate_drivers" not in coverage.satisfied


class TestADriverQuestionIsAnsweredByDrivers:
    def test_drivers_answer_it(self) -> None:
        contract = build_contract(
            "What are the main causes of overfitting?",
            QuestionType.CAUSAL_DRIVERS,
            entities=("overfitting",),
        )
        coverage = assess_coverage(
            contract,
            ["candidate_drivers"],
            claims=[
                SideClaim(
                    subject="",
                    text=(
                        "When a model lacks a clear signal from training data it fits random noise."
                    ),
                    answer_slot="candidate_drivers",
                )
            ],
        )
        assert coverage.answered

    def test_the_causation_limit_is_reported_when_absent(self) -> None:
        """Non-core by deliberate deviation, but never silent: the
        coverage narrative names it as unestablished."""
        contract = build_contract(
            "What are the main causes of overfitting?",
            QuestionType.CAUSAL_DRIVERS,
            entities=("overfitting",),
        )
        coverage = assess_coverage(
            contract,
            ["candidate_drivers"],
            claims=[
                SideClaim(
                    subject="",
                    text="Excess model capacity contributes to overfitting.",
                    answer_slot="candidate_drivers",
                )
            ],
        )
        assert any(
            "causation" in gap.lower() or "association" in gap.lower()
            for gap in coverage.limitations()
        )
