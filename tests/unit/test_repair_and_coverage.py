"""Bounded rewriting, and whether the report answered the question.

Repair exists because three of six claims in the frozen set were
refused for how they were phrased rather than for lacking support.
It is dangerous for the same reason it is useful: a rewrite that may
change anything can turn an association into a cause, and one that
may delete anything can drop the unsupported half of a bundled claim
and publish the rest. Most of these tests are about refusing that.

Coverage exists because a report could finish with five published
claims and answer nothing that was asked.
"""

from __future__ import annotations

import pytest

from agentic_research.answer_contract import QuestionType, build_contract
from agentic_research.answer_coverage import assess_coverage
from agentic_research.citations.repair import (
    REPAIRABLE_GUARDS,
    UNREPAIRABLE_GUARDS,
    failed_guard_names,
    is_repairable,
    validate_rewrite,
)

GUARD = "every cited quote failed a deterministic guard: {}"


class TestWhatMayBeRepaired:
    @pytest.mark.parametrize("guard", sorted(REPAIRABLE_GUARDS))
    def test_a_wording_guard_is_eligible(self, guard: str) -> None:
        ok, _ = is_repairable(GUARD.format(guard), every_proposition_supported=True)
        assert ok

    @pytest.mark.parametrize("guard", sorted(UNREPAIRABLE_GUARDS))
    def test_an_assertion_guard_is_not(self, guard: str) -> None:
        """Causation, exclusivity, framing and hedging are about what
        the claim says, not how. Rephrasing them is laundering."""
        ok, why = is_repairable(GUARD.format(guard), every_proposition_supported=True)
        assert not ok
        assert "not how" in why or "unknown" in why

    def test_unsupported_evidence_is_never_eligible(self) -> None:
        """Repair is not a second chance at evidence."""
        ok, why = is_repairable(GUARD.format("atomicity"), every_proposition_supported=False)
        assert not ok
        assert "does not support" in why

    def test_a_threshold_failure_is_not_a_guard_failure(self) -> None:
        ok, _ = is_repairable(
            "best entailment 0.42 from E1 is below the 0.98 support threshold",
            every_proposition_supported=True,
        )
        assert not ok

    def test_a_mixed_failure_is_refused(self) -> None:
        """One unrepairable guard blocks the whole attempt."""
        ok, _ = is_repairable(GUARD.format("atomicity, causal"), every_proposition_supported=True)
        assert not ok

    def test_guard_names_are_read_from_the_reason(self) -> None:
        assert failed_guard_names(GUARD.format("atomicity, numeric")) == {
            "atomicity",
            "numeric",
        }


class TestWhatARewriteMayNotDo:
    ORIGINAL = "Developers took 19% longer, per the study authors."

    def test_it_may_not_add_a_number(self) -> None:
        ok, why = validate_rewrite("Developers took longer.", "Developers took 19% longer.")
        assert not ok
        assert "numbers" in why

    def test_it_may_not_change_a_number(self) -> None:
        ok, _ = validate_rewrite(self.ORIGINAL, "Developers took 20% longer.")
        assert not ok

    def test_it_may_not_add_a_subject(self) -> None:
        ok, why = validate_rewrite(self.ORIGINAL, "Developers using Copilot took 19% longer.")
        assert not ok
        assert "subjects" in why

    def test_it_may_not_strengthen_the_claim(self) -> None:
        ok, why = validate_rewrite(
            "AI tools may slow developers.", "AI tools always slow developers."
        )
        assert not ok
        assert "more strongly" in why

    def test_it_may_not_invent_causation(self) -> None:
        """The single most dangerous rewrite available."""
        ok, why = validate_rewrite(
            "Teams reviewing more code reported fewer defects.",
            "Code review causes fewer defects.",
        )
        assert not ok
        assert "cause" in why

    def test_it_may_not_return_the_original(self) -> None:
        ok, _ = validate_rewrite(self.ORIGINAL, self.ORIGINAL)
        assert not ok

    def test_it_may_not_return_nothing(self) -> None:
        ok, _ = validate_rewrite(self.ORIGINAL, "   ")
        assert not ok

    def test_it_may_drop_a_number(self) -> None:
        """Dropping is safe; adding is not."""
        ok, _ = validate_rewrite("Developers took 19% longer.", "Developers took longer.")
        assert ok

    def test_it_may_soften(self) -> None:
        ok, _ = validate_rewrite("AI tools slow developers.", "AI tools may slow developers.")
        assert ok

    def test_a_plain_rewording_is_allowed(self) -> None:
        ok, _ = validate_rewrite(
            "In a trial devs took 19% longer per the authors.",
            "Developers took 19% longer in a trial.",
        )
        assert ok


class TestCoverage:
    COMPARISON = build_contract(
        "How does REST differ from GraphQL?", "comparison", entities=["REST", "GraphQL"]
    )

    def test_two_definitions_answer_nothing(self) -> None:
        """The live failure, stated as coverage."""
        cov = assess_coverage(self.COMPARISON, [])
        assert cov.answered is False
        assert "did not answer the question" in cov.limitations()[0]

    def test_a_contrast_answers_it(self) -> None:
        cov = assess_coverage(self.COMPARISON, ["direct_contrast"])
        assert cov.answered is True
        assert not any("did not answer" in limit for limit in cov.limitations())

    def test_a_slot_the_question_never_asked_for_counts_for_nothing(self) -> None:
        cov = assess_coverage(self.COMPARISON, ["tradeoffs"])
        assert cov.satisfied == ()
        assert cov.answered is False

    def test_missing_slots_become_specific_limitations(self) -> None:
        cov = assess_coverage(self.COMPARISON, ["direct_contrast"])
        assert any("relate" in limit for limit in cov.limitations())

    def test_duplicates_are_reported(self) -> None:
        cov = assess_coverage(self.COMPARISON, ["direct_contrast", "direct_contrast"])
        assert cov.duplicates == ("direct_contrast",)
        assert any("repeat" in limit for limit in cov.limitations())

    def test_a_multipart_question_names_the_part_it_missed(self) -> None:
        contract = build_contract(
            "What is RAG and what are its failure modes?",
            "synthesis",
            parts=["what RAG is", "its failure modes"],
        )
        cov = assess_coverage(contract, ["part_1"])
        assert cov.answered is False
        assert any("failure modes" in limit for limit in cov.limitations())

    def test_an_unusable_contract_says_so_first(self) -> None:
        broken = build_contract("what is an llm?", "comparison", entities=["llm"])
        cov = assess_coverage(broken, [])
        limits = cov.limitations()
        assert len(limits) == 1
        assert "could not be turned into" in limits[0]


class TestRepairIsWiredCorrectly:
    """The path exists and cannot become a way through.

    The behaviour is asserted at the unit level above. These pin the
    wiring, because a repair loop that skipped one of these steps
    would still look like it worked.
    """

    def _source(self) -> str:
        import inspect

        from agentic_research.graph.nodes import reporting

        return inspect.getsource(reporting._repair_wording)

    def test_the_rewrite_is_validated_before_it_is_re_verified(self) -> None:
        """A rewrite that smuggled something in must never reach the
        gates that would have to notice."""
        source = self._source()
        assert source.index("validate_rewrite") < source.index("verify_claim")

    def test_every_gate_runs_again(self) -> None:
        source = self._source()
        assert "verify_claim" in source
        assert "deterministic_relevance" in source

    def test_there_is_no_second_attempt(self) -> None:
        source = self._source()
        assert "while" not in source
        assert "retry" not in source.lower()

    def test_a_failed_rewrite_is_discarded_not_partially_kept(self) -> None:
        source = self._source()
        assert source.count("continue") >= 3

    def test_eligibility_requires_every_proposition_supported(self) -> None:
        import inspect

        from agentic_research.graph.nodes import reporting

        caller = inspect.getsource(reporting._check_entailment)
        assert "every_proposition_supported=_propositions_supported(" in caller

    def test_an_empty_rewrite_is_a_valid_answer(self) -> None:
        """ "It cannot be fixed by rewording" is correct and common."""
        assert "if not text:" in self._source()

    def test_the_repairer_is_not_the_synthesiser(self) -> None:
        assert "ModelRole.CRITIC" in self._source()


class TestAComparisonHasTwoHonestAnswers:
    """The finding hosted acceptance produced.

    v1.2.0 was asked "How does a large language model differ from a
    neural network?". It found and published that one is a subset of
    the other -- which is the answer -- and then reported that it had
    not answered, because a subset is not a contrast.

    When one subject is a category containing the other there is no
    contrast to find. Demanding one makes the engine wrong about
    itself, which is worse than being strict.
    """

    CONTRACT = build_contract(
        "How does a large language model differ from a neural network?",
        QuestionType.COMPARISON,
        entities=("large language model (LLM)", "neural network"),
    )

    def test_a_relationship_answers_a_comparison(self) -> None:
        """The exact slots the acceptance run published."""
        coverage = assess_coverage(self.CONTRACT, ["relationship", "dimension", "dimension"])
        assert coverage.answered
        assert coverage.missing_core == ()

    def test_a_direct_contrast_still_answers_it(self) -> None:
        """Non-vacuity in the other direction: the original slot was
        not replaced, only given an alternative."""
        coverage = assess_coverage(self.CONTRACT, ["direct_contrast"])
        assert coverage.answered

    def test_dimensions_alone_do_not(self) -> None:
        """The failure the contract exists for. Naming axes along which
        two things differ, without saying how they differ or how they
        relate, is not an answer."""
        coverage = assess_coverage(self.CONTRACT, ["dimension", "dimension"])
        assert not coverage.answered
        assert coverage.missing_core == ("direct_contrast",)

    def test_publishing_nothing_relevant_does_not(self) -> None:
        coverage = assess_coverage(self.CONTRACT, [])
        assert not coverage.answered

    def test_the_alternative_is_declared_not_inferred(self) -> None:
        """No slot gets this by accident: it is one entry on one slot."""
        from agentic_research.answer_contract import CANONICAL_SLOTS

        with_alternatives = {
            (str(qt), s.name)
            for qt, slots in CANONICAL_SLOTS.items()
            for s in slots
            if s.satisfied_by
        }
        assert with_alternatives == {("comparison", "direct_contrast")}

    def test_it_does_not_leak_into_other_question_types(self) -> None:
        """A definition answered by a relationship claim is still
        unanswered -- the slot does not even exist there."""
        definition = build_contract(
            "What is a large language model?",
            QuestionType.DEFINITION,
            entities=("large language model",),
        )
        coverage = assess_coverage(definition, ["relationship"])
        assert not coverage.answered


class TestEveryCorePartMustBeAnswered:
    """A multi-part question makes each part its own core slot, and the
    strict reading is the one that matters there.

    The `answered` docstring claimed "at least one core slot filled"
    while the code required all of them. Every canonical type has
    exactly one core slot, so the two readings only diverge for
    multi-part questions -- where answering one part of three is not
    answering the question.
    """

    def test_one_part_of_three_is_not_an_answer(self) -> None:
        contract = build_contract(
            "What is X, how is it deployed, and what does it cost?",
            QuestionType.SYNTHESIS,
            parts=("what_it_is", "how_deployed", "what_it_costs"),
        )
        core = [s.name for s in contract.core_slots]
        assert len(core) == 3, "each part should be its own core slot"

        assert not assess_coverage(contract, core[:1]).answered
        assert not assess_coverage(contract, core[:2]).answered
        assert assess_coverage(contract, core).answered

    def test_the_missing_parts_are_named(self) -> None:
        """Named from the contract rather than guessed: the slots are
        positional, so a test that invented their names would be
        asserting its own assumption."""
        contract = build_contract(
            "What is X, how is it deployed, and what does it cost?",
            QuestionType.SYNTHESIS,
            parts=("what_it_is", "how_deployed", "what_it_costs"),
        )
        core = [s.name for s in contract.core_slots]
        coverage = assess_coverage(contract, core[:1])
        assert set(coverage.missing_core) == set(core[1:])
        assert coverage.missing_core, "a missing part must be reported, not inferred"
