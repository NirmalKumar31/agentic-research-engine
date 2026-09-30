"""What the question requires, fixed before anything is retrieved.

A visitor asked how a large language model differs from a neural
network and received five supported claims describing what a large
language model is. Every gate passed. Nothing in the pipeline had been
told what the question required, so nothing could notice.

These tests pin the shape of that requirement. The shape is canonical
per question type rather than generated, because a model asked to
decide both the shape of an answer and its content will produce a
shape its content already fits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_research.answer_contract import (
    CANONICAL_SLOTS,
    QuestionType,
    build_contract,
    type_from_output_format,
)

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "examples" / "quality-eval" / "cases.json"


class TestTheShapeOfEachAnswer:
    def test_a_comparison_requires_a_contrast(self) -> None:
        """The slot the live failure was missing. Definitions of each
        side, however well supported, do not fill it."""
        c = build_contract("a vs b", "comparison", entities=["a", "b"])
        assert [s.name for s in c.core_slots] == ["direct_contrast"]

    def test_a_definition_requires_a_definition(self) -> None:
        """The relevance gate must reject definitions answering
        comparison questions, not definitions as such."""
        c = build_contract("what is x?", "definition", entities=["x"])
        assert [s.name for s in c.core_slots] == ["definition"]

    def test_a_procedure_requires_steps(self) -> None:
        c = build_contract("how do I x?", "procedural", entities=["x"])
        assert [s.name for s in c.core_slots] == ["steps"]

    def test_a_causal_question_requires_causal_evidence(self) -> None:
        c = build_contract("why did x?", "causal", entities=["x"])
        assert [s.name for s in c.core_slots] == ["causal_evidence"]

    def test_every_question_type_has_exactly_one_core_slot_or_parts(self) -> None:
        """More than one core slot makes "did it answer?" ambiguous."""
        for qtype, slots in CANONICAL_SLOTS.items():
            if qtype is QuestionType.SYNTHESIS:
                continue
            assert sum(1 for s in slots if s.core) == 1, qtype

    def test_named_dimensions_become_their_own_slots(self) -> None:
        """Named axes are added; the generic placeholder stays.

        Dropping the placeholder once named axes existed meant a claim
        contributing to the comparison but not sitting on a named axis
        had no slot to declare, so structural relevance refused it as
        unknown -- naming axes would have reduced what a comparison
        could publish. The placeholder cannot complete a contrast; only
        a named axis can.
        """
        contract = build_contract(
            "How does A differ from B on latency and cost?",
            QuestionType.COMPARISON,
            entities=("A", "B"),
            dimensions=("latency", "cost"),
        )
        names = {s.name for s in contract.required_slots}
        assert {"latency", "cost"} <= names
        assert "dimension" in names

    def test_the_placeholder_survives_when_no_dimension_is_named(self) -> None:
        c = build_contract("a vs b", "comparison", entities=["a", "b"])
        assert c.has_slot("dimension")


class TestItRefusesRatherThanGuesses:
    def test_a_one_sided_comparison_is_unusable(self) -> None:
        """With one subject there is nothing to contrast, and the
        contract would be satisfiable by a definition."""
        c = build_contract("what is an llm?", "comparison", entities=["llm"])
        assert c.usable is False
        assert "two subjects" in c.unusable_reason

    def test_a_multipart_question_without_named_parts_is_unusable(self) -> None:
        c = build_contract("x and y?", "synthesis")
        assert c.usable is False
        assert "parts named" in c.unusable_reason

    def test_an_empty_question_is_unusable(self) -> None:
        assert build_contract("   ", "definition").usable is False

    def test_an_unrecognised_type_is_unusable(self) -> None:
        c = build_contract("q", "vibes")
        assert c.usable is False
        assert "unrecognised" in c.unusable_reason

    def test_an_unusable_contract_carries_no_slots(self) -> None:
        """So nothing downstream can accidentally satisfy one."""
        assert build_contract("q", "vibes").required_slots == ()


class TestMultiPartQuestions:
    def test_each_part_becomes_a_core_slot(self) -> None:
        c = build_contract(
            "what is RAG and what are its failure modes?",
            "synthesis",
            parts=["what is RAG", "failure modes"],
        )
        assert len(c.core_slots) == 2
        assert all("An answer to:" in s.description for s in c.core_slots)

    def test_answering_one_part_cannot_look_complete(self) -> None:
        c = build_contract("a and b?", "synthesis", parts=["a", "b"])
        satisfied = {"part_1"}
        missing = [s.name for s in c.core_slots if s.name not in satisfied]
        assert missing == ["part_2"]


class TestItMatchesTheFrozenFixtures:
    """The contract has to describe the cases the release is measured
    against, or the two are talking about different questions."""

    @pytest.fixture(scope="class")
    @classmethod
    def cases(cls) -> list[dict]:
        return json.loads(CASES.read_text())["cases"]

    def test_every_fixture_type_is_a_known_question_type(self, cases: list[dict]) -> None:
        for case in cases:
            declared = case["contract"]["question_type"]
            if declared == "insufficient_evidence":
                continue  # a property of the evidence, not of the question
            if declared == "terminology_mismatch":
                continue  # a property of the sources, not of the question
            QuestionType(declared)

    def test_the_comparison_fixture_slots_are_all_derivable(self, cases: list[dict]) -> None:
        case = next(c for c in cases if c["id"] == "comparison-llm-vs-nn")
        contract = build_contract(
            case["question"],
            "comparison",
            entities=case["contract"]["entities"],
            dimensions=["architecture_or_scope", "training_objective", "capabilities"],
        )
        assert set(case["contract"]["required_slots"]) <= contract.slot_names


class TestTheExistingAnalysisLabelMaps:
    """The analysis stage already emits an output_format. Two
    classifications that can disagree are worse than one."""

    def test_known_formats_map(self) -> None:
        assert type_from_output_format("comparison") is QuestionType.COMPARISON
        assert type_from_output_format("howto") is QuestionType.PROCEDURAL
        assert type_from_output_format("timeline") is QuestionType.TEMPORAL

    def test_an_unknown_format_maps_to_nothing(self) -> None:
        assert type_from_output_format("interpretive_dance") is None


class TestTheSerialisedContractCarriesItsAlternatives:
    """A client that cannot see `satisfied_by` computes the wrong answer.

    The hosted v1.2.1 run published a claim filling `relationship`,
    which then discharged the comparison's `direct_contrast`
    requirement. The payload omitted `satisfied_by`, so the interface
    recomputed coverage, found the core slot unfilled, and would have
    rendered "this report does not answer the question" directly above
    a report saying the opposite.

    No canonical slot declares an alternative any more -- both cases
    that did were semantic shortcuts, and whether a relationship
    explains away a contrast now depends on its kind, which a slot name
    cannot express. The serialisation is still exercised here, because
    the field remains part of the payload contract and a client must be
    able to read it; it is simply no longer populated from the
    canonical table.
    """

    def test_satisfied_by_is_serialised_when_a_slot_declares_one(self) -> None:
        from agentic_research.answer_contract import AnswerContract, AnswerSlot

        contract = AnswerContract(
            question="q",
            question_type=QuestionType.COMPARISON,
            required_slots=(
                AnswerSlot(
                    name="direct_contrast",
                    description="d",
                    satisfied_by=("relationship",),
                ),
            ),
        )
        slots = {s["name"]: s for s in contract.to_dict()["required_slots"]}  # type: ignore[union-attr,index]
        assert slots["direct_contrast"]["satisfied_by"] == ["relationship"]

    def test_no_canonical_slot_declares_an_alternative(self) -> None:
        """The audit's finding, pinned: a static alternative asserts
        that one slot's name always implies another's satisfaction, and
        neither case that used it was ever that."""
        from agentic_research.answer_contract import CANONICAL_SLOTS

        declared = {
            (str(qt), slot.name)
            for qt, slots in CANONICAL_SLOTS.items()
            for slot in slots
            if slot.satisfied_by
        }
        assert declared == set()

    def test_a_slot_without_alternatives_serialises_an_empty_list(self) -> None:
        """Present and empty, not absent. A client checking the field
        should not have to distinguish "no alternatives" from "this
        build does not report them"."""
        contract = build_contract(
            "What is a vector database?", QuestionType.DEFINITION, entities=("vector database",)
        )
        for slot in contract.to_dict()["required_slots"]:  # type: ignore[union-attr]
            assert slot["satisfied_by"] == []

    def test_the_payload_round_trips_the_discharge(self) -> None:
        """The property that matters, unchanged: a consumer reading only
        the payload reaches the same verdict the engine did.

        What carries it has changed. The contract can no longer say
        that `relationship` discharges `direct_contrast`, because that
        depends on the relationship's kind -- so the fact now travels
        in the coverage assessment instead. If it did not, the client
        would recompute "unanswered" and render that above a report
        saying the opposite, which is the v1.2.1 contradiction this
        class exists to prevent.
        """
        from agentic_research.answer_coverage import assess_coverage
        from agentic_research.comparison import SideClaim

        contract = build_contract(
            "How does a large language model differ from a neural network?",
            QuestionType.COMPARISON,
            entities=("large language model", "neural network"),
            comparison_subjects=("large language model", "neural network"),
        )
        coverage = assess_coverage(
            contract,
            ["relationship"],
            claims=[
                SideClaim(
                    subject="",
                    text=("A large language model is a kind of neural network trained on text."),
                    answer_slot="relationship",
                )
            ],
        )
        payload = coverage.to_dict()
        assert payload["answered"] is True
        assert payload["relationship_discharge"] == "subtype"
        assert "direct_contrast" in payload["satisfied_slots"]
