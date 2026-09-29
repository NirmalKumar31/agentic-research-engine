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
        c = build_contract(
            "a vs b", "comparison", entities=["a", "b"], dimensions=["cost", "latency"]
        )
        assert c.has_slot("cost")
        assert c.has_slot("latency")
        assert not c.has_slot("dimension"), "the placeholder should give way to real ones"

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
    which discharges the comparison's `direct_contrast` requirement.
    The engine's own limitations therefore did not say the question was
    unanswered. The payload omitted `satisfied_by`, so the interface
    recomputed coverage, found the core slot unfilled, and would have
    rendered "this report does not answer the question" directly above
    a report saying the opposite.
    """

    def test_satisfied_by_is_serialised(self) -> None:
        contract = build_contract(
            "How does a large language model differ from a neural network?",
            QuestionType.COMPARISON,
            entities=("large language model", "neural network"),
        )
        slots = {s["name"]: s for s in contract.to_dict()["required_slots"]}  # type: ignore[union-attr,index]
        assert slots["direct_contrast"]["satisfied_by"] == ["relationship"]

    def test_a_slot_without_alternatives_serialises_an_empty_list(self) -> None:
        """Present and empty, not absent. A client checking the field
        should not have to distinguish "no alternatives" from "this
        build does not report them"."""
        contract = build_contract(
            "What is a vector database?", QuestionType.DEFINITION, entities=("vector database",)
        )
        for slot in contract.to_dict()["required_slots"]:  # type: ignore[union-attr]
            assert slot["satisfied_by"] == []

    def test_the_payload_round_trips_the_alternative(self) -> None:
        """The property that matters: a consumer reading only the dict
        can reach the same verdict the engine did."""
        contract = build_contract(
            "How does X differ from Y?",
            QuestionType.COMPARISON,
            entities=("X", "Y"),
        )
        payload = contract.to_dict()
        published = {"relationship"}
        core = [s for s in payload["required_slots"] if s["core"]]  # type: ignore[union-attr]
        discharged = [
            s for s in core if s["name"] in published or set(s["satisfied_by"]) & published
        ]
        assert len(discharged) == len(core), "a client could not see the alternative"
