"""The quality stages, wired into the graph rather than beside it.

Each piece was built and measured against fixed evidence first. These
assert it is actually reachable from a run: the contract is built
before retrieval, claims carry the slot they claim to fill, an
irrelevant claim is withheld with its own reason, and the report says
which required part is missing.
"""

from __future__ import annotations

from agentic_research.answer_contract import QuestionType
from agentic_research.graph.nodes.planning import contract_from_analysis
from agentic_research.graph.prompts import synthesizer_user
from agentic_research.models import OutputFormat, QueryAnalysis


def analysis(**kw) -> QueryAnalysis:
    base = {
        "original_query": "q",
        "normalized_query": "q",
        "intent": "i",
        "entities": [],
        "constraints": [],
        "output_format": OutputFormat.OVERVIEW,
    }
    base.update(kw)
    return QueryAnalysis(**base)  # type: ignore[arg-type]


class TestTheContractIsBuiltFromTheAnalysis:
    def test_a_comparison_analysis_yields_a_comparison_contract(self) -> None:
        c = contract_from_analysis(
            analysis(
                output_format=OutputFormat.COMPARISON,
                entities=["SMOTE", "cost-sensitive learning"],
            )
        )
        assert c.usable
        assert c.question_type is QuestionType.COMPARISON
        assert [s.name for s in c.core_slots] == ["direct_contrast"]

    def test_a_one_sided_comparison_is_unusable(self) -> None:
        """Refused rather than guessed. The run continues and the
        report says the question could not be pinned down."""
        c = contract_from_analysis(
            analysis(output_format=OutputFormat.COMPARISON, entities=["SMOTE"])
        )
        assert c.usable is False

    def test_the_analysis_label_is_not_reclassified(self) -> None:
        """Two classifications that can disagree are worse than one."""
        for fmt, expected in (
            (OutputFormat.HOWTO, QuestionType.PROCEDURAL),
            (OutputFormat.TIMELINE, QuestionType.TEMPORAL),
            (OutputFormat.OVERVIEW, QuestionType.DEFINITION),
        ):
            c = contract_from_analysis(analysis(output_format=fmt, entities=["x"]))
            assert c.question_type is expected, fmt

    def test_constraints_reach_the_contract(self) -> None:
        c = contract_from_analysis(analysis(entities=["x"], constraints=["figures for 2025 only"]))
        assert "figures for 2025 only" in c.constraints


class TestTheSynthesiserIsToldWhatToFill:
    def test_the_slots_appear_in_the_prompt(self) -> None:
        prompt = synthesizer_user(
            "How does A differ from B?",
            "comparison",
            "- S1-e1: something",
            "",
            answer_slots=[("direct_contrast", "An explicit statement of how they differ")],
        )
        assert "direct_contrast" in prompt
        assert "An explicit statement of how they differ" in prompt

    def test_it_says_an_unslotted_claim_does_not_belong(self) -> None:
        """Without this the model writes whatever the evidence
        supports, which is how a comparison was answered with five
        definitions."""
        prompt = synthesizer_user(
            "q", "comparison", "e", "", answer_slots=[("direct_contrast", "d")]
        )
        assert "does not belong in the report" in prompt

    def test_no_slot_block_when_there_is_no_contract(self) -> None:
        prompt = synthesizer_user("q", "overview", "e", "", answer_slots=None)
        assert "answer_slot" not in prompt


class TestClaimsCarryTheirSlot:
    def test_the_schema_offers_it(self) -> None:
        from agentic_research.schemas import ClaimOut

        assert "answer_slot" in ClaimOut.model_fields

    def test_the_model_keeps_it(self) -> None:
        from agentic_research.models import Claim

        assert Claim(text="t", answer_slot="direct_contrast").answer_slot == "direct_contrast"

    def test_it_survives_conversion_from_the_model_output(self) -> None:
        from agentic_research.graph.nodes.reporting import _to_claim
        from agentic_research.schemas import ClaimOut

        claim = _to_claim(
            ClaimOut(
                text="t", evidence_ids=["S1-e1"], kind="factual", answer_slot="direct_contrast"
            )
        )
        assert claim.answer_slot == "direct_contrast"

    def test_a_missing_slot_becomes_empty_rather_than_invented(self) -> None:
        from agentic_research.graph.nodes.reporting import _to_claim
        from agentic_research.schemas import ClaimOut

        assert _to_claim(ClaimOut(text="t", evidence_ids=[], kind="factual")).answer_slot == ""


class TestIrrelevanceIsItsOwnVerdict:
    def test_the_verdict_type_admits_it(self) -> None:
        """Reporting a supported-but-off-topic claim as unsupported
        would misstate why it was withheld."""
        from agentic_research.citations.publication import ClaimVerdict

        assert "irrelevant" in ClaimVerdict.__args__  # type: ignore[attr-defined]

    def test_the_judgment_record_admits_it(self) -> None:
        from agentic_research.models import ClaimJudgment, ClaimKind

        judgment = ClaimJudgment(claim_text="c", kind=ClaimKind.FACTUAL, verdict="irrelevant")
        assert judgment.verdict == "irrelevant"


class TestCoverageReachesTheReader:
    def test_a_comparison_with_no_contrast_says_it_did_not_answer(self) -> None:
        from agentic_research.answer_coverage import assess_coverage

        contract = contract_from_analysis(
            analysis(output_format=OutputFormat.COMPARISON, entities=["A", "B"])
        )
        coverage = assess_coverage(contract, ["dimension"])
        assert coverage.answered is False
        assert any("did not answer the question" in limit for limit in coverage.limitations())

    def test_an_unusable_contract_warns_before_anything_else(self) -> None:
        from agentic_research.answer_coverage import assess_coverage

        contract = contract_from_analysis(
            analysis(output_format=OutputFormat.COMPARISON, entities=["only-one"])
        )
        limits = assess_coverage(contract, []).limitations()
        assert "could not be turned into" in limits[0]


class TestTheJudgementIsAskedAndFailsClosed:
    """The gate the deterministic checks cannot be.

    Structure catches a claim about the wrong subject, a definition
    filling a contrast slot, a figure from the wrong year. It cannot
    catch a claim that is on topic, correctly shaped, and still not an
    answer. That judgement is asked of a model, and until it was, the
    call site passed True unconditionally -- a parameter that looked
    like a gate and was a decoration.
    """

    def test_no_judgement_withholds(self) -> None:
        from agentic_research.answer_contract import build_contract
        from agentic_research.citations.relevance import assess_relevance

        contract = build_contract(
            "What is a vector database?", "definition", entities=["vector database"]
        )
        claim = "A vector database stores embeddings and retrieves them by search."
        assert assess_relevance(claim, "definition", contract).publishable is False

    def test_a_rejection_withholds(self) -> None:
        from agentic_research.answer_contract import build_contract
        from agentic_research.citations.relevance import assess_relevance

        contract = build_contract(
            "What is a vector database?", "definition", entities=["vector database"]
        )
        claim = "A vector database stores embeddings and retrieves them by search."
        assert (
            assess_relevance(claim, "definition", contract, model_says_relevant=False).publishable
            is False
        )

    def test_structure_is_checked_before_the_judge_is_asked(self) -> None:
        """Structure is free and the judgement costs a provider
        request. A claim that fails structure must not buy one."""
        from agentic_research.answer_contract import build_contract
        from agentic_research.citations.relevance import deterministic_relevance

        contract = build_contract("How fast is PostgreSQL?", "numeric", entities=["PostgreSQL"])
        v = deterministic_relevance(
            "MySQL sustained 42,000 inserts per second.", "measured_value", contract
        )
        assert v.publishable is False

    def test_the_judge_is_not_the_model_that_wrote_the_claim(self) -> None:
        """A model marking its own homework finds it relevant."""
        import inspect

        from agentic_research.graph.nodes import reporting

        source = inspect.getsource(reporting._judge_relevance)
        assert "ModelRole.CRITIC" in source
        assert "SYNTHESIZER" not in source

    def test_it_is_one_call_for_the_whole_report(self) -> None:
        """A public run has twenty provider requests in total; a gate
        costing one per claim would be the most expensive thing in it."""
        import inspect

        from agentic_research.graph.nodes import reporting

        source = inspect.getsource(reporting._judge_relevance)
        assert "[claim.text for claim in claims]" in source

    def test_a_failed_call_returns_nothing_rather_than_guessing(self) -> None:
        import inspect

        from agentic_research.graph.nodes import reporting

        source = inspect.getsource(reporting._judge_relevance)
        assert "return {}" in source
