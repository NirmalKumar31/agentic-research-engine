"""Six defects behind "the output looks thin", each from a live run.

Two hosted runs on v1.9.0 produced good infrastructure and poor answers:

*"What types of vector index are used for similarity search?"* — six
excellent sources, 15 evidence items, and **one** published claim, that
claim being "Tree-based indexes are one type of vector index."

*"langchain vs langgraph differences?"* — 30 evidence items, five
published claims, **four of them about LangGraph alone**, three named
axes satisfied, and still "this report does not answer the question".
Correctly: no axis carried a claim about both subjects. Each claim was
true, supported and relevant. Together they were not a comparison.

None of it was a verification failure. The gates behaved exactly as
designed, on an input that was starved before it reached them.
"""

from __future__ import annotations

from agentic_research.answer_contract import QuestionType, build_contract
from agentic_research.comparison import SideClaim, build_comparison_pairs
from agentic_research.evidence.quality import base_quality_for, classify_source
from agentic_research.evidence.topicality import alias_groups
from agentic_research.graph.prompts import PLANNER_SYSTEM, synthesizer_user
from agentic_research.models import SourceType


class TestTheSynthesiserIsAskedForBalance:
    """Four claims about LangGraph and none about LangChain."""

    def test_a_comparison_prompt_demands_one_claim_per_subject(self) -> None:
        rendered = synthesizer_user(
            "How does LangChain differ from LangGraph?",
            "comparison",
            "evidence",
            "",
            comparison_subjects=["LangChain", "LangGraph"],
        )
        assert "one claim per subject" in rendered
        assert "LangChain, LangGraph" in rendered

    def test_it_says_what_an_unbalanced_report_is_not(self) -> None:
        """A rule without its consequence gets edited away."""
        rendered = synthesizer_user("q", "comparison", "e", "", comparison_subjects=["A", "B"])
        assert "is not a comparison" in rendered

    def test_it_tells_the_model_what_to_do_when_evidence_is_one_sided(self) -> None:
        """Otherwise the instruction is unfollowable and gets ignored."""
        rendered = synthesizer_user("q", "comparison", "e", "", comparison_subjects=["A", "B"])
        assert "different axis where it covers both" in rendered

    def test_a_non_comparison_gets_no_such_instruction(self) -> None:
        rendered = synthesizer_user("q", "list", "e", "")
        assert "one claim per subject" not in rendered

    def test_one_subject_is_not_a_comparison(self) -> None:
        rendered = synthesizer_user("q", "comparison", "e", "", comparison_subjects=["A"])
        assert "one claim per subject" not in rendered

    def test_the_node_passes_the_subjects(self) -> None:
        import inspect

        from agentic_research.graph.nodes import reporting

        body = inspect.getsource(reporting.synthesize_report)
        assert "comparison_subjects=" in body


class TestTheClaimBudgetTracksTheQuestionNotTheContract:
    """`list` has two slots whatever it asks for.

    So a six-dimension list question asked for four claims and published
    one, while a comparison that happened to gain three named axes asked
    for twelve and published seven. The budget was tracking the shape of
    the contract rather than the size of the question.
    """

    def _budget(self, contract, sub_questions: int) -> int:
        from agentic_research.graph.nodes.reporting import (
            _CLAIMS_PER_SLOT,
        )

        # The pure arithmetic the node performs; the node itself needs a
        # live run context for the call-budget check above it.
        return max(
            _CLAIMS_PER_SLOT * len(contract.required_slots),
            _CLAIMS_PER_SLOT * sub_questions,
        )

    def test_a_six_part_list_question_is_no_longer_capped_at_four(self) -> None:
        contract = build_contract(
            "What types of vector index are used for similarity search?",
            QuestionType.LIST,
            entities=("vector index",),
        )
        assert len(contract.required_slots) == 2
        assert self._budget(contract, 0) == 4, "slot-only budget, the old behaviour"
        assert self._budget(contract, 6) == 12, "six planned dimensions"

    def test_it_never_asks_for_less_than_the_contract_needs(self) -> None:
        """Non-vacuity in the other direction: a narrow plan must not
        shrink a contract with many slots."""
        contract = build_contract(
            "How does A differ from B?",
            QuestionType.COMPARISON,
            entities=("A", "B"),
            comparison_subjects=("A", "B"),
            dimensions=("cost", "latency", "accuracy"),
        )
        assert self._budget(contract, 1) == _expected_from_slots(contract)

    def test_the_node_reads_the_planned_sub_questions(self) -> None:
        import inspect

        from agentic_research.graph.nodes import reporting

        body = inspect.getsource(reporting.synthesize_report)
        assert 'len(state.get("sub_questions", []))' in body


def _expected_from_slots(contract) -> int:
    from agentic_research.graph.nodes.reporting import _CLAIMS_PER_SLOT

    return _CLAIMS_PER_SLOT * len(contract.required_slots)


class TestAnAliasCountsAsSpeakingAboutASubject:
    """ "unlike LCEL's linear flow" is a contrast with LangChain."""

    CLAIM = (
        "LangGraph's state persists and accumulates throughout execution, "
        "unlike LCEL's linear flow of data from output to input."
    )

    def _contract(self):
        return build_contract(
            "langchain vs langgraph differences?",
            QuestionType.COMPARISON,
            entities=["LangChain", "LangGraph", "LangChain Expression Language (LCEL)"],
            comparison_subjects=["LangChain", "LangGraph"],
            dimensions=["state management"],
        )

    def _aliases(self):
        return alias_groups(
            "langchain vs langgraph differences?",
            ["LangChain", "LangGraph", "LangChain Expression Language (LCEL)"],
        )

    def test_without_aliases_the_claim_speaks_about_one_subject(self) -> None:
        """Non-vacuity: this is the behaviour that lost the pair."""
        from agentic_research.comparison import _speaks_about

        assert not _speaks_about(self.CLAIM, "LangChain", ())

    def test_with_the_run_s_own_aliases_it_speaks_about_both(self) -> None:
        from agentic_research.comparison import _speaks_about

        aliases = self._aliases()
        assert _speaks_about(self.CLAIM, "LangGraph", aliases)
        assert _speaks_about(self.CLAIM, "LangChain", aliases)

    def test_the_pair_now_forms(self) -> None:
        pairs = build_comparison_pairs(
            self._contract(),
            [SideClaim(subject="", text=self.CLAIM, answer_slot="state management")],
            aliases=self._aliases(),
        )
        assert len(pairs) == 1

    def test_an_unrelated_acronym_does_not_bridge_subjects(self) -> None:
        """Bounded: only aliases this run's own analysis names."""
        from agentic_research.comparison import _speaks_about

        aliases = alias_groups("", ["LangChain", "LangGraph"])
        assert not _speaks_about("GPU memory is limited.", "LangChain", aliases)


class TestThePlannerCannotOutrunTheSourceBudget:
    """Coverage needs two distinct sources per sub-question.

    Six sub-questions therefore need twelve source-slots against a budget
    of six, which is why a run reported "only limited evidence was found"
    four times. Arithmetic, not retrieval.
    """

    def test_the_prompt_asks_for_fewer_dimensions(self) -> None:
        assert "two to four sub-questions" in PLANNER_SYSTEM

    def test_the_prompt_states_the_arithmetic(self) -> None:
        assert "two independent sources" in PLANNER_SYSTEM
        assert "starved" in PLANNER_SYSTEM

    def test_the_node_caps_deterministically(self) -> None:
        """A prompt is a request; the cap is the guarantee."""
        import inspect

        from agentic_research.graph.nodes import planning

        body = inspect.getsource(planning.plan_research)
        assert "ctx().budget.max_sources // 2" in body

    def test_the_cap_matches_what_coverage_requires(self) -> None:
        for sources, expected in [(6, 3), (4, 2), (10, 5), (2, 2), (1, 2)]:
            assert max(2, sources // 2) == expected, sources


class TestSourceClassificationGapsFoundLive:
    def test_medium_publications_on_their_own_domains_are_blogs(self) -> None:
        """These scored `other` (0.50) — above a blog (0.45) — so three
        were selected while the blog penalty never applied."""
        for domain in ("pub.towardsai.net", "ai.plainenglish.io"):
            assert classify_source(f"https://{domain}/p", domain) is SourceType.BLOG

    def test_project_documentation_on_a_pages_domain_is_official(self) -> None:
        for domain in ("langchain-ai.github.io", "numpy.readthedocs.io"):
            assert classify_source(f"https://{domain}/x", domain) is SourceType.OFFICIAL_DOCS

    def test_tutorial_sites_are_not_promoted_to_reference(self) -> None:
        """REFERENCE (0.70) is for reviewed encyclopaedic sources.
        Rating variable-quality tutorial content above a vendor's own
        page would be guessing at quality rather than classifying
        provenance."""
        for domain in ("geeksforgeeks.org", "datacamp.com"):
            assert classify_source(f"https://{domain}/p", domain) is not SourceType.REFERENCE

    def test_the_ordering_that_matters_holds(self) -> None:
        order = [
            SourceType.OFFICIAL_DOCS,
            SourceType.REFERENCE,
            SourceType.OTHER,
            SourceType.BLOG,
            SourceType.SOCIAL,
        ]
        scores = [base_quality_for(t) for t in order]
        assert scores == sorted(scores, reverse=True), scores
