"""Queries in the question's vocabulary, not a stack of specialist terms.

The measured failure. Asked "What are the main causes of overfitting in
machine learning?", the engine sent six queries averaging eleven words
of research jargon, among them

    parametric knowledge long tail facts factual recall generalization
    failures language models hallucinations

Only papers contain those term combinations, so the six pages read were
a tweet, a LinkedIn-style blog, a Substack newsletter and two papers on
double descent. One of five sub-questions reached its evidence
threshold and nothing published.

The cause was an instruction in the prompt -- "Include the specific
technical terms an authoritative page would use" -- with no length
bound. The model obeyed it exactly. This is the third time in this
project that a prompt taught the defect it then got blamed for, after
the synthesiser examples that began "The source reports".
"""

from __future__ import annotations

import pytest

from agentic_research.graph.prompts import QUERY_WRITER_SYSTEM, query_writer_user


class TestThePromptNoLongerAsksForJargon:
    def test_the_instruction_that_caused_it_is_gone(self) -> None:
        """Non-vacuity: this assertion fails against the old prompt."""
        assert "specific technical terms an authoritative page" not in QUERY_WRITER_SYSTEM

    def test_it_asks_for_the_question_s_own_vocabulary(self) -> None:
        assert "own vocabulary" in QUERY_WRITER_SYSTEM

    def test_it_bounds_query_length(self) -> None:
        assert "three to eight words" in QUERY_WRITER_SYSTEM

    def test_it_protects_names_dates_and_versions_from_that_bound(self) -> None:
        """A raw word cap would truncate "GPT-4 Turbo" or a release
        date, which changes the question rather than shortening it."""
        lowered = QUERY_WRITER_SYSTEM.lower()
        assert "proper name" in lowered
        assert "version number" in lowered
        assert "date" in lowered

    def test_it_states_the_consequence_rather_than_only_the_rule(self) -> None:
        """The prompt carries the observed failure, so a future edit
        can see what the rule is protecting against."""
        assert "overfitting" in QUERY_WRITER_SYSTEM


class TestQueryStyleDependsOnTheAnswerShape:
    """A causes question and a comparison do not want the same query."""

    @pytest.mark.parametrize(
        "shape,expected",
        [
            ("list", "reader-level"),
            ("causal", "why something happens"),
            ("definition", "what is X"),
            ("comparison", "both subjects together"),
            ("numeric", "quantity"),
            ("procedural", "how to do something"),
            ("temporal", "when"),
        ],
    )
    def test_each_shape_gets_its_own_guidance(self, shape: str, expected: str) -> None:
        rendered = query_writer_user("SQ1: x", [], question="q", answer_shape=shape)
        assert expected in rendered, rendered

    def test_an_unknown_shape_adds_no_guidance_rather_than_the_wrong_guidance(self) -> None:
        rendered = query_writer_user("SQ1: x", [], question="q", answer_shape="")
        assert "Shape of answer required" not in rendered

    def test_the_user_s_original_wording_is_supplied(self) -> None:
        """The sub-questions have already been rephrased once by the
        planner; rebuilding the topic from them is how the topic
        drifted."""
        rendered = query_writer_user(
            "SQ1: x", [], question="What are the main causes of overfitting?"
        )
        assert "What are the main causes of overfitting?" in rendered

    def test_previously_issued_queries_are_still_listed(self) -> None:
        rendered = query_writer_user("SQ1: x", ["earlier query"], question="q")
        assert "earlier query" in rendered


class TestTheRationaleReachesTheQuery:
    """`SearchQuery.rationale` existed and was never populated.

    A query cannot be judged on its own: the overfitting run's queries
    looked plausible until you could see which part of the answer each
    was meant to supply.
    """

    def test_the_node_carries_the_rationale_out_of_the_model_call(self) -> None:
        import inspect

        from agentic_research.graph.nodes import planning

        body = inspect.getsource(planning.generate_queries)
        # The flattening that drops it is the tell for six earlier
        # defects in this repository.
        assert "(q.sub_question_id, q.text)" not in body
        assert "rationale=rationale.strip()" in body

    def test_the_fallback_path_also_records_why(self) -> None:
        import inspect

        from agentic_research.graph.nodes import planning

        body = inspect.getsource(planning.generate_queries)
        assert "fallback: the sub-question text" in body

    def test_query_length_is_logged_as_a_number(self) -> None:
        from agentic_research.graph.nodes.planning import _median_word_count

        assert _median_word_count(["a b c", "a b c d e"]) == 4.0
        assert _median_word_count(["one two three"]) == 3.0
        assert _median_word_count([]) == 0.0

    def test_the_jargon_query_that_failed_measures_as_long(self) -> None:
        """Non-vacuity for the metric: the query that caused the
        failure must register above the guidance band."""
        from agentic_research.graph.nodes.planning import _median_word_count

        failed = (
            "parametric knowledge long tail facts factual recall "
            "generalization failures language models hallucinations"
        )
        assert _median_word_count([failed]) > 8
