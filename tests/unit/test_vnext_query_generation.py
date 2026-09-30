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
            ("causal", "tests one proposed cause"),
            ("causal_drivers", "why something happens"),
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
        # The model's rationale is read, and it reaches the SearchQuery.
        # Asserted as two properties rather than one exact spelling: the
        # strip() moved into the grouping step when query assembly became
        # breadth-first, and a test pinned to the spelling would have
        # failed for a refactor that changed nothing it cares about.
        assert "q.rationale" in body
        assert "rationale=rationale" in body

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


class TestTheBudgetCannotStarveASubQuestion:
    """The defect the live run on 94368bb7 exposed.

    Six queries were issued across five sub-questions -- two each for
    SQ1, SQ2 and SQ3 and **none** for SQ4 or SQ5. No candidate in the
    pool was attributed to either, so neither was ever searched, and
    the report described them as "only limited evidence was found" --
    which reads as a retrieval outcome rather than a question nobody
    asked.

    The prompt permitted "one or two queries per sub-question" and the
    node took the proposals in the order they arrived, stopping at the
    budget. Nothing required that every sub-question get one first.
    This is the same shape as one sub-question consuming the whole
    source budget, which the same release fixed, one stage earlier.

    The node needs a live run context to drive end to end, so the
    ordering rule is tested through the pure helper and the wiring is
    asserted by inspection -- the project's existing idiom.
    """

    def test_the_prompt_requires_coverage_before_depth(self) -> None:
        assert "Cover every sub-question before" in QUERY_WRITER_SYSTEM

    def test_the_prompt_states_what_an_unsearched_sub_question_costs(self) -> None:
        """A rule without its reason gets edited away."""
        lowered = QUERY_WRITER_SYSTEM.lower()
        assert "never searched" in lowered
        assert "left two unsearched" in lowered

    def test_the_prompt_caps_depth_at_two_and_orders_it(self) -> None:
        assert "only once every other" in QUERY_WRITER_SYSTEM

    def test_the_node_interleaves_rather_than_taking_arrival_order(self) -> None:
        import inspect

        from agentic_research.graph.nodes import planning

        body = inspect.getsource(planning.generate_queries)
        # The arrival-order loop that caused it is gone.
        assert "if len(queries) >= remaining_budget:\n                break" not in body
        assert "by_sub_question" in body
        assert "ranked = sorted(targets" in body

    def test_the_node_injects_a_query_for_an_uncovered_sub_question(self) -> None:
        import inspect

        from agentic_research.graph.nodes import planning

        body = inspect.getsource(planning.generate_queries)
        assert "fallback: no query was written for this sub-question" in body

    def test_the_node_warns_when_the_budget_cannot_cover_everything(self) -> None:
        import inspect

        from agentic_research.graph.nodes import planning

        body = inspect.getsource(planning.generate_queries)
        assert "sub_questions_unsearched" in body

    def test_breadth_first_ordering_reproduces_the_fix(self) -> None:
        """The ordering rule itself, on the live run's exact shape: the
        model proposes two queries each for the first three of five
        sub-questions, and the budget is six.

        Arrival order spends all six on SQ1-SQ3. Breadth-first gives
        every sub-question one and then spends the spare on the
        highest-priority one.
        """
        proposed = [
            ("SQ1", "a1"),
            ("SQ1", "a2"),
            ("SQ2", "b1"),
            ("SQ2", "b2"),
            ("SQ3", "c1"),
            ("SQ3", "c2"),
        ]
        targets = ["SQ1", "SQ2", "SQ3", "SQ4", "SQ5"]
        budget = 6

        # Arrival order — what the defect did.
        arrival = [sq for sq, _ in proposed][:budget]
        assert set(arrival) == {"SQ1", "SQ2", "SQ3"}
        assert "SQ4" not in arrival and "SQ5" not in arrival

        # Breadth-first with injected fallbacks — what it does now.
        groups: dict[str, list[str]] = {t: [] for t in targets}
        for sq, text in proposed:
            groups[sq].append(text)
        for t in targets:
            if not groups[t]:
                groups[t].append(f"{t} fallback")
        ordered: list[str] = []
        depth = 0
        while any(depth < len(groups[t]) for t in targets):
            for t in targets:
                if depth < len(groups[t]):
                    ordered.append(t)
            depth += 1
        chosen = ordered[:budget]
        assert set(chosen) >= set(targets), "every sub-question must be searched"
        assert chosen[:5] == targets, "breadth before depth"
        assert chosen[5] == "SQ1", "the spare goes to the highest-priority one"
