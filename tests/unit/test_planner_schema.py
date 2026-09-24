"""The planner asks only for what the engine consumes.

A corpus freeze failed because of this. `qwen3:4b` spent most of its
2,000-token output allowance writing a `strategy_note` and a `rationale`
per sub-question, truncated mid-JSON before finishing the list, failed
structured-output repair twice, and fell back to a single research
dimension. The frozen corpus was unusable.

Neither field had an operational consumer. Both were written, stored, and
never read: no query generation, coverage assessment or synthesis step
touched them. Removing them from the LLM schema is the fix. Raising the
token cap would only have bought more room for narration.
"""

from __future__ import annotations

import json

from agentic_research.schemas import PlanOut, SubQuestionOut

# Roughly four characters per token, the same estimate the budget uses.
CHARS_PER_TOKEN = 4
PLANNER_OUTPUT_CAP_TOKENS = 2_000


class TestTheSchemaAsksOnlyForUsedFields:
    def test_the_planner_no_longer_emits_prose(self) -> None:
        assert "strategy_note" not in PlanOut.model_fields
        assert "rationale" not in SubQuestionOut.model_fields

    def test_a_sub_question_carries_only_text_and_priority(self) -> None:
        assert set(SubQuestionOut.model_fields) == {"text", "priority"}

    def test_the_prompt_does_not_ask_for_an_explanation(self) -> None:
        from agentic_research.graph.prompts import PLANNER_SYSTEM

        assert "do not explain your" in PLANNER_SYSTEM.lower()


class TestAFullPlanFitsInTheOutputCap:
    """The property that actually failed: a complete, useful plan has to
    fit in the planner's allowance with room to spare."""

    @staticmethod
    def _plan(count: int, question_chars: int = 140) -> str:
        return PlanOut(
            sub_questions=[
                SubQuestionOut(text="q" * question_chars, priority=(i % 3) + 1)
                for i in range(count)
            ]
        ).model_dump_json()

    def test_eight_long_sub_questions_fit_comfortably(self) -> None:
        """Eight is the schema maximum and 140 characters is a long
        research question, so this is the worst realistic case."""
        tokens = len(self._plan(8)) / CHARS_PER_TOKEN
        assert tokens < PLANNER_OUTPUT_CAP_TOKENS / 2, (
            f"a maximal plan needs {tokens:.0f} tokens of a {PLANNER_OUTPUT_CAP_TOKENS} cap"
        )

    def test_the_typical_six_question_plan_is_far_inside_the_cap(self) -> None:
        tokens = len(self._plan(6)) / CHARS_PER_TOKEN
        assert tokens < PLANNER_OUTPUT_CAP_TOKENS / 4

    def test_removing_the_prose_fields_shrank_the_requested_output(self) -> None:
        """The same plan with the old fields, to size what was removed."""
        lean = len(self._plan(6))
        with_prose = len(
            json.dumps(
                {
                    "strategy_note": "s" * 300,
                    "sub_questions": [
                        {"text": "q" * 140, "rationale": "r" * 220, "priority": 1} for _ in range(6)
                    ],
                }
            )
        )
        assert with_prose > lean * 2, "the prose fields were the bulk of the output"

    def test_the_json_schema_itself_stayed_small(self) -> None:
        """The schema is sent on every planner call, so it costs input
        tokens on each one too."""
        assert len(json.dumps(PlanOut.model_json_schema())) < 2_000


class TestHistoricalArtifactsStillLoad:
    def test_a_plan_without_a_strategy_note_is_valid(self) -> None:
        from agentic_research.models import QueryAnalysis, ResearchPlan, SubQuestion

        plan = ResearchPlan(
            analysis=QueryAnalysis(original_query="q", normalized_query="q", intent="compare"),
            sub_questions=[SubQuestion(id="SQ1", text="A question?")],
        )
        assert plan.strategy_note == ""
        assert plan.sub_questions[0].rationale == ""

    def test_an_old_artifact_carrying_both_fields_still_parses(self) -> None:
        """Recordings and frozen corpora predate the removal."""
        from agentic_research.models import ResearchPlan

        plan = ResearchPlan.model_validate(
            {
                "analysis": {
                    "original_query": "q",
                    "normalized_query": "q",
                    "intent": "compare",
                },
                "sub_questions": [
                    {"id": "SQ1", "text": "A question?", "rationale": "because", "priority": 1}
                ],
                "strategy_note": "an old note",
            }
        )
        assert plan.strategy_note == "an old note"
        assert plan.sub_questions[0].rationale == "because"
