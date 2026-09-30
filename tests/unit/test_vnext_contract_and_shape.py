"""Phase 0 regression fixtures: the live failures, frozen before the fix.

Every test here drives the real
``AnalysisOut -> QueryAnalysis -> AnswerContract`` path. Asserting against
hand-built dictionaries is what let four earlier defects pass a full suite
while production was wrong, so the analysis object is constructed the way
the planning node constructs it and the contract is derived by the same
function the graph calls.

Baselines these were written against (hosted, commit b16ad010):

* "How does retrieval-augmented generation differ from fine-tuning?"
  published three supported, relevant, cited claims and then reported
  "This report does not answer the question", because the analyst
  returned three entities -- RAG, fine-tuning and *language models* --
  and every entity was treated as a side of the comparison that the
  published claims had to speak about.
* "What are the main causes of hallucination in large language models?"
  classified as ``list`` on one run and ``definition`` on an earlier
  near-identical phrasing.
* "What is the context window size of GPT-4 Turbo?" classified as
  ``definition``, so a figure was checked against a slot asking what the
  subject *is*.
"""

from __future__ import annotations

import pytest

from agentic_research.answer_contract import QuestionType
from agentic_research.graph.nodes.planning import contract_from_analysis
from agentic_research.models import OutputFormat, QueryAnalysis

RAG_QUESTION = (
    "How does retrieval-augmented generation differ from fine-tuning for language models?"
)


def analysis(**kw: object) -> QueryAnalysis:
    """A QueryAnalysis built the way the planning node builds one."""
    base: dict[str, object] = {
        "original_query": kw.pop("original_query", "q"),
        "normalized_query": kw.pop("normalized_query", "q"),
        "intent": "i",
        "output_format": OutputFormat.OVERVIEW,
    }
    base.update(kw)
    return QueryAnalysis(**base)  # type: ignore[arg-type]


class TestAContextNounIsNotAComparisonSide:
    """The RAG-vs-fine-tuning defect.

    "for language models" is the domain both subjects operate in. It is
    useful for retrieval and it is not a third thing being contrasted.
    """

    def _contract(self):
        return contract_from_analysis(
            analysis(
                original_query=RAG_QUESTION,
                normalized_query=RAG_QUESTION,
                output_format=OutputFormat.COMPARISON,
                entities=["retrieval-augmented generation", "fine-tuning", "language models"],
                # A named axis, which the analyst is now asked to
                # propose for every comparison. Without one the sides
                # have nowhere to meet, and two true facts about two
                # subjects are not a contrast.
                dimensions=["knowledge_update"],
            )
        )

    def test_exactly_two_subjects_are_contrasted(self) -> None:
        contract = self._contract()
        assert contract.usable
        assert contract.comparison_subjects == (
            "retrieval-augmented generation",
            "fine-tuning",
        )

    def test_the_context_noun_is_kept_but_not_as_a_side(self) -> None:
        """Dropped entirely it would stop helping retrieval; kept as a
        side it makes the comparison unanswerable."""
        contract = self._contract()
        assert "language models" in contract.entities
        assert "language models" not in contract.comparison_subjects

    def test_one_claim_per_real_side_answers_the_comparison(self) -> None:
        from agentic_research.answer_coverage import assess_coverage

        contract = self._contract()
        from agentic_research.comparison import SideClaim

        coverage = assess_coverage(
            contract,
            ["knowledge_update", "knowledge_update"],
            claims=[
                SideClaim(
                    subject="",
                    text="Keeping a fine-tuning approach current requires retraining.",
                    answer_slot="knowledge_update",
                ),
                SideClaim(
                    subject="",
                    text="Retrieval-augmented generation retrieves passages at query time.",
                    answer_slot="knowledge_update",
                ),
            ],
        )
        assert coverage.answered, coverage.limitations()
        # The contrast exists as structure, referencing the verified
        # claims, rather than as a new sentence nothing checked.
        assert len(coverage.comparison_pairs) == 1
        assert set(coverage.comparison_pairs[0].subjects) == {
            "retrieval-augmented generation",
            "fine-tuning",
        }

    def test_a_genuine_three_way_comparison_still_needs_three(self) -> None:
        q = "How do Postgres, MySQL and SQLite differ?"
        contract = contract_from_analysis(
            analysis(
                original_query=q,
                normalized_query=q,
                output_format=OutputFormat.COMPARISON,
                entities=["Postgres", "MySQL", "SQLite"],
            )
        )
        assert contract.comparison_subjects == ("Postgres", "MySQL", "SQLite")


class TestShapeIsStableAcrossWording:
    """Two phrasings of one question must not get different contracts.

    The model proposes a shape; explicit wording overrides it. The
    override is deliberately narrow -- it fires only on unambiguous
    question forms -- because guessing broadly is how a definition
    question became a list.
    """

    @pytest.mark.parametrize(
        "question,expected",
        [
            (
                "What are the main causes of hallucination in large language models?",
                QuestionType.CAUSAL_DRIVERS,
            ),
            ("What causes hallucinations in large language models?", QuestionType.CAUSAL_DRIVERS),
            ("What are the main reasons LLMs hallucinate?", QuestionType.CAUSAL_DRIVERS),
            ("What are the risk factors for overfitting?", QuestionType.CAUSAL_DRIVERS),
            ("What types of regularisation prevent overfitting?", QuestionType.LIST),
            ("What is the context window size of GPT-4 Turbo?", QuestionType.NUMERIC),
            ("How many parameters does Llama 3 have?", QuestionType.NUMERIC),
            ("How do I fine-tune a language model?", QuestionType.PROCEDURAL),
            ("When was GPT-4 released?", QuestionType.TEMPORAL),
            ("What is retrieval-augmented generation?", QuestionType.DEFINITION),
        ],
    )
    def test_explicit_wording_decides_the_shape(
        self, question: str, expected: QuestionType
    ) -> None:
        """Every case is given the *wrong* model label on purpose: if the
        override does not fire, the model's label survives and the test
        fails."""
        contract = contract_from_analysis(
            analysis(
                original_query=question,
                normalized_query=question,
                output_format=OutputFormat.OVERVIEW,
                entities=["subject"],
            )
        )
        assert contract.question_type is expected

    def test_a_corrected_shape_records_where_it_came_from(self) -> None:
        """An override that leaves no trace is unauditable."""
        q = "What are the main causes of overfitting?"
        contract = contract_from_analysis(
            analysis(
                original_query=q,
                normalized_query=q,
                output_format=OutputFormat.OVERVIEW,
                entities=["overfitting"],
            )
        )
        assert contract.shape_source == "corrected-from-wording"

    def test_an_unopinionated_question_keeps_the_model_shape(self) -> None:
        """The override must not fire on wording it cannot read."""
        q = "Tell me about vector databases in production."
        contract = contract_from_analysis(
            analysis(
                original_query=q,
                normalized_query=q,
                output_format=OutputFormat.DECISION_SUPPORT,
                entities=["vector databases"],
            )
        )
        assert contract.question_type is QuestionType.RECOMMENDATION
        assert contract.shape_source == "model"
