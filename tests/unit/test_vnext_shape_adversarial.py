"""Adversarial answer-shape evaluation.

Stability across wording is the property that matters. The live defect
was not that one question got the wrong shape; it was that two
phrasings of the *same* question got different shapes, so identical
questions were held to different requirements and one of them reported
itself unanswered.

Read the whole file as the specification of what the deterministic
reader claims. Where it returns ``None`` it is claiming nothing, and
the model's own reading stands -- that is a deliberate answer, not a
gap, and the ambiguity cases below pin it.
"""

from __future__ import annotations

import pytest

from agentic_research.answer_contract import QuestionType
from agentic_research.question_form import comparison_sides, shape_from_wording

# Near-equivalent phrasings that must land on one shape each.
EQUIVALENCE_CLASSES: dict[QuestionType, list[str]] = {
    QuestionType.CAUSAL_DRIVERS: [
        "What are the main causes of hallucination in large language models?",
        "What causes hallucinations in large language models?",
        "What are the main reasons LLMs hallucinate?",
        "What are the risk factors for overfitting?",
        "What factors contribute to overfitting?",
        "Why do large language models hallucinate?",
        "Why does overfitting happen?",
    ],
    QuestionType.CAUSAL: [
        "Does smoking cause lung cancer?",
        "Can overfitting cause poor generalization?",
        "Did the regulation lead to lower prices?",
        "Is dropout responsible for better generalization?",
        "Does more training data result in better accuracy?",
    ],
    QuestionType.LIST: [
        "What types of regularisation prevent overfitting?",
        "What are the benefits of retrieval-augmented generation?",
        "What are the drawbacks of fine-tuning?",
        "What are some examples of vector databases?",
        "What are the components of a RAG pipeline?",
    ],
    QuestionType.NUMERIC: [
        "What is the context window size of GPT-4 Turbo?",
        "How many parameters does Llama 3 have?",
        "How much does GPT-4 Turbo cost per million tokens?",
        "What is the latency of a typical vector search?",
        "What percentage of queries hit the cache?",
        "How long does fine-tuning take?",
    ],
    QuestionType.PROCEDURAL: [
        "How do I fine-tune a language model?",
        "How can we deploy a vector database?",
        "How to build a RAG pipeline?",
        "What are the steps to evaluate a retriever?",
    ],
    QuestionType.TEMPORAL: [
        "When was GPT-4 released?",
        "When did OpenAI publish the GPT-4 system card?",
        "What is the latest version of Llama?",
    ],
    QuestionType.DEFINITION: [
        "What is retrieval-augmented generation?",
        "Define retrieval-augmented generation.",
        "What is the meaning of overfitting?",
    ],
    QuestionType.RECOMMENDATION: [
        "Should we adopt retrieval-augmented generation?",
        "Should I use a vector database?",
        "Which should I choose for analytics?",
        "Is it worth fine-tuning a smaller model?",
    ],
    QuestionType.COMPARISON: [
        "How does retrieval-augmented generation differ from fine-tuning?",
        "How does retrieval-augmented generation differ from fine-tuning for language models?",
        "What is the difference between RAG and fine-tuning?",
        "What are the differences between Postgres and MySQL?",
        "RAG versus fine-tuning",
        "How do Postgres, MySQL and SQLite differ?",
        "How does Postgres compare to MySQL?",
    ],
}


class TestEveryPhrasingInAClassAgrees:
    @pytest.mark.parametrize(
        "expected,question",
        [(shape, q) for shape, qs in EQUIVALENCE_CLASSES.items() for q in qs],
        ids=[q[:44] for qs in EQUIVALENCE_CLASSES.values() for q in qs],
    )
    def test_the_wording_reads_as_its_class(self, expected: QuestionType, question: str) -> None:
        assert shape_from_wording(question) is expected

    def test_every_supported_shape_is_exercised(self) -> None:
        """Except SYNTHESIS, which is decided by named parts rather than
        by wording -- see the multi-part cases below."""
        from agentic_research.answer_contract import CANONICAL_SLOTS

        covered = set(EQUIVALENCE_CLASSES)
        missing = set(CANONICAL_SLOTS) - covered - {QuestionType.SYNTHESIS}
        assert missing == set()


class TestContextNounsNeverBecomeSides:
    """The RAG defect, generalised across settings and prepositions."""

    @pytest.mark.parametrize(
        "question,expected",
        [
            (
                "How does retrieval-augmented generation differ from fine-tuning for language models?",
                ("retrieval-augmented generation", "fine-tuning"),
            ),
            (
                "How does RAG differ from fine-tuning in production?",
                ("RAG", "fine-tuning"),
            ),
            (
                "What is the difference between Postgres and MySQL for analytics?",
                ("Postgres", "MySQL"),
            ),
            (
                "How does Redis compare to Memcached under heavy load?",
                ("Redis", "Memcached"),
            ),
        ],
    )
    def test_the_setting_is_excluded(self, question: str, expected: tuple[str, ...]) -> None:
        assert comparison_sides(question) == expected

    def test_a_three_way_comparison_keeps_three(self) -> None:
        assert comparison_sides("How do Postgres, MySQL and SQLite differ?") == (
            "Postgres",
            "MySQL",
            "SQLite",
        )

    def test_sides_are_distinct(self) -> None:
        sides = comparison_sides("How does RAG differ from RAG?")
        assert len(sides) == len({s.lower() for s in sides})


class TestAmbiguityIsLeftToTheModel:
    """Returning None is the answer, not a failure to find one.

    Guessing broadly is how a definition question became a list. The
    reader claims only what the wording states outright.
    """

    @pytest.mark.parametrize(
        "question",
        [
            "Tell me about vector databases in production.",
            "Vector databases",
            "I need help choosing a database.",
            "Explain the tradeoffs we should think about.",
            "",
            "   ",
        ],
    )
    def test_no_shape_is_claimed(self, question: str) -> None:
        assert shape_from_wording(question) is None

    @pytest.mark.parametrize(
        "question",
        [
            "Tell me about vector databases.",
            "Overview of retrieval strategies",
            "",
        ],
    )
    def test_no_sides_are_invented(self, question: str) -> None:
        assert comparison_sides(question) == ()

    def test_the_word_versus_in_passing_is_not_a_comparison(self) -> None:
        """A keyword match would fire here; sides are required."""
        assert comparison_sides("What does versus mean in legal citations?") == ()


class TestMultiPartQuestions:
    """Multi-part questions are decided by named parts, not wording.

    Left to the model deliberately: splitting "what is X and how much
    does it cost" on "and" would also split "risks and benefits of X",
    which is one list. The fallback when the model names no parts is
    pinned in test_quality_pipeline_integration.
    """

    def test_a_conjunction_is_not_split_into_parts_by_wording(self) -> None:
        # Reads as a list, because "benefits" is explicit. That is the
        # correct reading and it is not SYNTHESIS.
        assert shape_from_wording("What are the risks and benefits of RAG?") is QuestionType.LIST

    def test_wording_reads_only_one_half_of_a_two_part_question(self) -> None:
        """A recorded limit, not a claim of correctness.

        "What is RAG, and how much does it cost to run?" contains an
        explicit numeric form, so the reader returns NUMERIC -- it has
        no way to see the question has two halves. Overriding a model
        that correctly said `synthesis` would make the answer worse, so
        `contract_from_analysis` does not apply the override when the
        model named parts. That guard is the real protection.
        """
        q = "What is RAG, and how much does it cost to run?"
        assert shape_from_wording(q) is QuestionType.NUMERIC

    def test_named_parts_survive_the_wording_override(self) -> None:
        from agentic_research.graph.nodes.planning import contract_from_analysis
        from agentic_research.models import OutputFormat, QueryAnalysis

        q = "What is RAG, and how much does it cost to run?"
        contract = contract_from_analysis(
            QueryAnalysis(
                original_query=q,
                normalized_query=q,
                intent="i",
                output_format=OutputFormat.SYNTHESIS,
                parts=["What is RAG?", "How much does RAG cost to run?"],
            )
        )
        assert contract.question_type is QuestionType.SYNTHESIS
        assert len(contract.core_slots) == 2


class TestPrecedenceIsDeliberate:
    """Where two readings both match, the more specific one wins."""

    def test_a_figure_beats_a_definition(self) -> None:
        """ "What is the context window size of X" opens like a
        definition and asks for a number."""
        assert shape_from_wording("What is the context window size of X?") is QuestionType.NUMERIC

    def test_a_comparison_beats_a_list(self) -> None:
        """ "differences" would match neither, but "types" does -- and a
        comparison of two named things is still a comparison."""
        assert (
            shape_from_wording("What types of difference are there between Postgres and MySQL?")
            is QuestionType.COMPARISON
        )

    def test_a_procedure_beats_a_list(self) -> None:
        assert (
            shape_from_wording("What are the steps to reduce overfitting?")
            is QuestionType.PROCEDURAL
        )

    def test_a_current_figure_reads_as_numeric_not_temporal(self) -> None:
        """Deliberate. The question wants a number; that it is
        time-sensitive is carried by the analysis stage's own
        `time_sensitive` flag rather than by reshaping the answer into
        a timeline."""
        assert shape_from_wording("What is the current context window of GPT-4?") is (
            QuestionType.NUMERIC
        )
        assert shape_from_wording("What is the current price of GPT-4 Turbo?") is (
            QuestionType.NUMERIC
        )


class TestSmallWordingChangesDoNotChangeAnExplicitShape:
    """Paired invariance. The live defect was not a wrong shape; it was
    two phrasings of one question getting different shapes, so identical
    questions were held to different requirements.

    Each pair differs only in something that carries no meaning.
    """

    PAIRS: list[tuple[str, str]] = [
        # Punctuation.
        (
            "What are the main causes of overfitting?",
            "What are the main causes of overfitting",
        ),
        ("How does RAG differ from fine-tuning?", "How does RAG differ from fine-tuning."),
        ("When was GPT-4 released?", "When was GPT-4 released???"),
        # Capitalisation.
        (
            "What are the main causes of overfitting?",
            "WHAT ARE THE MAIN CAUSES OF OVERFITTING?",
        ),
        ("How do I fine-tune a model?", "how do i fine-tune a model?"),
        ("Does smoking cause lung cancer?", "does smoking cause lung cancer?"),
        # Leading and trailing whitespace.
        (
            "What is the context window size of GPT-4 Turbo?",
            "   What is the context window size of GPT-4 Turbo?  ",
        ),
        # Acronym expansion, both directions.
        (
            "How does RAG differ from fine-tuning?",
            "How does retrieval-augmented generation differ from fine-tuning?",
        ),
        (
            "What causes hallucination in LLMs?",
            "What causes hallucination in large language models?",
        ),
        # Small, meaning-preserving wording changes.
        (
            "What are the main causes of overfitting?",
            "What are the principal causes of overfitting?",
        ),
        ("What causes overfitting?", "What causes overfitting in practice?"),
        ("How many parameters does Llama 3 have?", "How many parameters has Llama 3 got?"),
        (
            "How does Postgres differ from MySQL?",
            "How does Postgres differ from MySQL for analytics?",
        ),
    ]

    @pytest.mark.parametrize("left,right", PAIRS, ids=[left[:38] for left, _ in PAIRS])
    def test_the_pair_reads_as_one_shape(self, left: str, right: str) -> None:
        assert shape_from_wording(left) == shape_from_wording(right), (
            f"{left!r} and {right!r} were read as different shapes"
        )

    @pytest.mark.parametrize("left,right", PAIRS, ids=[left[:38] for left, _ in PAIRS])
    def test_the_pair_reads_as_an_explicit_shape(self, left: str, right: str) -> None:
        """Non-vacuity: a pair that both read as None would satisfy the
        test above while proving nothing."""
        assert shape_from_wording(left) is not None, left

    def test_comparison_sides_survive_the_same_changes(self) -> None:
        expected = ("Postgres", "MySQL")
        for question in [
            "How does Postgres differ from MySQL?",
            "how does postgres differ from mysql?",
            "How does Postgres differ from MySQL",
            "  How does Postgres differ from MySQL for analytics?  ",
            "What is the difference between Postgres and MySQL?",
        ]:
            sides = comparison_sides(question)
            assert len(sides) == 2, question
            assert {s.lower() for s in sides} == {e.lower() for e in expected}, question


class TestTheDocumentedClaimsMatchTheCode:
    """The checkpoint once said eight shapes were deterministic. This
    pins what is actually claimed, so the document and the code cannot
    drift."""

    def test_the_shapes_the_document_lists_are_the_shapes_that_fire(self) -> None:
        import pathlib
        import re

        doc = pathlib.Path("docs/ANSWER-SHAPES.md").read_text()
        table = re.findall(r"^\| `([a-z_]+)` \| ", doc, re.M)
        deterministic = set(table)
        # Every shape the document lists must be reachable from wording.
        reached = {
            shape
            for shape in (shape_from_wording(q) for qs in EQUIVALENCE_CLASSES.values() for q in qs)
            if shape is not None
        }
        assert {str(s) for s in reached} <= deterministic

    def test_the_document_states_what_is_model_decided(self) -> None:
        import pathlib

        doc = pathlib.Path("docs/ANSWER-SHAPES.md").read_text()
        assert "remains model-decided" in doc or "What remains model-decided" in doc
        assert "synthesis" in doc
        assert "unusable" in doc
