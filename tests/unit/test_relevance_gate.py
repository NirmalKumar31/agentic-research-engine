"""Whether a claim answers the question, asked separately from whether
it is true.

Every other gate asks whether the evidence carries the claim. None
asked whether the claim had anything to do with the question, so a
live run answered "how does a large language model differ from a
neural network" with five supported definitions of a large language
model, and a fixture answers a question about PostgreSQL with a MySQL
benchmark at entailment 0.999.
"""

from __future__ import annotations

from agentic_research.answer_contract import build_contract
from agentic_research.citations.relevance import assess_relevance

RELEVANT = True  # what a permissive independent judgement would say


def check(claim: str, slot: str | None, contract, **kw):
    return assess_relevance(claim, slot, contract, model_says_relevant=RELEVANT, **kw)


COMPARISON = build_contract(
    "difference between llm and neural networks?",
    "comparison",
    entities=["large language model", "neural network"],
)
POSTGRES = build_contract(
    "How fast is PostgreSQL at bulk inserts?", "numeric", entities=["PostgreSQL"]
)
ATTENTION = build_contract(
    "How does self-attention work in transformers?",
    "definition",
    entities=["self-attention", "transformers"],
)
VECTORDB = build_contract("What is a vector database?", "definition", entities=["vector database"])


class TestTheLiveFailure:
    def test_a_definition_cannot_fill_a_contrast_slot(self) -> None:
        """Five of these published on the deployed service."""
        v = check(
            "A large language model is a machine learning model for language tasks.",
            "direct_contrast",
            COMPARISON,
        )
        assert v.publishable is False
        assert "contrast" in v.reason

    def test_an_actual_contrast_publishes(self) -> None:
        v = check(
            "Large language models differ from earlier neural networks in scale "
            "and in a self-supervised training objective.",
            "direct_contrast",
            COMPARISON,
        )
        assert v.publishable is True

    def test_naming_both_subjects_is_not_a_contrast(self) -> None:
        """ "LLMs are neural networks" names both and contrasts nothing."""
        v = check("Large language models are neural networks.", "direct_contrast", COMPARISON)
        assert v.publishable is False


class TestWrongSubject:
    def test_a_different_named_product_is_refused(self) -> None:
        """Supported at 0.999 and about another database entirely."""
        v = check(
            "MySQL sustained 42,000 bulk inserts per second on the benchmark hardware.",
            "measured_value",
            POSTGRES,
        )
        assert v.publishable is False
        assert v.entity_aligned is False
        assert "MySQL" in v.reason

    def test_an_off_topic_definition_is_refused(self) -> None:
        v = check(
            "Transformer attention computes a weighted sum over value vectors.",
            "definition",
            VECTORDB,
        )
        assert v.publishable is False


class TestDifferentWordsForTheSameThing:
    """The over-correction these exist to prevent. A source may answer
    correctly without using the question's vocabulary."""

    def test_a_paper_that_never_says_self_attention_still_answers(self) -> None:
        v = check(
            "Scaled dot-product attention computes a weighted sum of value vectors "
            "weighted by the softmax of query-key dot products.",
            "definition",
            ATTENTION,
        )
        assert v.publishable is True

    def test_an_acronym_answers_the_spelled_out_question(self) -> None:
        c = build_contract(
            "What is a large language model?", "definition", entities=["large language model"]
        )
        assert check("An LLM is trained on text corpora.", "definition", c).publishable

    def test_a_plural_answers_a_singular_question(self) -> None:
        c = build_contract("What is a neural network?", "definition", entities=["neural network"])
        assert check(
            "Neural networks are trained by gradient descent.", "definition", c
        ).publishable

    def test_an_inflected_verb_answers_a_noun_phrase(self) -> None:
        c = build_contract(
            "How do you rotate an API key?", "procedural", entities=["API key rotation"]
        )
        v = check(
            "Rotating a key means creating a second key, then revoking the first.", "steps", c
        )
        assert v.publishable is True


class TestPeriods:
    def test_a_figure_from_the_wrong_year_is_refused(self) -> None:
        c = build_contract(
            "How many advisories did the project publish in 2025?",
            "temporal",
            entities=["the project"],
        )
        v = check("The project published 14 security advisories in 2023.", "figure_for_period", c)
        assert v.publishable is False
        assert v.period_aligned is False

    def test_a_claim_naming_no_year_is_not_thereby_wrong(self) -> None:
        c = build_contract(
            "How many advisories were published in 2025?", "temporal", entities=["advisories"]
        )
        v = check("Fourteen advisories were published.", "figure_for_period", c)
        assert v.publishable is True


class TestItFailsClosed:
    def test_a_missing_judgement_withholds(self) -> None:
        """An unanswered relevance question is not a yes.

        The claim passes every deterministic check, so the missing
        judgement is the only thing left to refuse it -- otherwise
        this would pass for the wrong reason.
        """
        v = assess_relevance(
            "A vector database stores embeddings and retrieves them by nearest-neighbour search.",
            "definition",
            VECTORDB,
            model_says_relevant=None,
        )
        assert v.publishable is False
        assert v.checked is False

    def test_a_negative_judgement_withholds(self) -> None:
        v = assess_relevance(
            "A vector database stores embeddings.",
            "definition",
            VECTORDB,
            model_says_relevant=False,
        )
        assert v.publishable is False

    def test_an_unusable_contract_withholds_everything(self) -> None:
        broken = build_contract("what is an llm?", "comparison", entities=["llm"])
        v = check("Anything.", "direct_contrast", broken)
        assert v.publishable is False
        assert v.checked is False

    def test_a_slot_the_question_never_asked_for_is_refused(self) -> None:
        v = check("A vector database stores embeddings.", "tradeoffs", VECTORDB)
        assert v.publishable is False
        assert "not required" in v.reason

    def test_a_claim_declaring_no_slot_is_refused(self) -> None:
        assert check("A vector database stores embeddings.", None, VECTORDB).publishable is False
