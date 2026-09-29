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

    def test_a_claim_declaring_no_slot_still_needs_the_judgement(self) -> None:
        """Reversed, deliberately, and narrowed to what fails closed.

        This previously asserted that a claim declaring no slot is
        refused outright. That was grouped here by analogy with the
        cases above rather than argued, and it cost more than it
        bought: a local run on qwen3:4b, which omits the field on
        every claim, published nothing at all.

        What fails closed is the judgement, not the field. A slotless
        claim still cannot publish without an affirmative relevance
        verdict -- that is asserted here -- and it counts toward no
        slot, so coverage still reports the question unanswered.
        Support and the independent judgement are untouched.
        """
        assert (
            assess_relevance(
                "A vector database stores embeddings.",
                None,
                VECTORDB,
                model_says_relevant=None,
            ).publishable
            is False
        )
        assert check("A vector database stores embeddings.", None, VECTORDB).publishable is True


class TestAMissingSlotIsNotARejection:
    """A local run found this; no test did.

    The fake synthesiser in tests/fakes.py declares an answer_slot on
    every claim, so the whole suite exercised a world in which models
    always comply. qwen3:4b does not. A real local run withheld three
    otherwise-publishable claims with "the claim declares no answer
    slot" -- refused for a missing field rather than for anything
    about what they said -- and published nothing at all.

    The slot is the synthesiser's statement of intent, not a property
    of the claim. Its absence means the slot-specific checks cannot
    run, not that the claim is irrelevant.
    """

    def test_a_claim_without_a_slot_can_still_publish(self) -> None:
        verdict = check(
            "Large language models are built on transformer architectures.",
            None,
            COMPARISON,
        )
        assert verdict.publishable, verdict.reason

    def test_it_says_no_slot_was_declared_rather_than_claiming_one(self) -> None:
        """The record must not imply a slot was satisfied. Coverage
        reads slots, and inventing one here would make a report claim
        it answered a part nothing declared."""
        verdict = check("Large language models use transformers.", None, COMPARISON)
        assert verdict.answer_slot is None
        assert verdict.slot_satisfied is False

    def test_the_slotless_claim_still_needs_the_judgement(self) -> None:
        """Nothing structural rejected it is not the same as something
        affirmed it, and with no slot the judge is all there is."""
        from agentic_research.citations.relevance import assess_relevance

        withheld = assess_relevance(
            "Large language models use transformers.",
            None,
            COMPARISON,
            model_says_relevant=None,
        )
        assert not withheld.publishable

        rejected = assess_relevance(
            "Large language models use transformers.",
            None,
            COMPARISON,
            model_says_relevant=False,
        )
        assert not rejected.publishable

    def test_the_checks_that_need_no_slot_still_run(self) -> None:
        """Non-vacuity. Skipping the slot checks must not skip the
        rest: a claim about a different named subject is refused
        whether or not it declared a slot."""
        verdict = check(
            "MySQL sustained 40,000 inserts per second in the benchmark.",
            None,
            POSTGRES,
        )
        assert not verdict.publishable
        assert "MySQL" in verdict.reason or "PostgreSQL" in verdict.reason

    def test_an_unusable_contract_still_withholds_without_a_slot(self) -> None:
        unusable = build_contract("compare them", "comparison", entities=["only one"])
        assert not unusable.usable
        assert not check("Anything at all.", None, unusable).publishable

    def test_a_declared_but_unknown_slot_is_still_refused(self) -> None:
        """Only the *absent* slot is forgiven. A slot the question
        never asked for is a claim answering something else."""
        verdict = check("Large language models use transformers.", "invented_slot", COMPARISON)
        assert not verdict.publishable
        assert "not required by this question" in verdict.reason


class TestCoverageStaysHonestWithoutSlots:
    def test_slotless_claims_do_not_count_as_answering(self) -> None:
        """The other half of the fix. Claims may now publish without a
        slot, and a report built only from those must still report
        that it did not answer -- nothing declared which part of the
        question it filled."""
        from agentic_research.answer_coverage import assess_coverage

        coverage = assess_coverage(COMPARISON, [])
        assert not coverage.answered
        assert coverage.limitations()
