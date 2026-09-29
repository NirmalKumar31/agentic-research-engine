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


class TestTheJudgeIsToldWhatTheSlotsMean:
    """A bare slot name is not a question a model can answer well.

    The judge was shown `direct_contrast`, `dimension`,
    `relationship` and nothing else, while the contract carried a
    sentence describing each. A capable model infers them; a 4B one
    does not, and a local run rejected "Large language models are a
    specific type of neural network architecture" for not addressing
    how the two relate -- which is the `relationship` slot almost
    verbatim.

    Fourth instance of the same shape in this release: the engine
    computes something useful and it stops at a boundary.
    """

    @staticmethod
    def prompt(contract) -> str:
        from agentic_research.graph.prompts import relevance_user

        return relevance_user(contract.question, contract.required_slots, ["A claim."])

    def test_each_slot_carries_its_description(self) -> None:
        text = self.prompt(COMPARISON)
        for slot in COMPARISON.required_slots:
            assert slot.name in text
            assert slot.description in text, f"{slot.name} lost its description"

    def test_the_required_slot_is_marked(self) -> None:
        """A claim filling an optional part is worth less than one
        filling the part the answer turns on, and the judge could not
        tell them apart."""
        text = self.prompt(COMPARISON)
        core = [s for s in COMPARISON.required_slots if s.core]
        assert core
        for slot in core:
            assert f"{slot.name} (required)" in text
        for slot in COMPARISON.required_slots:
            if not slot.core:
                assert f"{slot.name} (required)" not in text

    def test_the_instruction_was_not_loosened(self) -> None:
        """A line telling the judge that filling one part is enough was
        tried and reverted. It loosens the gate this release exists to
        add, three local runs showed no effect, and measuring it on the
        hosted critic costs a paid run. The descriptions are pure added
        information; the instruction is not, so it stays out."""
        text = self.prompt(COMPARISON)
        assert "even if it does not cover the others" not in text

    def test_a_contract_with_no_slots_still_renders(self) -> None:
        unusable = build_contract("compare them", "comparison", entities=["only one"])
        assert not unusable.usable
        assert "(none stated)" in self.prompt(unusable)

    def test_the_claims_are_still_indexed(self) -> None:
        """The verdicts come back by index; losing the numbering would
        silently misattribute every judgement."""
        from agentic_research.graph.prompts import relevance_user

        text = relevance_user(COMPARISON.question, COMPARISON.required_slots, ["A.", "B."])
        assert "0. A." in text
        assert "1. B." in text


class TestTheJudgeIsAskedTheContractsQuestion:
    """The critic and the contract disagreed, in production, twice.

    Two hosted runs on the same question:

        "LLMs are built upon deep neural networks."  -> relevant
        "An LLM is a neural network."                -> irrelevant,
            "states the relationship but does not explain how an LLM
             differs from a neural network"

    Those are the same answer. The second was refused for not being a
    contrast -- which is exactly what the `relationship` slot exists
    to say is unnecessary when one subject is a kind of the other.

    The judge was asked "does this help answer the question?" while
    being shown a list of parts it was not asked about. It applied its
    own notion of answering and contradicted the contract the report
    is scored against. Asking it the contract's question is a
    tightening, not a loosening: it may no longer freelance, and a
    claim filling no listed part still fails.
    """

    def test_the_instruction_references_the_listed_parts(self) -> None:
        from agentic_research.graph.prompts import relevance_user

        text = relevance_user(COMPARISON.question, COMPARISON.required_slots, ["A claim."])
        assert "fills one of the parts" in text

    def test_it_says_to_judge_each_claim_alone(self) -> None:
        """The second run put ten claims in one batch and rejected
        nine. Whether that was comparative judging is not proven, but
        the prompt should not leave it open -- these are different
        claims, not candidates for one place."""
        from agentic_research.graph.prompts import relevance_user

        text = relevance_user(COMPARISON.question, COMPARISON.required_slots, ["A.", "B."])
        assert "Judge each claim on its own" in text
        assert "competing for one place" in text

    def test_filling_no_part_is_still_a_refusal(self) -> None:
        """Non-vacuity. Aligning the judge to the contract must not
        turn it into a rubber stamp."""
        from agentic_research.graph.prompts import RELEVANCE_SYSTEM

        assert "fills none of the listed parts does not answer" in RELEVANCE_SYSTEM
        assert "not a licence to accept everything" in RELEVANCE_SYSTEM

    def test_the_subset_case_is_stated(self) -> None:
        """The specific inconsistency, written down with the two claims
        that produced it, so the next reader knows it was measured
        rather than imagined."""
        from agentic_research.graph.prompts import RELEVANCE_SYSTEM

        assert "a kind of the other" in RELEVANCE_SYSTEM
        assert "An LLM is a neural network" in RELEVANCE_SYSTEM

    def test_the_structural_gate_is_untouched(self) -> None:
        """The prompt changed; the free deterministic checks did not.
        A claim about a different named subject is still refused
        before any model sees it."""
        verdict = check(
            "MySQL sustained 40,000 inserts per second in the benchmark.",
            "measured_value",
            POSTGRES,
        )
        assert not verdict.publishable
