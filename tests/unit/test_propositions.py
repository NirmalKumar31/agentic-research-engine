"""Splitting a claim into what it independently asserts.

A claim bundling a measured figure with an assertion its quote never
contained published at entailment 0.983, because the sentence as a
whole was close enough to the quote as a whole. Verifying the sentence
verified the average of its parts.

The danger in fixing that is over-splitting. "A and B", a comma, or
several verbs do not mean several assertions, and a splitter that
thinks so shreds a procedure into steps and then fails each step for
lacking evidence the original never needed. Half these tests exist to
stop that.
"""

from __future__ import annotations

import pytest

from agentic_research.citations.fake_nli import FakeScorer
from agentic_research.citations.propositions import decompose
from agentic_research.citations.semantic import verify_claim


def count(text: str) -> int:
    return len(decompose(text))


class TestOneAssertionStaysWhole:
    """Each of these looks splittable and is not."""

    def test_a_procedure_is_one_assertion(self) -> None:
        assert (
            count(
                "Rotating a key means creating a second key, updating clients to use it, "
                "then revoking the first."
            )
            == 1
        )

    def test_a_compound_predicate_is_one_assertion(self) -> None:
        assert (
            count(
                "A vector database stores high-dimensional embeddings and retrieves them "
                "by approximate nearest-neighbour search."
            )
            == 1
        )

    def test_a_list_of_dimensions_is_one_assertion(self) -> None:
        """ "in scale, architecture, and objective" is three things one
        claim says, not three claims."""
        assert (
            count(
                "Large language models differ from earlier neural networks in scale, "
                "decoder-only architecture, and a self-supervised next-token objective."
            )
            == 1
        )

    def test_a_pronoun_continues_the_same_subject(self) -> None:
        assert count("The model stores embeddings, and it retrieves them quickly.") == 1

    def test_a_plain_sentence_is_one_assertion(self) -> None:
        assert count("Code review reduces production defects.") == 1
        assert len(decompose("Code review reduces production defects.")) <= 1


class TestSeparateAssertionsSplit:
    def test_a_second_subject_is_a_second_assertion(self) -> None:
        assert (
            count(
                "In a randomised trial developers took 19% longer, and the authors note "
                "the result contradicts participant self-reports."
            )
            == 2
        )

    def test_two_named_subjects_split(self) -> None:
        assert (
            count(
                "REST exposes resources at URLs, and GraphQL lets a client request "
                "exactly the fields it needs."
            )
            == 2
        )

    def test_an_explanation_is_its_own_assertion(self) -> None:
        """A supported figure must not carry an unsupported reason into
        print on its back."""
        assert count("Revenue fell 12%, because the largest customer did not renew.") == 2

    def test_a_consequence_is_its_own_assertion(self) -> None:
        assert count("Latency rose by 40 ms, which means the cache was bypassed.") == 2


class TestSupportIsCheckedPerProposition:
    QUOTE = (
        "In a randomised controlled trial, developers allowed to use AI tools took "
        "19% longer to complete tasks than the control group."
    )
    SUPPORTED = "Developers took 19% longer to complete tasks in a randomised trial"
    UNSUPPORTED = "the authors note the result contradicts participant self-reports"
    BUNDLED = f"{SUPPORTED}, and {UNSUPPORTED}."

    def scorer(self) -> FakeScorer:
        # Generous on the whole sentence and on the supported half;
        # nothing for the half the quote does not contain.
        return FakeScorer(
            {
                (self.QUOTE, self.BUNDLED): (0.983, 0.017, 0.0),
                (self.QUOTE, self.SUPPORTED): (0.991, 0.009, 0.0),
            },
            default=(0.0, 1.0, 0.0),
        )

    def test_one_unsupported_proposition_withholds_the_claim(self) -> None:
        """The blocker. 0.983 on the sentence is not support for the
        half the quote never mentions."""
        v = verify_claim(self.BUNDLED, [("E1", self.QUOTE)], self.scorer(), support_threshold=0.98)
        assert v.publishable is False
        assert "asserts 2 things" in v.reason

    def test_the_supported_half_publishes_on_its_own(self) -> None:
        """It reaches print only as its own claim, never by deleting
        the other half from a bundled one."""
        v = verify_claim(
            f"{self.SUPPORTED}.",
            [("E1", self.QUOTE)],
            FakeScorer({(self.QUOTE, f"{self.SUPPORTED}."): (0.991, 0.009, 0.0)}),
            support_threshold=0.98,
        )
        assert v.publishable is True

    def test_without_the_check_the_bundle_publishes(self) -> None:
        """The defect, kept as a test so the fix cannot be quietly
        reverted."""
        v = verify_claim(
            self.BUNDLED,
            [("E1", self.QUOTE)],
            self.scorer(),
            support_threshold=0.98,
            check_propositions=False,
        )
        assert v.publishable is True

    def test_a_single_assertion_is_unaffected(self) -> None:
        quote = "Panel GMM models reveal that accumulated technical debt reduces velocity."
        claim = "Accumulated technical debt reduces future velocity."
        v = verify_claim(
            claim,
            [("E1", quote)],
            FakeScorer({(quote, claim): (0.99, 0.01, 0.0)}),
            support_threshold=0.98,
        )
        assert v.publishable is True


class TestEdges:
    def test_empty_text_has_no_propositions(self) -> None:
        assert decompose("") == []

    def test_whitespace_only_has_no_propositions(self) -> None:
        assert decompose("   \n ") == []

    @pytest.mark.parametrize(
        "text",
        [
            "A.",
            "It works.",
            "The study found a 19% increase.",
        ],
    )
    def test_short_sentences_stay_whole(self, text: str) -> None:
        assert count(text) == 1
