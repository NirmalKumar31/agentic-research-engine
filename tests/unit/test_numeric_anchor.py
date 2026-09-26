"""Numeric literals are compared deterministically, not by the model.

The verifier receives intact evidence and still mutates numbers in its
own reasoning: "5x" reported as ".5x", "1.0" as "1:0", "32-bit" as
"3:2-bit". One calibration case was rejected on a discrepancy that did
not exist in the input, and a prompt instruction to compare carefully
did not stop it.

The anchor decides nothing. It reports which of the claim's literals
appear verbatim in the evidence, so the model cannot assert a mismatch
the text does not contain.
"""

from __future__ import annotations

import pytest

from agentic_research.citations.numerics import anchor, literals


class TestExtractionKeepsTokensIntact:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("slows query performance by 5x", ["5x"]),
            ("uses 80% less memory", ["80%"]),
            ("NIST AI RMF 1.0 is organised", ["1.0"]),
            ("under 100M vectors", ["100M"]),
            ("p99 under 4ms rather than 25ms", ["4ms", "25ms"]),
            ("you may need 64GB+ of dedicated RAM", ["64GB+"]),
            ("reducing 32-bit floats to 8-bit integers", ["32-bit", "8-bit"]),
            ("a drop of 40-60% in QPS", ["40-60%"]),
            ("more than 20,000+ fields", ["20,000+"]),
        ],
    )
    def test_each_shape_survives(self, text: str, expected: list[str]) -> None:
        assert literals(text) == expected

    @pytest.mark.parametrize(
        ("original", "corruption"),
        [("5x", ".5x"), ("32-bit", "3:2-bit"), ("1.0", "1:0"), ("100M", "10:00M")],
    )
    def test_the_helper_never_produces_the_observed_corruptions(
        self, original: str, corruption: str
    ) -> None:
        """The exact mutations seen in baseline verifier reasons."""
        found = literals(f"the value is {original} exactly")
        assert original in found
        assert corruption not in found

    def test_a_multiplier_is_not_split_into_bare_digits(self) -> None:
        assert literals("5x slower") == ["5x"]

    def test_a_bit_width_is_not_split(self) -> None:
        assert "32" not in literals("32-bit floats")

    def test_spacing_variants_are_one_literal(self) -> None:
        assert len(literals("5x and 5 x")) == 1

    def test_prose_with_no_numbers_yields_nothing(self) -> None:
        assert literals("Vector databases are easier to deploy.") == []


class TestTheAnchorReportsPresence:
    def test_matching_literals_are_reported_as_found(self) -> None:
        block = anchor(
            "reduces memory by 80% but is 5x slower",
            'Quote: "which is 5x slower but uses 80% less memory"',
        )
        assert "Found verbatim in the cited evidence: 80%, 5x" in block
        assert "Not found verbatim: none" in block

    def test_a_literal_absent_from_the_evidence_is_reported_missing(self) -> None:
        block = anchor("throughput drops 75%", 'Quote: "throughput drops 40-60%"')
        assert "Not found verbatim: 75%" in block

    def test_a_claim_without_numbers_gets_no_anchor(self) -> None:
        """Nothing should invite the model to hunt for numbers that were
        never at issue."""
        assert anchor("Vector databases are easy to deploy.", 'Quote: "easy"') == ""

    def test_it_is_evidence_of_presence_not_of_support(self) -> None:
        """A matching number does not make a claim supported: the claim
        below inverts the relationship the evidence states."""
        block = anchor(
            "memory grows 80% when using IVF-PQ",
            'Quote: "uses 80% less memory"',
        )
        assert "Found verbatim in the cited evidence: 80%" in block
        # The anchor says nothing about direction, and must not.
        assert "support" not in block.lower()
