"""The compound-claim detector. Generic wording only."""

from __future__ import annotations

import pytest

from agentic_research.citations.atomicity import (
    compound_markers,
    is_atomic,
    looks_compound,
    sentence_count,
)


@pytest.mark.parametrize(
    "text",
    [
        "Compression reduced memory by 75% while maintaining high accuracy.",
        "The index halved storage without sacrificing recall.",
        "Latency fell, thereby improving throughput.",
        "The system supports replication as well as sharding across regions.",
        "Version 2 added streaming, which also reduced memory use.",
        "Engine A is faster, whereas Engine B uses less memory.",
        "Throughput rose, while error rates stayed flat.",
    ],
)
def test_fused_propositions_are_flagged(text: str) -> None:
    assert looks_compound(text), compound_markers(text)


@pytest.mark.parametrize(
    "text",
    [
        "Quantization reduced memory usage by 75%.",
        "The framework defines four functions: Govern, Map, Measure and Manage.",
        "The source reports minimal impact on recall after quantization.",
        "Deployments may need 64GB of dedicated RAM.",
        "The study enrolled 120 participants and ran for six weeks.",
    ],
)
def test_single_propositions_and_noun_lists_are_not_flagged(text: str) -> None:
    """A bare "and" joining noun phrases is not a compound claim.
    Flagging it would fire on almost every real sentence."""
    assert not looks_compound(text), compound_markers(text)


def test_markers_are_reported_for_the_audit() -> None:
    assert compound_markers("Memory fell 75% while maintaining accuracy, thereby cutting cost") == [
        "while maintaining",
        "thereby",
    ]


def test_empty_input_is_safe() -> None:
    assert compound_markers("") == []
    assert not looks_compound("")


class TestSentenceCounting:
    """A claim spanning two sentences cannot be verified as one.

    Every guard that reasons about "the sentence that supports this
    claim" needs the claim to be a single proposition. Given two, it
    scopes to whichever half matches more words and never examines the
    other -- which is how the release audit published a claim whose
    first half had deleted the source's own voice while the guards
    inspected its clean second half.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Throughput rose 20%.",
            "The index handles 20,000 queries per second under load.",
            "Latency stayed under 5 ms, even at high concurrency.",
            "Profiles apply the framework to a sector, e.g. credit underwriting.",
            "The U.S. deployment uses three regions.",
            "Costs fell by 40% (approx. $12,000 per month).",
        ],
    )
    def test_single_sentence_claims_are_atomic(self, text: str) -> None:
        assert is_atomic(text), sentence_count(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Throughput rose 20%. Latency fell by half.",
            "The dataset covers 2019-2020. Fraud was 0.18% of transactions.",
            "Memory fell 60%. The authors report minimal impact on recall.",
        ],
    )
    def test_multi_sentence_claims_are_not_atomic(self, text: str) -> None:
        assert not is_atomic(text)
        assert sentence_count(text) == 2

    def test_abbreviations_do_not_split_a_sentence(self) -> None:
        """Under-counting is the safe direction: it passes the claim to
        the other guards rather than withholding it on punctuation."""
        assert sentence_count("Applies to a sector, e.g. medical imaging triage.") == 1
        assert sentence_count("Vendors such as Acme Inc. ship this by default.") == 1

    def test_empty_text_counts_as_nothing(self) -> None:
        assert sentence_count("") == 0
        assert is_atomic("")
