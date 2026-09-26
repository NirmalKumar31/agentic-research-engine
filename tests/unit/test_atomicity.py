"""The compound-claim detector. Generic wording only."""

from __future__ import annotations

import pytest

from agentic_research.citations.atomicity import (
    compound_markers,
    compound_propositions,
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


class TestPropositionAtomicity:
    """Atomicity is about propositions, not punctuation.

    Counting sentences was never the real test: "X increased, Y
    decreased" is one sentence and two independently falsifiable
    assertions, and a gate that checks one quote against one claim
    cannot honestly call that atomic.

    The heuristic splits on coordinators and counts segments carrying
    their own predicate. It is wrong sometimes, and the direction is
    chosen: when a sentence cannot be shown to assert one thing it is
    treated as compound and withheld.
    """

    @pytest.mark.parametrize(
        ("case", "text"),
        [
            ("E1", "X increased, Y decreased."),
            ("E2", "X improved accuracy and reduced latency."),
            ("E3", "X improved accuracy while reducing memory."),
            ("E4", "X was faster but less accurate."),
            ("E7", "The model was trained and evaluated on the same dataset."),
            ("D", "Quantization reduces memory use, and accuracy remains high."),
            ("D", "Vector databases improve retrieval while relational databases simplify it."),
            ("D", "The system lowered latency but increased memory consumption."),
            ("D", "Accuracy was 92%, recall was 81%."),
            ("D", "X is cheaper; Y is faster."),
        ],
    )
    def test_two_assertions_are_refused(self, case: str, text: str) -> None:
        assert not is_atomic(text), f"{case}: {compound_propositions(text)}"

    @pytest.mark.parametrize(
        ("case", "text"),
        [
            ("E5", "Precision, recall, and F1 were reported."),
            ("E6", "Supports CSV and Parquet."),
            ("D", "The benchmark reports precision, recall, and F1."),
            ("D", "The study evaluated training and inference latency."),
            ("C", "The model can process text and images."),
            ("C", "The study used training, validation, and test sets."),
            ("C", "Precision, recall, and F1 are evaluation metrics."),
        ],
    )
    def test_enumeration_inside_one_assertion_survives(self, case: str, text: str) -> None:
        """One predicate shared across a list is still one claim.
        Rejecting every comma would withhold most real sentences."""
        assert is_atomic(text), f"{case}: {compound_propositions(text)}"

    @pytest.mark.parametrize(
        "text",
        [
            "Throughput rose 20%.",
            "To search 10M vectors at sub-10ms speed, you may need 64GB+ of dedicated RAM.",
            "For applications retrieving large result sets, batch efficiency varies.",
            "AUC-PR is recommended for rare fraud cases where high recall is critical.",
        ],
    )
    def test_ordinary_single_claims_survive(self, text: str) -> None:
        assert is_atomic(text), compound_propositions(text)

    def test_ed_nouns_are_not_read_as_predicates(self) -> None:
        """ "at sub-10ms speed" is not an assertion. Without this the
        -ed heuristic withholds a perfectly atomic claim."""
        assert is_atomic("At high speed, the index stays small.")
        assert is_atomic("A hundred queries per second, sustained.")

    def test_the_historical_fusion_is_refused(self) -> None:
        """E9: the transformation that produced the development set's
        only false positive must not be publishable as one claim."""
        assert not is_atomic("Quantization cut memory 75% while maintaining high recall accuracy.")

    def test_the_known_comma_spliced_published_claim_is_now_refused(self) -> None:
        """E8: a real claim from the previous audit. It passed the
        sentence-counting rule because it contains one period."""
        claim = (
            "Quantization reduces memory costs with minimal recall impact, reducing "
            "32-bit floats to 8-bit integers cuts memory 75% while maintaining high accuracy."
        )
        assert sentence_count(claim) == 1
        assert not is_atomic(claim)

    def test_reasons_are_reported_for_the_audit(self) -> None:
        reasons = compound_propositions("X was faster but less accurate.")
        assert any("contrastive" in r for r in reasons)

    def test_empty_and_whitespace_are_safe(self) -> None:
        assert is_atomic("")
        assert compound_propositions("   ") == []
