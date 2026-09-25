"""The compound-claim detector. Generic wording only."""

from __future__ import annotations

import pytest

from agentic_research.citations.atomicity import compound_markers, looks_compound


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
