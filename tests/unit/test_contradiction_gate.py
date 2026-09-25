"""Contradiction summaries are assertions and are gated like claims.

`left_summary` and `right_summary` are model-written prose about what two
sources say. They reached the published report with no support check at
all: structural resolution proved only that evidence existed on each
side, which is what `is_auditable` reports and all it ever meant.

So a contradiction could state "Source A says X while Source B says Y"
with neither half following from the quotes cited, and the fail-closed
publication gate never looked at it.
"""

from __future__ import annotations

import pytest

from agentic_research.citations.publication import (
    claim_key,
    filter_report_by_verification,
)
from agentic_research.models import (
    CitationVerification,
    Contradiction,
    ResearchReport,
)


def contradiction() -> Contradiction:
    return Contradiction(
        topic="whether ensembles win",
        left_summary="The study reports ensembles outperformed single models.",
        left_evidence_ids=["S1-e1"],
        right_summary="The survey reports no consistent winner across datasets.",
        right_evidence_ids=["S2-e1"],
    )


def verdicts(left: str | None, right: str | None) -> dict:
    c = contradiction()
    out = {}
    if left:
        out[claim_key(c.left_summary, c.left_evidence_ids)] = left
    if right:
        out[claim_key(c.right_summary, c.right_evidence_ids)] = right
    return out


def report_with_contradiction() -> ResearchReport:
    return ResearchReport(title="T", contradictions=[contradiction()])


class TestBothSidesMustBeSupported:
    def test_published_when_both_sides_are_supported(self) -> None:
        filtered, _ = filter_report_by_verification(
            report_with_contradiction(), verdicts("supported", "supported")
        )
        assert len(filtered.contradictions) == 1

    @pytest.mark.parametrize("bad", ["partially_supported", "unsupported"])
    def test_dropped_when_the_left_side_fails(self, bad: str) -> None:
        filtered, _ = filter_report_by_verification(
            report_with_contradiction(), verdicts(bad, "supported")
        )
        assert filtered.contradictions == []

    @pytest.mark.parametrize("bad", ["partially_supported", "unsupported"])
    def test_dropped_when_the_right_side_fails(self, bad: str) -> None:
        filtered, _ = filter_report_by_verification(
            report_with_contradiction(), verdicts("supported", bad)
        )
        assert filtered.contradictions == []

    def test_dropped_when_one_side_was_never_checked(self) -> None:
        """No verdict is not the same as no objection -- the same rule
        that governs claims."""
        filtered, _ = filter_report_by_verification(
            report_with_contradiction(), verdicts("supported", None)
        )
        assert filtered.contradictions == []

    def test_dropped_when_neither_side_was_checked(self) -> None:
        filtered, _ = filter_report_by_verification(report_with_contradiction(), {})
        assert filtered.contradictions == []

    def test_the_removal_is_disclosed(self) -> None:
        filtered, _ = filter_report_by_verification(
            report_with_contradiction(), verdicts("unsupported", "supported")
        )
        assert any("disagreement" in limit for limit in filtered.limitations)


class TestAuditableStillMeansStructural:
    def test_auditability_is_not_overloaded_with_semantics(self) -> None:
        """`is_auditable` answers "is there evidence on both sides?" and
        must keep answering only that."""
        assert contradiction().is_auditable is True

        one_sided = contradiction().model_copy(update={"right_evidence_ids": []})
        assert one_sided.is_auditable is False

    def test_semantic_support_has_its_own_counters(self) -> None:
        fields = CitationVerification.model_fields
        assert "contradictions_auditable" in fields
        assert "contradiction_sides_checkable" in fields
        assert "contradiction_sides_checked" in fields
        assert "contradictions_semantically_supported" in fields

    def test_a_structurally_auditable_contradiction_can_still_be_dropped(self) -> None:
        """The whole point: evidence on both sides says nothing about
        whether either sentence follows from it."""
        c = contradiction()
        assert c.is_auditable is True

        filtered, _ = filter_report_by_verification(
            ResearchReport(title="T", contradictions=[c]),
            verdicts("partially_supported", "supported"),
        )
        assert filtered.contradictions == []
