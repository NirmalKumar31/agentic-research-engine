"""Citation extraction, validation and repair."""

from __future__ import annotations

import pytest

from agentic_research.citations.verifier import (
    extract_markers,
    looks_factual,
    repair_report,
    strip_markers,
    verify_structure,
)
from agentic_research.models import (
    CitationIssueType,
    Claim,
    ReportSection,
    ResearchReport,
)


def report_with(*claims: Claim, sections: list[ReportSection] | None = None) -> ResearchReport:
    return ResearchReport(
        title="T",
        executive_summary="s",
        key_findings=list(claims),
        sections=sections or [],
    )


class TestMarkerParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Latency fell [S3].", ["S3"]),
            ("Both agree [S1][S4].", ["S1", "S4"]),
            ("Comma form [S2, S7].", ["S2", "S7"]),
            ("Repeated [S1][S1].", ["S1"]),
            ("No citation here.", []),
        ],
    )
    def test_extract(self, text: str, expected: list[str]) -> None:
        assert extract_markers(text) == expected

    def test_strip(self) -> None:
        assert strip_markers("Latency fell [S3].") == "Latency fell ."


class TestFactualDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "Throughput improved by 40 percent under load.",
            "The study found lower latency across all runs.",
            "Version 3 requires a newer runtime than before.",
        ],
    )
    def test_assertions_are_factual(self, text: str) -> None:
        assert looks_factual(Claim(text=text))

    @pytest.mark.parametrize("text", ["This section covers the tradeoffs.", "In summary."])
    def test_framing_sentences_are_not(self, text: str) -> None:
        assert not looks_factual(Claim(text=text))

    def test_interpretation_is_never_factual(self) -> None:
        assert not looks_factual(
            Claim(text="Costs rose 40 percent overall.", is_interpretation=True)
        )


class TestStructuralVerification:
    def test_citations_from_schema_field_are_counted(self) -> None:
        """The primary path: models fill source_ids rather than writing markers."""
        report = report_with(Claim(text="Latency fell sharply", citation_ids=["S1"]))
        result = verify_structure(report, {"S1"})
        assert result.total_citations == 1
        assert result.valid_citations == 1
        assert result.citation_validity_rate == 1.0

    def test_citations_left_in_prose_are_still_picked_up(self) -> None:
        """Fallback path: some models ignore the field and inline markers."""
        report = report_with(Claim(text="Latency fell sharply [S1]."))
        result = verify_structure(report, {"S1"})
        assert result.total_citations == 1
        assert report.key_findings[0].citation_ids == ["S1"]

    def test_both_sources_merge_without_double_counting(self) -> None:
        report = report_with(Claim(text="Latency fell [S1].", citation_ids=["S1", "S2"]))
        verify_structure(report, {"S1", "S2"})
        assert report.key_findings[0].citation_ids == ["S1", "S2"]

    def test_hallucinated_source_is_an_error(self) -> None:
        """The failure this subsystem exists to prevent."""
        report = report_with(Claim(text="Costs rose 40 percent", citation_ids=["S99"]))
        result = verify_structure(report, {"S1"})
        assert result.has_errors
        assert any(i.type is CitationIssueType.UNKNOWN_SOURCE for i in result.issues)
        assert result.citation_validity_rate == 0.0

    def test_uncited_factual_claim_is_flagged(self) -> None:
        report = report_with(Claim(text="Throughput improved by 40 percent"))
        result = verify_structure(report, {"S1"})
        assert any(i.type is CitationIssueType.UNCITED_CLAIM for i in result.issues)
        assert result.citation_coverage_rate == 0.0

    def test_unused_source_is_informational_only(self) -> None:
        report = report_with(Claim(text="Latency fell", citation_ids=["S1"]))
        result = verify_structure(report, {"S1", "S2"})
        assert result.unused_source_ids == ["S2"]
        assert not result.has_errors

    def test_sections_are_verified_too(self) -> None:
        report = report_with(
            sections=[
                ReportSection(
                    heading="H",
                    claims=[Claim(text="Costs rose 40 percent", citation_ids=["S99"])],
                )
            ]
        )
        result = verify_structure(report, {"S1"})
        assert result.has_errors

    def test_empty_report_is_vacuously_valid(self) -> None:
        result = verify_structure(report_with(), set())
        assert result.citation_validity_rate == 1.0
        assert not result.has_errors


class TestRepair:
    def test_dangling_marker_is_removed_but_the_sentence_survives(self) -> None:
        """Deleting the sentence would discard a possibly-true claim; deleting
        the reference only removes something that is definitely wrong."""
        report = report_with(Claim(text="Costs rose 40 percent", citation_ids=["S99"]))
        fixed, removed = repair_report(report, {"S1"})
        assert removed == 1
        assert "Costs rose 40 percent" in fixed.key_findings[0].text
        assert fixed.key_findings[0].citation_ids == []

    def test_valid_citations_are_kept_when_removing_an_invalid_one(self) -> None:
        report = report_with(Claim(text="Costs rose", citation_ids=["S1", "S99"]))
        fixed, removed = repair_report(report, {"S1"})
        assert removed == 1
        assert fixed.key_findings[0].citation_ids == ["S1"]

    def test_repair_is_a_no_op_when_everything_resolves(self) -> None:
        report = report_with(Claim(text="Costs rose", citation_ids=["S1"]))
        fixed, removed = repair_report(report, {"S1"})
        assert removed == 0
        assert fixed.key_findings[0].citation_ids == ["S1"]

    def test_repaired_report_passes_reverification(self) -> None:
        report = report_with(Claim(text="Costs rose 40 percent", citation_ids=["S99"]))
        fixed, _ = repair_report(report, {"S1"})
        assert not verify_structure(fixed, {"S1"}).has_errors
