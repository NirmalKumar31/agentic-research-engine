"""When every claim is withheld, show the sources' own words instead.

A report whose synthesis all failed verification is correct and
useless: a title and a source list. The evidence is still there and
still verified verbatim, so it is shown as itself. The danger is
presenting excerpts as though they were findings, so these tests hold
the line between the two.
"""

from __future__ import annotations

from agentic_research.evidence.store import EvidenceStore
from agentic_research.models import (
    CitationVerification,
    Claim,
    ClaimKind,
    EvidenceItem,
    QuoteMatch,
    ResearchReport,
    SourceDocument,
)
from agentic_research.report import render_markdown

FALLBACK_NOTICE = "No synthesized claim passed evidence verification"


def _source() -> SourceDocument:
    return SourceDocument(
        id="S1",
        url="https://example.test/a",
        canonical_url="https://example.test/a",
        title="A Technical Note",
        domain="example.test",
    )


def _evidence(quote: str, evidence_id: str = "S1-e1") -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        source_id="S1",
        sub_question_id="q1",
        claim=quote,
        quote=quote,
        quote_match=QuoteMatch.EXACT_NORMALIZED,
    )


def _render(report: ResearchReport, evidence: list[EvidenceItem]) -> str:
    return render_markdown(report, [_source()], CitationVerification(), None, evidence=evidence)


class TestFallbackAppears:
    def test_excerpts_shown_when_no_claim_survived(self) -> None:
        text = _render(ResearchReport(title="T"), [_evidence("Throughput reached 12,000 QPS.")])
        assert FALLBACK_NOTICE in text
        assert "Throughput reached 12,000 QPS." in text

    def test_the_excerpt_is_verbatim_and_attributed(self) -> None:
        quote = "As of the 2021 release, the v1 wire format was unsupported."
        text = _render(ResearchReport(title="T"), [_evidence(quote)])
        assert f'"{quote}"' in text
        assert "[S1]" in text

    def test_excerpts_are_marked_as_quotations_not_findings(self) -> None:
        text = _render(ResearchReport(title="T"), [_evidence("A quoted sentence.")])
        assert "not findings" in text
        assert "## Key findings" not in text


class TestFallbackStaysOut:
    def test_absent_when_a_claim_was_published(self) -> None:
        report = ResearchReport(
            title="T",
            key_findings=[
                Claim(
                    text="Throughput reached 12,000 QPS.",
                    kind=ClaimKind.FACTUAL,
                    evidence_ids=["S1-e1"],
                    citation_ids=["S1"],
                )
            ],
        )
        assert FALLBACK_NOTICE not in _render(report, [_evidence("Throughput reached 12,000 QPS.")])

    def test_absent_when_there_is_no_citable_evidence(self) -> None:
        """Nothing to show. The notice alone would promise excerpts that
        do not follow."""
        uncitable = EvidenceItem(
            id="S1-e1",
            source_id="S1",
            sub_question_id="q1",
            claim="unmatched text",
            quote="unmatched text",
            quote_match=QuoteMatch.NONE,
        )
        assert FALLBACK_NOTICE not in _render(ResearchReport(title="T"), [uncitable])

    def test_absent_when_verification_never_ran(self) -> None:
        """Without a verification record nothing was withheld, so an
        empty report is a failed run, not a fully-withheld one."""
        text = render_markdown(
            ResearchReport(title="T"), [_source()], None, evidence=[_evidence("A quoted sentence.")]
        )
        assert FALLBACK_NOTICE not in text


class TestExcerptsAreNotClaims:
    def test_published_claim_count_stays_zero(self) -> None:
        """The whole point. An excerpt is the source's own sentence with
        no synthesis over it; counting it as a finding would undo the
        withholding that produced it."""
        verification = CitationVerification(final_published_claims=0, evidence_only_excerpts=3)
        assert verification.final_published_claims == 0
        assert verification.evidence_only_excerpts == 3

    def test_the_two_counters_are_separate_fields(self) -> None:
        verification = CitationVerification()
        assert verification.evidence_only_excerpts == 0
        assert verification.final_published_claims == 0

    def test_excerpt_count_is_bounded(self) -> None:
        from agentic_research.report import MAX_EXCERPTS

        evidence = [_evidence(f"Sentence number {i}.", f"S1-e{i}") for i in range(30)]
        text = _render(ResearchReport(title="T"), evidence)
        shown = sum(1 for line in text.splitlines() if line.startswith('- "Sentence number'))
        assert shown == MAX_EXCERPTS

    def test_store_reports_only_citable_evidence(self) -> None:
        store = EvidenceStore([_source()], [_evidence("A quoted sentence.")])
        assert len(store.citable_evidence()) == 1
