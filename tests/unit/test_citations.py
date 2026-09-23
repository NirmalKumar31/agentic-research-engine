"""Evidence-first claim provenance: resolution, validation and auditability.

The property under test throughout is that a claim in the final report can be
traced to the exact evidence item behind it, and that anything which cannot be
so traced is removed and reported rather than displayed as if it were sound.
"""

from __future__ import annotations

import pytest

from agentic_research.citations.verifier import (
    extract_markers,
    resolve_claim,
    resolve_contradiction,
    resolve_report,
    strip_markers,
    verify_structure,
)
from agentic_research.evidence.store import EvidenceStore
from agentic_research.models import (
    CitationIssueType,
    Claim,
    ClaimKind,
    Contradiction,
    DiscoveryRef,
    EvidenceItem,
    QuoteMatch,
    ReportSection,
    ResearchReport,
    SourceDocument,
)


def source(sid: str, domain: str = "a.com") -> SourceDocument:
    return SourceDocument(
        id=sid,
        url=f"https://{domain}/{sid}",
        canonical_url=f"https://{domain}/{sid}",
        title=f"Title {sid}",
        domain=domain,
        text="source body text",
        content_hash=f"h{sid}",
    )


def evidence(
    eid: str,
    sid: str,
    sqid: str = "SQ1",
    *,
    match: QuoteMatch = QuoteMatch.EXACT_NORMALIZED,
    page: int | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        id=eid,
        source_id=sid,
        sub_question_id=sqid,
        claim=f"finding {eid}",
        quote=f"quoted span for {eid}",
        quote_match=match,
        relevance=0.8,
        page=page,
        discovery=DiscoveryRef(query_id="Q1", sub_question_id=sqid),
    )


@pytest.fixture
def store() -> EvidenceStore:
    return EvidenceStore(
        [source("S1"), source("S2", "b.com"), source("S3", "c.com")],
        [
            evidence("S1-e1", "S1"),
            evidence("S1-e2", "S1"),
            evidence("S2-e1", "S2", "SQ2"),
            evidence("S3-e1", "S3", match=QuoteMatch.FUZZY),
        ],
    )


class TestCitationsAreDerivedNotDeclared:
    """The engine resolves evidence -> source. A model never names a source."""

    def test_citation_ids_are_derived_from_evidence_ids(self, store: EvidenceStore) -> None:
        claim = Claim(text="Latency fell", evidence_ids=["S1-e1", "S2-e1"])
        resolved, issues = resolve_claim(claim, store)
        assert resolved.citation_ids == ["S1", "S2"]
        assert issues == []

    def test_two_evidence_items_from_one_source_cite_it_once(self, store: EvidenceStore) -> None:
        claim = Claim(text="Latency fell", evidence_ids=["S1-e1", "S1-e2"])
        resolved, _ = resolve_claim(claim, store)
        assert resolved.citation_ids == ["S1"]
        assert resolved.evidence_ids == ["S1-e1", "S1-e2"]

    def test_model_supplied_citation_ids_are_ignored(self, store: EvidenceStore) -> None:
        """Even if a model fills citation_ids, the engine overwrites them."""
        claim = Claim(text="x", evidence_ids=["S1-e1"], citation_ids=["S9", "S8"])
        resolved, _ = resolve_claim(claim, store)
        assert resolved.citation_ids == ["S1"]

    def test_duplicate_evidence_ids_collapse(self, store: EvidenceStore) -> None:
        claim = Claim(text="x", evidence_ids=["S1-e1", "S1-e1"])
        resolved, _ = resolve_claim(claim, store)
        assert resolved.evidence_ids == ["S1-e1"]


class TestNonexistentEvidenceIsRejected:
    def test_unknown_evidence_id_is_an_error_and_is_dropped(self, store: EvidenceStore) -> None:
        claim = Claim(text="Costs rose 40 percent", evidence_ids=["S9-e9"])
        resolved, issues = resolve_claim(claim, store)

        assert resolved.evidence_ids == []
        assert resolved.citation_ids == []
        assert [i.type for i in issues] == [CitationIssueType.UNKNOWN_EVIDENCE]
        assert issues[0].severity == "error"

    def test_the_sentence_survives_when_its_reference_does_not(self, store: EvidenceStore) -> None:
        """Dropping the reference is conservative; dropping the sentence would
        discard a claim that may be true and merely mis-referenced."""
        claim = Claim(text="Costs rose 40 percent", evidence_ids=["S9-e9"])
        resolved, _ = resolve_claim(claim, store)
        assert "Costs rose 40 percent" in resolved.text

    def test_valid_references_survive_alongside_an_invalid_one(self, store: EvidenceStore) -> None:
        claim = Claim(text="x", evidence_ids=["S1-e1", "S9-e9"])
        resolved, issues = resolve_claim(claim, store)
        assert resolved.evidence_ids == ["S1-e1"]
        assert resolved.citation_ids == ["S1"]
        assert len(issues) == 1


class TestUncitableEvidenceCannotGroundAClaim:
    """A claim must never rest on a quote that was not found in the source."""

    def test_fuzzy_quote_evidence_is_rejected(self, store: EvidenceStore) -> None:
        claim = Claim(text="Throughput doubled", evidence_ids=["S3-e1"])
        resolved, issues = resolve_claim(claim, store)

        assert resolved.evidence_ids == []
        assert issues[0].type is CitationIssueType.UNCITABLE_EVIDENCE
        assert issues[0].severity == "error"
        assert "fuzzy" in issues[0].detail

    def test_unmatched_quote_evidence_is_rejected(self) -> None:
        store = EvidenceStore([source("S1")], [evidence("S1-e1", "S1", match=QuoteMatch.NONE)])
        _, issues = resolve_claim(Claim(text="x", evidence_ids=["S1-e1"]), store)
        assert issues[0].type is CitationIssueType.UNCITABLE_EVIDENCE


class TestLegacyMarkersCannotMasquerade:
    """Bracket text a model leaves in prose never went through resolution."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Latency fell [S3].", ["S3"]),
            ("Both agree [S1][S4].", ["S1", "S4"]),
            ("Comma form [S2, S7].", ["S2", "S7"]),
            ("Evidence form [S1-e2].", ["S1-e2"]),
            ("No citation here.", []),
        ],
    )
    def test_extract_markers(self, text: str, expected: list[str]) -> None:
        assert extract_markers(text) == expected

    def test_markers_are_stripped_from_rendered_text(self, store: EvidenceStore) -> None:
        claim = Claim(text="Latency fell sharply [S3].", evidence_ids=["S1-e1"])
        resolved, _ = resolve_claim(claim, store)
        assert "[S3]" not in resolved.text
        assert resolved.citation_ids == ["S1"]

    def test_strip_markers_tidies_spacing(self) -> None:
        assert strip_markers("Latency fell [S3] sharply.") == "Latency fell sharply."


class TestClaimKindReplacesTheInterpretationLoophole:
    def test_framing_needs_no_evidence(self, store: EvidenceStore) -> None:
        report = ResearchReport(
            title="T",
            key_findings=[Claim(text="This section compares them.", kind=ClaimKind.FRAMING)],
        )
        resolved, issues = resolve_report(report, store)
        result = verify_structure(resolved, store, resolution_issues=issues)
        assert result.substantive_claims == 0
        assert not any(i.type is CitationIssueType.UNCITED_CLAIM for i in result.issues)

    def test_synthesis_still_owes_evidence(self, store: EvidenceStore) -> None:
        """The old is_interpretation flag exempted exactly this case."""
        report = ResearchReport(
            title="T",
            key_findings=[
                Claim(text="Taken together these favour approach B.", kind=ClaimKind.SYNTHESIS)
            ],
        )
        resolved, issues = resolve_report(report, store)
        result = verify_structure(resolved, store, resolution_issues=issues)
        assert result.substantive_claims == 1
        assert any(i.type is CitationIssueType.UNCITED_CLAIM for i in result.issues)

    def test_synthesis_with_multiple_evidence_items_is_accepted(self, store: EvidenceStore) -> None:
        claim = Claim(
            text="Across both, cost dominates.",
            evidence_ids=["S1-e1", "S2-e1"],
            kind=ClaimKind.SYNTHESIS,
        )
        resolved, issues = resolve_claim(claim, store)
        assert issues == []
        assert resolved.citation_ids == ["S1", "S2"]


class TestSummaryIsVerified:
    def test_summary_claims_are_included_in_verification(self, store: EvidenceStore) -> None:
        """A prose executive_summary bypassed coverage and entailment entirely."""
        report = ResearchReport(
            title="T",
            summary_claims=[Claim(text="Uncited summary assertion", evidence_ids=[])],
        )
        resolved, issues = resolve_report(report, store)
        result = verify_structure(resolved, store, resolution_issues=issues)

        assert result.substantive_claims == 1
        assert any(i.type is CitationIssueType.UNCITED_CLAIM for i in result.issues)

    def test_summary_renders_as_prose(self, store: EvidenceStore) -> None:
        report = ResearchReport(
            title="T",
            summary_claims=[
                Claim(text="First point.", evidence_ids=["S1-e1"]),
                Claim(text="Second point.", evidence_ids=["S2-e1"]),
            ],
        )
        assert report.executive_summary == "First point. Second point."


class TestContradictionsAreAuditable:
    def test_both_sides_resolve_to_citations(self, store: EvidenceStore) -> None:
        contradiction = Contradiction(
            topic="cost",
            left_summary="cheap",
            left_evidence_ids=["S1-e1"],
            right_summary="expensive",
            right_evidence_ids=["S2-e1"],
        )
        resolved, issues = resolve_contradiction(contradiction, store)
        assert resolved.left_citation_ids == ["S1"]
        assert resolved.right_citation_ids == ["S2"]
        assert resolved.is_auditable
        assert issues == []

    def test_one_sided_contradiction_is_flagged(self, store: EvidenceStore) -> None:
        contradiction = Contradiction(
            topic="cost",
            left_summary="cheap",
            left_evidence_ids=["S1-e1"],
            right_summary="expensive",
            right_evidence_ids=["S9-e9"],
        )
        resolved, issues = resolve_contradiction(contradiction, store)
        assert not resolved.is_auditable
        assert any(i.type is CitationIssueType.UNAUDITABLE_CONTRADICTION for i in issues)

    def test_bracket_text_inside_a_contradiction_cannot_bypass_verification(
        self, store: EvidenceStore
    ) -> None:
        contradiction = Contradiction(
            topic="cost",
            left_summary="cheap [S9]",
            left_evidence_ids=["S1-e1"],
            right_summary="expensive [S8]",
            right_evidence_ids=["S2-e1"],
        )
        resolved, _ = resolve_contradiction(contradiction, store)
        assert "[S9]" not in resolved.left_summary
        assert "[S8]" not in resolved.right_summary
        assert resolved.left_citation_ids == ["S1"]


class TestEndToEndAudit:
    def test_a_claim_traces_to_quote_source_query_and_sub_question(
        self, store: EvidenceStore
    ) -> None:
        """The full chain the README claims. Walked here, not asserted."""
        claim, _ = resolve_claim(Claim(text="x", evidence_ids=["S2-e1"]), store)

        item = store.evidence_by_id(claim.evidence_ids[0])
        assert item is not None
        assert item.quote
        assert item.quote_verified

        src = store.source(item.source_id)
        assert src is not None and src.id in claim.citation_ids

        assert item.discovery is not None
        assert item.discovery.query_id == "Q1"
        assert item.discovery.sub_question_id == "SQ2" == item.sub_question_id

    def test_unused_sources_are_reported(self, store: EvidenceStore) -> None:
        report = ResearchReport(title="T", key_findings=[Claim(text="x", evidence_ids=["S1-e1"])])
        resolved, issues = resolve_report(report, store)
        result = verify_structure(resolved, store, resolution_issues=issues)
        assert set(result.unused_source_ids) == {"S2", "S3"}
        assert not result.has_errors

    def test_integrity_denominator_includes_dropped_references(self, store: EvidenceStore) -> None:
        """Integrity must not read 100% purely because bad refs were removed."""
        report = ResearchReport(
            title="T",
            key_findings=[Claim(text="x", evidence_ids=["S1-e1", "S9-e9"])],
        )
        resolved, issues = resolve_report(report, store)
        result = verify_structure(resolved, store, resolution_issues=issues)
        assert result.total_evidence_refs == 2
        assert result.resolvable_evidence_refs == 1
        assert result.evidence_integrity_rate == 0.5

    def test_sections_are_verified_too(self, store: EvidenceStore) -> None:
        report = ResearchReport(
            title="T",
            sections=[ReportSection(heading="H", claims=[Claim(text="x", evidence_ids=["S9-e9"])])],
        )
        resolved, issues = resolve_report(report, store)
        result = verify_structure(resolved, store, resolution_issues=issues)
        assert result.has_errors

    def test_empty_report_is_vacuously_clean(self, store: EvidenceStore) -> None:
        resolved, issues = resolve_report(ResearchReport(title="T"), store)
        result = verify_structure(resolved, store, resolution_issues=issues)
        assert result.citation_integrity_rate == 1.0
        assert result.evidence_integrity_rate == 1.0
        assert not result.has_errors
