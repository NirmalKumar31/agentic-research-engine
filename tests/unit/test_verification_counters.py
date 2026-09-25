"""What each verification counter means, pinned as arithmetic.

The definitions drifted from the implementation. `generated_substantive_
claims` was documented as raw synthesiser output but assigned after
deduplication, and structural totals were recomputed only when a semantic
removal happened -- so a report that lost claims to deduplication alone
reported citation totals for a document the reader never saw.
"""

from __future__ import annotations

from agentic_research.citations.publication import (
    deduplicate_claims,
    filter_report_by_verification,
    key_of,
)
from agentic_research.models import (
    CitationVerification,
    Claim,
    ClaimKind,
    ReportSection,
    ResearchReport,
)


def claim(text: str, ids: list[str]) -> Claim:
    return Claim(text=text, evidence_ids=ids, citation_ids=[], kind=ClaimKind.FACTUAL)


class TestTheCountersAreUnambiguous:
    def test_three_identical_claims_count_as_three_generated_one_checked(self) -> None:
        """generated counts synthesiser output; dedup is recorded
        separately; verification sees one."""
        repeated = claim("The same finding.", ["S1-e1"])
        report = ResearchReport(
            title="T",
            summary_claims=[repeated],
            key_findings=[repeated],
            sections=[ReportSection(heading="A", claims=[repeated])],
        )

        generated = len(report.substantive_claims())
        assert generated == 3

        deduped, duplicates = deduplicate_claims(report)
        assert duplicates == 2
        assert len(deduped.substantive_claims()) == 1

        published, removed = filter_report_by_verification(deduped, {key_of(repeated): "supported"})
        assert removed == 0
        assert len(published.substantive_claims()) == 1

    def test_a_deduplicated_run_with_no_semantic_removal_still_recounts(self) -> None:
        """The case that used to leave stale structural totals behind."""
        repeated = claim("Stated twice.", ["S1-e1"])
        other = claim("A second finding.", ["S1-e2"])
        report = ResearchReport(
            title="T", summary_claims=[repeated, other], key_findings=[repeated]
        )

        deduped, duplicates = deduplicate_claims(report)
        assert duplicates == 1

        published, removed = filter_report_by_verification(
            deduped, {key_of(repeated): "supported", key_of(other): "supported"}
        )
        assert removed == 0
        # Two distinct claims survive, and the citation totals a reader
        # sees must describe these two.
        assert len(published.substantive_claims()) == 2

    def test_no_duplicates_leaves_every_counter_alone(self) -> None:
        a = claim("One.", ["S1-e1"])
        b = claim("Two.", ["S1-e2"])
        report = ResearchReport(title="T", summary_claims=[a, b])

        deduped, duplicates = deduplicate_claims(report)
        assert duplicates == 0
        assert deduped is report

    def test_the_counter_exists_on_the_model(self) -> None:
        assert "duplicate_claims_removed" in CitationVerification.model_fields


class TestPipelineOrder:
    def test_generated_is_recorded_before_deduplication(self) -> None:
        """Assigned after dedup, it silently reported a smaller number
        than the synthesiser produced."""
        import inspect

        from agentic_research.graph.nodes import reporting

        body = inspect.getsource(reporting.verify_citations)
        assert body.index("generated = len(") < body.index("deduplicate_claims(")

    def test_structural_metrics_are_computed_on_the_published_report(self) -> None:
        import inspect

        from agentic_research.graph.nodes import reporting

        body = inspect.getsource(reporting.verify_citations)
        assert body.index("filter_report_by_verification(") < body.rindex("verify_structure(")


class TestZeroDenominatorIsNotAScore:
    """An empty report has no reference to get wrong, so the integrity
    rate is 1.0 by construction. Rendering "100%" beside a report that
    asserts nothing reads as an observed quality result -- the README did
    exactly that for two of three recordings."""

    def test_the_invariant_still_holds_internally(self) -> None:
        empty = CitationVerification()
        assert empty.total_evidence_refs == 0
        assert empty.evidence_integrity_rate == 1.0

    def test_but_it_is_flagged_as_having_no_denominator(self) -> None:
        assert CitationVerification().has_evidence_references is False

    def test_a_real_report_reports_normally(self) -> None:
        v = CitationVerification(total_evidence_refs=4, resolvable_evidence_refs=3)
        assert v.has_evidence_references is True
        assert v.evidence_integrity_rate == 0.75

    def test_the_rendered_report_says_n_a_not_a_percentage(self) -> None:
        from agentic_research.report import _verification_section

        lines = " ".join(_verification_section(CitationVerification()))
        assert "n/a, no references" in lines
        assert "100%" not in lines

    def test_metrics_record_none_rather_than_one(self) -> None:
        """Matching how claim_support_rate already treats an unchecked
        run, so a dashboard cannot average a vacuous 1.0 into a score."""
        from agentic_research.config import Settings
        from agentic_research.llm.base import UsageTracker
        from agentic_research.metrics import build_metrics
        from agentic_research.retrieval.fetcher import FetchStats
        from agentic_research.search.service import SearchStats

        settings = Settings(llm_mode="local", tavily_api_key="tvly-test-key", _env_file=None)
        metrics = build_metrics(
            run_id="r",
            query="q",
            mode="local",
            model_assignments={},
            duration_s=1.0,
            state={"verification": CitationVerification().model_dump(mode="json")},
            usage=UsageTracker(10, settings.cloud_budget),
            search_stats=SearchStats(),
            fetch_stats=FetchStats(),
            environment={},
        )
        assert metrics.evidence_integrity_rate is None
