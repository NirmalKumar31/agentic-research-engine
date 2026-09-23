"""Run metrics and the evaluation harness."""

from __future__ import annotations

from agentic_research.config import ModelRole, Provider
from agentic_research.evaluation.evaluators import evaluate_run
from agentic_research.llm.base import (
    AttemptKind,
    LLMCallRecord,
    ProviderAttempt,
    UsageTracker,
)
from agentic_research.metrics import build_metrics
from agentic_research.models import (
    CitationIssue,
    CitationIssueType,
    CitationVerification,
    DiscoveryRef,
    EvidenceItem,
    QuoteMatch,
    SourceDocument,
    SubQuestion,
)
from agentic_research.retrieval.fetcher import FetchStats
from agentic_research.search.service import SearchStats


def source(sid: str, domain: str, text: str = "body text for the source") -> SourceDocument:
    return SourceDocument(
        id=sid,
        url=f"https://{domain}/{sid}",
        canonical_url=f"https://{domain}/{sid}",
        title=f"T{sid}",
        domain=domain,
        text=text,
        content_hash=SourceDocument.hash_text(text + sid),
        word_count=200,
    )


def evidence(eid: str, sid: str, sqid: str, verified: bool = True) -> EvidenceItem:
    return EvidenceItem(
        id=eid,
        source_id=sid,
        sub_question_id=sqid,
        claim="c",
        quote="q",
        relevance=0.8,
        quote_match=QuoteMatch.EXACT_NORMALIZED if verified else QuoteMatch.FUZZY,
        discovery=DiscoveryRef(query_id="Q1", sub_question_id=sqid),
    )


def sample_state() -> dict:
    verification = CitationVerification(
        total_claims=6,
        substantive_claims=4,
        total_citations=5,
        resolvable_citations=5,
        total_evidence_refs=6,
        resolvable_evidence_refs=5,
        checkable_claims=4,
        checked_claims=4,
        supported_claims=3,
        partially_supported_claims=1,
        unsupported_claims=0,
        entailment_exhaustive=True,
        contradictions_total=2,
        contradictions_auditable=1,
        unused_source_ids=["S3"],
        issues=[CitationIssue(type=CitationIssueType.UNCITED_CLAIM, severity="warning")],
    )
    return {
        "sub_questions": [
            SubQuestion(id="SQ1", text="a", rationale="r"),
            SubQuestion(id="SQ2", text="b", rationale="r"),
        ],
        "sources": [source("S1", "a.com"), source("S2", "b.com"), source("S3", "a.com")],
        "evidence": [
            evidence("S1-e1", "S1", "SQ1"),
            evidence("S2-e1", "S2", "SQ1"),
            evidence("S1-e2", "S1", "SQ2", verified=False),
        ],
        "completed_queries": [],
        "round_number": 2,
        "counters": {"search_results": 20, "fetches_avoided": 12, "duplicate_urls": 12},
        "stage_timings": [{"stage": "search", "seconds": 1.5}, {"stage": "search", "seconds": 2.0}],
        "errors": [],
        "verification": verification.model_dump(mode="json"),
        "stop_reason": "coverage sufficient",
    }


class TestBuildMetrics:
    def test_aggregates_state_and_service_counters(self) -> None:
        tracker = UsageTracker(max_calls=10)
        tracker.record(
            LLMCallRecord(ModelRole.PLANNER, Provider.OPENAI, "gpt-6-sol", "S", 1.0, 1000, 500)
        )
        metrics = build_metrics(
            run_id="r1",
            query="q",
            mode="hybrid",
            model_assignments={},
            duration_s=12.5,
            state=sample_state(),
            usage=tracker,
            search_stats=SearchStats(calls=6, credits=6.0),
            fetch_stats=FetchStats(attempted=5, succeeded=4),
        )
        assert metrics.research_rounds == 2
        assert metrics.unique_sources == 3
        assert metrics.evidence_items == 3
        assert metrics.exact_quotes == 2
        assert metrics.fuzzy_quotes == 1
        assert metrics.citable_evidence == 2
        assert metrics.distinct_domains == 2
        assert metrics.fetches_avoided == 12
        assert metrics.llm_calls == 1

    def test_stage_timings_are_summed_per_stage(self) -> None:
        metrics = build_metrics(
            run_id="r1",
            query="q",
            mode="local",
            model_assignments={},
            duration_s=1.0,
            state=sample_state(),
            usage=UsageTracker(10),
            search_stats=SearchStats(),
            fetch_stats=FetchStats(),
        )
        assert metrics.stage_seconds["search"] == 3.5

    def test_citation_rates_are_derived_from_verification(self) -> None:
        metrics = build_metrics(
            run_id="r1",
            query="q",
            mode="local",
            model_assignments={},
            duration_s=1.0,
            state=sample_state(),
            usage=UsageTracker(10),
            search_stats=SearchStats(),
            fetch_stats=FetchStats(),
        )
        assert metrics.citation_integrity_rate == 1.0
        # One of six evidence references the model made did not resolve.
        assert metrics.evidence_integrity_rate == round(5 / 6, 4)
        assert metrics.citation_coverage_rate == 0.75
        # Partial support is excluded from the supported numerator.
        assert metrics.claim_support_rate == 0.75
        assert metrics.partial_support_rate == 0.25
        assert metrics.support_breakdown == {
            "supported": 3,
            "partially_supported": 1,
            "unsupported": 0,
            "not_checked": 0,
        }
        assert metrics.entailment_exhaustive is True
        assert metrics.contradictions_auditable == 1

    def test_unknown_pricing_marks_cost_incomplete(self) -> None:
        tracker = UsageTracker(max_calls=10)
        tracker.record(
            LLMCallRecord(ModelRole.CRITIC, Provider.OPENAI, "gpt-mystery", "S", 1.0, 10, 10)
        )
        tracker.record_attempt(
            ProviderAttempt(
                ModelRole.CRITIC,
                Provider.OPENAI,
                "gpt-mystery",
                "S",
                AttemptKind.INITIAL,
                1.0,
                10,
                10,
            )
        )
        metrics = build_metrics(
            run_id="r1",
            query="q",
            mode="cloud",
            model_assignments={},
            duration_s=1.0,
            state=sample_state(),
            usage=tracker,
            search_stats=SearchStats(),
            fetch_stats=FetchStats(),
        )
        assert not metrics.cost_is_complete
        assert "unknown price" in metrics.cost_display

    def test_handles_an_empty_run_without_dividing_by_zero(self) -> None:
        metrics = build_metrics(
            run_id="r1",
            query="q",
            mode="local",
            model_assignments={},
            duration_s=0.1,
            state={},
            usage=UsageTracker(10),
            search_stats=SearchStats(),
            fetch_stats=FetchStats(),
        )
        assert metrics.unique_sources == 0
        assert metrics.quote_fidelity_rate == 0.0


class TestEvaluators:
    def test_evaluates_a_completed_run(self) -> None:
        state = sample_state()
        metrics = {"search_results": 20, "fetches_avoided": 12}
        results = {m.name: m for m in evaluate_run(state, metrics)}

        assert results["citation_integrity"].value == 1.0
        assert results["evidence_integrity"].value == round(5 / 6, 4)
        assert results["quote_fidelity"].value == round(2 / 3, 4)
        assert results["quote_drift"].value == round(1 / 3, 4)
        # Exhaustive, so it keeps the plain name.
        assert results["claim_support"].value == 0.75
        assert results["partial_support"].value == 0.25
        assert results["contradiction_auditability"].value == 0.5
        assert results["duplicate_avoidance"].value == 0.6
        # Two domains across three sources; the largest holds two of them.
        assert results["source_diversity"].value == round(1 - 2 / 3, 4)
        assert results["unused_source_rate"].value == round(1 / 3, 4)

    def test_evidence_coverage_uses_the_same_rule_as_routing(self) -> None:
        """SQ1 has two verified items from two sources; SQ2 has one unverified."""
        results = {m.name: m for m in evaluate_run(sample_state(), {})}
        assert results["evidence_coverage"].value == 0.5

    def test_missing_data_yields_none_rather_than_a_fake_zero(self) -> None:
        results = {m.name: m for m in evaluate_run({}, {})}
        assert results["quote_fidelity"].value is None
        assert results["source_diversity"].value is None
        assert results["duplicate_avoidance"].value is None
        assert all(m.format() == "n/a" for m in evaluate_run({}, {}) if m.value is None)
