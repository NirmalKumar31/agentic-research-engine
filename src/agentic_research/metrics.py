"""Run metrics.

Everything here is measured during the run. Nothing is estimated, and where a
number genuinely cannot be determined — token pricing for a model absent from
``pricing.toml`` — it is reported as unknown rather than filled in with a
plausible guess.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentic_research.evidence.quality import domain_concentration
from agentic_research.llm.base import UsageTracker
from agentic_research.models import (
    CitationVerification,
    EvidenceItem,
    QuoteMatch,
    SourceDocument,
)
from agentic_research.retrieval.fetcher import FetchStats
from agentic_research.search.service import SearchStats


class StageTiming(BaseModel):
    stage: str
    seconds: float
    detail: dict[str, Any] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    """The summary printed at the end of a run and written to metrics.json."""

    run_id: str
    query: str
    mode: str
    model_assignments: dict[str, str] = Field(default_factory=dict)

    duration_s: float = 0.0
    environment: dict[str, Any] = Field(default_factory=dict)
    """Python/OS/package/model versions. Latency figures are hardware
    specific and meaningless without the hardware."""
    research_rounds: int = 0
    stop_reason: str = ""

    # Search and retrieval
    search_queries: int = 0
    searches_failed: int = 0
    search_results: int = 0
    search_credits: float = 0.0
    pages_fetched: int = 0
    provider_content_reused: int = 0
    fetch_failures: int = 0
    fetch_status_breakdown: dict[str, int] = Field(default_factory=dict)

    # Deduplication
    duplicate_urls: int = 0
    already_known_urls: int = 0
    content_duplicates: int = 0
    fetches_avoided: int = 0

    # Evidence
    unique_sources: int = 0
    usable_sources: int = 0
    evidence_items: int = 0
    exact_quotes: int = 0
    fuzzy_quotes: int = 0
    unmatched_quotes: int = 0
    citable_evidence: int = 0
    cross_attributed_evidence: int = 0
    content_origins: dict[str, int] = Field(default_factory=dict)
    domain_concentration: float = 0.0
    distinct_domains: int = 0

    # Models. Logical calls and provider requests are different units: one
    # logical call can emit several requests (repair, compatibility retry),
    # and providers rate-limit and bill on requests.
    llm_calls: int = 0
    provider_requests: int = 0
    billable_provider_requests: int = 0
    failed_provider_requests: int = 0
    structured_repairs: int = 0
    compatibility_retries: int = 0
    transport_retries: int = 0
    rate_limit_refusals: int = 0
    provider_requests_by_model: dict[str, int] = Field(default_factory=dict)
    reserved_worst_case_usd: float = 0.0
    reservation_was_sufficient: bool = True
    """Recorded spend stayed within the worst case reserved before dispatch."""
    llm_failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    calls_by_provider: dict[str, int] = Field(default_factory=dict)
    calls_by_role: dict[str, int] = Field(default_factory=dict)
    known_cost_usd: float = 0.0
    cost_is_complete: bool = True
    unpriced_calls: int = 0

    # Citations and claim support. Names are literal: integrity means the
    # reference resolved, not that the source supports the claim.
    citation_integrity_rate: float | None = None
    evidence_integrity_rate: float | None = None
    citation_coverage_rate: float | None = None
    claim_support_rate: float | None = None
    partial_support_rate: float | None = None
    support_breakdown: dict[str, int] = Field(default_factory=dict)
    entailment_exhaustive: bool = False
    citations_total: int = 0
    evidence_refs_total: int = 0
    contradictions_total: int = 0
    contradictions_auditable: int = 0
    unused_sources: int = 0
    citations_repaired: bool = False

    # Timing and failures
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    errors: int = 0
    error_kinds: dict[str, int] = Field(default_factory=dict)

    @property
    def quote_fidelity_rate(self) -> float:
        """Share of extracted quotes found verbatim in their source.

        Exact-normalised matches only: whitespace and smart punctuation may
        differ, words may not. Fuzzy matches are counted separately and are
        never citable.
        """
        if not self.evidence_items:
            return 0.0
        return round(self.exact_quotes / self.evidence_items, 4)

    @property
    def fuzzy_quote_rate(self) -> float:
        if not self.evidence_items:
            return 0.0
        return round(self.fuzzy_quotes / self.evidence_items, 4)

    @property
    def quote_verification_rate(self) -> float:
        """Deprecated alias of :attr:`quote_fidelity_rate`."""
        return self.quote_fidelity_rate

    @property
    def cost_display(self) -> str:
        if not self.cost_is_complete:
            return f"${self.known_cost_usd:.4f} (+{self.unpriced_calls} calls of unknown price)"
        return f"${self.known_cost_usd:.4f}"


def build_metrics(
    *,
    run_id: str,
    query: str,
    mode: str,
    model_assignments: dict[str, str],
    duration_s: float,
    state: dict[str, Any],
    usage: UsageTracker,
    search_stats: SearchStats,
    fetch_stats: FetchStats,
    environment: dict[str, Any] | None = None,
) -> RunMetrics:
    """Assemble the run summary from state and the live service counters."""
    counters: dict[str, int] = state.get("counters", {}) or {}
    sources: list[SourceDocument] = state.get("sources", []) or []
    evidence: list[EvidenceItem] = state.get("evidence", []) or []
    usable = [s for s in sources if s.is_usable and not s.duplicate_of]
    domains = [s.domain for s in usable if s.domain]
    totals = usage.totals()

    stage_seconds: dict[str, float] = {}
    for timing in state.get("stage_timings", []) or []:
        name = str(timing.get("stage", "unknown"))
        stage_seconds[name] = round(
            stage_seconds.get(name, 0.0) + float(timing.get("seconds", 0.0)), 3
        )

    error_kinds: dict[str, int] = {}
    for err in state.get("errors", []) or []:
        kind = err.kind if hasattr(err, "kind") else str(err.get("kind", "unknown"))
        error_kinds[kind] = error_kinds.get(kind, 0) + 1

    verification = state.get("verification") or {}
    metrics = RunMetrics(
        run_id=run_id,
        query=query,
        mode=mode,
        model_assignments=model_assignments,
        duration_s=round(duration_s, 2),
        environment=environment or {},
        research_rounds=state.get("round_number", 0) or 0,
        stop_reason=state.get("stop_reason", "") or "",
        search_queries=len(state.get("completed_queries", []) or []),
        searches_failed=counters.get("searches_failed", 0),
        search_results=counters.get("search_results", 0),
        search_credits=round(search_stats.credits, 2),
        pages_fetched=fetch_stats.succeeded,
        provider_content_reused=counters.get("provider_content_reused", 0),
        fetch_failures=counters.get("fetch_failed", 0),
        fetch_status_breakdown=dict(fetch_stats.by_status),
        duplicate_urls=counters.get("duplicate_urls", 0),
        already_known_urls=counters.get("already_known", 0),
        content_duplicates=counters.get("content_duplicates", 0),
        fetches_avoided=counters.get("fetches_avoided", 0),
        unique_sources=len(sources),
        usable_sources=len(usable),
        evidence_items=len(evidence),
        exact_quotes=sum(1 for e in evidence if e.quote_match is QuoteMatch.EXACT_NORMALIZED),
        fuzzy_quotes=sum(1 for e in evidence if e.quote_match is QuoteMatch.FUZZY),
        unmatched_quotes=sum(1 for e in evidence if e.quote_match is QuoteMatch.NONE),
        citable_evidence=sum(1 for e in evidence if e.is_citable),
        cross_attributed_evidence=sum(1 for e in evidence if e.cross_attributed),
        content_origins=_count_origins(usable),
        domain_concentration=domain_concentration(domains),
        distinct_domains=len(set(domains)),
        llm_calls=totals.calls,
        provider_requests=totals.provider_requests,
        billable_provider_requests=totals.billable_provider_requests,
        failed_provider_requests=totals.failed_provider_requests,
        structured_repairs=totals.structured_repairs,
        compatibility_retries=totals.compatibility_retries,
        transport_retries=totals.transport_retries,
        rate_limit_refusals=totals.rate_limit_refusals,
        provider_requests_by_model=totals.provider_requests_by_model,
        reserved_worst_case_usd=totals.reserved_worst_case_usd,
        reservation_was_sufficient=totals.reservation_was_sufficient,
        llm_failed_calls=totals.failed_calls,
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        calls_by_provider=totals.by_provider,
        calls_by_role=totals.by_role,
        known_cost_usd=totals.known_cost_usd,
        cost_is_complete=totals.cost_is_complete,
        unpriced_calls=totals.unpriced_calls,
        stage_seconds=stage_seconds,
        errors=len(state.get("errors", []) or []),
        error_kinds=error_kinds,
    )

    if verification:
        # Rehydrate so the derived properties define the rates in one place
        # rather than being recomputed (and drifting) here.
        parsed = CitationVerification.model_validate(verification)
        metrics.citations_total = parsed.total_citations
        metrics.evidence_refs_total = parsed.total_evidence_refs
        metrics.unused_sources = len(parsed.unused_source_ids)
        metrics.citations_repaired = parsed.repaired
        metrics.citation_integrity_rate = parsed.citation_integrity_rate
        metrics.evidence_integrity_rate = parsed.evidence_integrity_rate
        metrics.citation_coverage_rate = parsed.citation_coverage_rate
        metrics.claim_support_rate = parsed.claim_support_rate if parsed.checked_claims else None
        metrics.partial_support_rate = (
            parsed.partial_support_rate if parsed.checked_claims else None
        )
        metrics.support_breakdown = parsed.support_breakdown
        metrics.entailment_exhaustive = parsed.entailment_exhaustive
        metrics.contradictions_total = parsed.contradictions_total
        metrics.contradictions_auditable = parsed.contradictions_auditable

    return metrics


def _count_origins(sources: list[SourceDocument]) -> dict[str, int]:
    """Where each source's text came from.

    Reported because quote fidelity against provider-returned content is a
    different guarantee from fidelity against an independently fetched page.
    """
    counts: dict[str, int] = {}
    for source in sources:
        key = source.content_origin.value
        counts[key] = counts.get(key, 0) + 1
    return counts
