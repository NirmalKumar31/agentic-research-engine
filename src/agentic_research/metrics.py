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
from agentic_research.models import EvidenceItem, SourceDocument
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
    verified_quotes: int = 0
    unverified_quotes: int = 0
    domain_concentration: float = 0.0
    distinct_domains: int = 0

    # Models
    llm_calls: int = 0
    llm_failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    calls_by_provider: dict[str, int] = Field(default_factory=dict)
    calls_by_role: dict[str, int] = Field(default_factory=dict)
    known_cost_usd: float = 0.0
    cost_is_complete: bool = True
    unpriced_calls: int = 0

    # Citations
    citation_validity_rate: float | None = None
    citation_coverage_rate: float | None = None
    citation_support_rate: float | None = None
    citations_total: int = 0
    unused_sources: int = 0
    citations_repaired: bool = False

    # Timing and failures
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    errors: int = 0
    error_kinds: dict[str, int] = Field(default_factory=dict)

    @property
    def quote_verification_rate(self) -> float:
        total = self.verified_quotes + self.unverified_quotes
        return round(self.verified_quotes / total, 4) if total else 0.0

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
        verified_quotes=sum(1 for e in evidence if e.quote_verified),
        unverified_quotes=sum(1 for e in evidence if not e.quote_verified),
        domain_concentration=domain_concentration(domains),
        distinct_domains=len(set(domains)),
        llm_calls=totals.calls,
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
        metrics.citations_total = int(verification.get("total_citations", 0))
        metrics.unused_sources = len(verification.get("unused_source_ids", []) or [])
        metrics.citations_repaired = bool(verification.get("repaired", False))
        total_citations = metrics.citations_total
        valid = int(verification.get("valid_citations", 0))
        metrics.citation_validity_rate = (
            round(valid / total_citations, 4) if total_citations else 1.0
        )
        factual = int(verification.get("factual_claims", 0))
        uncited = sum(
            1
            for issue in verification.get("issues", []) or []
            if issue.get("type") == "uncited_claim"
        )
        metrics.citation_coverage_rate = (
            round(max(0, factual - uncited) / factual, 4) if factual else 1.0
        )
        checked = int(verification.get("checked_claims", 0))
        supported = int(verification.get("supported_claims", 0))
        metrics.citation_support_rate = round(supported / checked, 4) if checked else None

    return metrics
