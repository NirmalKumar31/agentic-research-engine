"""Evaluation metrics.

Every metric here is computed from a completed run's own artifacts. None of
them need a human-labelled gold answer, which is deliberate: gold answers for
open research questions are expensive, quickly stale, and mostly measure
whether the model agrees with whoever wrote them.

What these measure instead is whether the system did what it claims to do —
cite real sources, support its claims, read diverse material, avoid redundant
work. A report can score well here and still be wrong about the world; that
limitation is stated in the docs rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_research.evidence.quality import domain_concentration
from agentic_research.evidence.store import EvidenceStore
from agentic_research.models import CitationVerification, EvidenceItem, SourceDocument


@dataclass
class Metric:
    name: str
    value: float | None
    detail: str
    higher_is_better: bool = True

    def format(self) -> str:
        if self.value is None:
            return "n/a"
        return f"{self.value:.1%}" if 0.0 <= self.value <= 1.0 else f"{self.value:.2f}"


def citation_validity(verification: CitationVerification) -> Metric:
    """Share of citation markers resolving to a source actually retrieved.

    The one metric that should be 100%. Anything less means the report points
    at a source that does not exist in the run, which is the failure mode the
    citation subsystem exists to eliminate.
    """
    return Metric(
        "citation_validity",
        verification.citation_validity_rate,
        f"{verification.valid_citations}/{verification.total_citations} citations resolve",
    )


def citation_coverage(verification: CitationVerification) -> Metric:
    """Share of factual claims carrying at least one citation.

    Not expected to reach 100%: the factual-claim detector is a heuristic and
    will flag some framing sentences.
    """
    uncited = sum(1 for i in verification.issues if i.type.value == "uncited_claim")
    return Metric(
        "citation_coverage",
        verification.citation_coverage_rate,
        f"{verification.factual_claims - uncited}/{verification.factual_claims} "
        "factual claims cited",
    )


def claim_support(verification: CitationVerification) -> Metric:
    """Share of sampled claims judged entailed by their cited evidence."""
    if verification.checked_claims == 0:
        return Metric("claim_support", None, "no claims were entailment-checked")
    return Metric(
        "claim_support",
        verification.support_rate,
        f"{verification.supported_claims}/{verification.checked_claims} claims supported",
    )


def quote_fidelity(evidence: list[EvidenceItem]) -> Metric:
    """Share of extracted quotes actually found in their source text.

    Measures extraction honesty directly, and needs no model to judge it.
    """
    if not evidence:
        return Metric("quote_fidelity", None, "no evidence extracted")
    verified = sum(1 for e in evidence if e.quote_verified)
    return Metric(
        "quote_fidelity",
        round(verified / len(evidence), 4),
        f"{verified}/{len(evidence)} quotes located in their source",
    )


def source_diversity(sources: list[SourceDocument]) -> Metric:
    """1 - the share of sources from the single most common domain.

    A report drawing every source from one publisher has not corroborated
    anything, however confident it sounds.
    """
    usable = [s for s in sources if s.is_usable and not s.duplicate_of]
    if not usable:
        return Metric("source_diversity", None, "no usable sources")
    domains = [s.domain for s in usable if s.domain]
    concentration = domain_concentration(domains)
    return Metric(
        "source_diversity",
        round(1.0 - concentration, 4),
        f"{len(set(domains))} distinct domains across {len(usable)} sources; "
        f"largest holds {concentration:.0%}",
    )


def evidence_coverage(state: dict[str, Any]) -> Metric:
    """Share of sub-questions that ended with adequate evidence.

    Adequate means at least two verified evidence items from at least two
    distinct sources — the same definition the routing logic uses, so the
    metric and the behaviour cannot drift apart.
    """
    sub_questions = state.get("sub_questions", []) or []
    if not sub_questions:
        return Metric("evidence_coverage", None, "no sub-questions")
    store = EvidenceStore(state.get("sources", []) or [], state.get("evidence", []) or [])
    covered = sum(1 for q in sub_questions if store.coverage_for(q).verdict == "covered")
    return Metric(
        "evidence_coverage",
        round(covered / len(sub_questions), 4),
        f"{covered}/{len(sub_questions)} sub-questions covered",
    )


def duplicate_avoidance(metrics: dict[str, Any]) -> Metric:
    """Share of raw search results that deduplication saved us from fetching.

    Higher is better here: it is a measure of redundant work avoided, not of
    redundancy present.
    """
    total = int(metrics.get("search_results", 0) or 0)
    if total == 0:
        return Metric("duplicate_avoidance", None, "no search results")
    avoided = int(metrics.get("fetches_avoided", 0) or 0)
    return Metric(
        "duplicate_avoidance",
        round(avoided / total, 4),
        f"{avoided}/{total} results were duplicates of a page already queued",
    )


def unused_source_rate(verification: CitationVerification, sources: list[SourceDocument]) -> Metric:
    """Share of retrieved sources the report never cited.

    Retrieval that nothing uses is wasted spend. Lower is better.
    """
    usable = [s for s in sources if s.is_usable and not s.duplicate_of]
    if not usable:
        return Metric("unused_source_rate", None, "no usable sources", higher_is_better=False)
    unused = len(verification.unused_source_ids)
    return Metric(
        "unused_source_rate",
        round(unused / len(usable), 4),
        f"{unused}/{len(usable)} retrieved sources were never cited",
        higher_is_better=False,
    )


def evaluate_run(state: dict[str, Any], metrics: dict[str, Any]) -> list[Metric]:
    """Run every evaluator against one completed run."""
    raw_verification = state.get("verification") or {}
    verification = (
        CitationVerification.model_validate(raw_verification)
        if raw_verification
        else CitationVerification()
    )
    sources = state.get("sources", []) or []
    evidence = state.get("evidence", []) or []

    return [
        citation_validity(verification),
        citation_coverage(verification),
        claim_support(verification),
        quote_fidelity(evidence),
        evidence_coverage(state),
        source_diversity(sources),
        duplicate_avoidance(metrics),
        unused_source_rate(verification, sources),
    ]
