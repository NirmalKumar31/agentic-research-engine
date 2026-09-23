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
from agentic_research.models import (
    CitationVerification,
    EvidenceItem,
    QuoteMatch,
    SourceDocument,
)


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


def citation_integrity(verification: CitationVerification) -> Metric:
    """Share of citation markers resolving to a source actually retrieved.

    Note what this does *not* say: nothing about whether the source supports
    the claim. That is ``claim_support``. It was previously published as
    "citation validity", which implied the stronger guarantee.

    Since the engine now derives citations from resolved evidence, this is a
    structural invariant rather than a measurement: anything below 100%
    indicates an engine bug, not a model error. The model-facing measurement
    is ``evidence_integrity``.
    """
    return Metric(
        "citation_integrity",
        verification.citation_integrity_rate,
        f"{verification.resolvable_citations}/{verification.total_citations} citations resolve",
    )


def evidence_integrity(verification: CitationVerification) -> Metric:
    """Share of evidence references the model made that actually resolved.

    This is the honest replacement for the old headline. A model referencing
    an evidence id that does not exist, or one whose quote never aligned to
    its source, is counted here.
    """
    return Metric(
        "evidence_integrity",
        verification.evidence_integrity_rate,
        f"{verification.resolvable_evidence_refs}/{verification.total_evidence_refs} "
        "evidence references resolved to citable evidence",
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
        f"{verification.substantive_claims - uncited}/{verification.substantive_claims} "
        "evidence-owing claims cited",
    )


def claim_support(verification: CitationVerification) -> Metric:
    """Share of checked claims fully entailed by their own cited evidence.

    Named ``claim_support`` when exhaustive and ``sampled_claim_support``
    otherwise, because a sampled figure reported under the same name as an
    exhaustive one is misleading. Partial support is reported separately
    rather than folded into either bucket.
    """
    if verification.checked_claims == 0:
        return Metric("claim_support", None, "no claims were entailment-checked")
    name = "claim_support" if verification.entailment_exhaustive else "sampled_claim_support"
    breakdown = verification.support_breakdown
    return Metric(
        name,
        verification.claim_support_rate,
        f"{verification.supported_claims}/{verification.checked_claims} supported; "
        f"{breakdown['partially_supported']} partial, {breakdown['unsupported']} "
        f"unsupported, {breakdown['not_checked']} not checked",
    )


def partial_support(verification: CitationVerification) -> Metric:
    """Share of checked claims only partially entailed. Lower is better."""
    if verification.checked_claims == 0:
        return Metric("partial_support", None, "no claims checked", higher_is_better=False)
    return Metric(
        "partial_support",
        verification.partial_support_rate,
        f"{verification.partially_supported_claims}/{verification.checked_claims} partial",
        higher_is_better=False,
    )


def contradiction_auditability(verification: CitationVerification) -> Metric:
    """Share of reported disagreements with citable evidence on both sides."""
    if verification.contradictions_total == 0:
        return Metric("contradiction_auditability", None, "no contradictions reported")
    return Metric(
        "contradiction_auditability",
        round(verification.contradictions_auditable / verification.contradictions_total, 4),
        f"{verification.contradictions_auditable}/{verification.contradictions_total} "
        "auditable on both sides",
    )


def quote_fidelity(evidence: list[EvidenceItem]) -> Metric:
    """Share of extracted quotes found verbatim in their source text.

    Exact-normalised only: whitespace and smart punctuation may differ, words
    may not. Fuzzy matches were previously counted here and described as
    verbatim; they are now reported separately by ``quote_drift`` and are
    never citable.
    """
    if not evidence:
        return Metric("quote_fidelity", None, "no evidence extracted")
    exact = sum(1 for e in evidence if e.quote_match is QuoteMatch.EXACT_NORMALIZED)
    return Metric(
        "quote_fidelity",
        round(exact / len(evidence), 4),
        f"{exact}/{len(evidence)} quotes found verbatim (exact-normalised)",
    )


def quote_drift(evidence: list[EvidenceItem]) -> Metric:
    """Share of quotes that were close but reworded. Lower is better."""
    if not evidence:
        return Metric("quote_drift", None, "no evidence extracted", higher_is_better=False)
    fuzzy = sum(1 for e in evidence if e.quote_match is QuoteMatch.FUZZY)
    return Metric(
        "quote_drift",
        round(fuzzy / len(evidence), 4),
        f"{fuzzy}/{len(evidence)} quotes matched only approximately",
        higher_is_better=False,
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
        evidence_integrity(verification),
        citation_integrity(verification),
        citation_coverage(verification),
        claim_support(verification),
        partial_support(verification),
        quote_fidelity(evidence),
        quote_drift(evidence),
        evidence_coverage(state),
        source_diversity(sources),
        contradiction_auditability(verification),
        duplicate_avoidance(metrics),
        unused_source_rate(verification, sources),
    ]
