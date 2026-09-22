"""Synthesis, citation verification and finalisation."""

from __future__ import annotations

from agentic_research.citations.verifier import (
    repair_report,
    strip_markers,
    verify_structure,
)
from agentic_research.config import ModelRole
from agentic_research.evidence.store import EvidenceStore
from agentic_research.graph.nodes.common import ctx, emit, error_from, stage
from agentic_research.graph.prompts import (
    SYNTHESIZER_SYSTEM,
    VERIFIER_SYSTEM,
    synthesizer_user,
    verifier_user,
)
from agentic_research.graph.state import ResearchState
from agentic_research.llm.base import LLMError
from agentic_research.models import (
    CitationIssue,
    CitationIssueType,
    Claim,
    ReportSection,
    ResearchReport,
)
from agentic_research.observability import get_logger
from agentic_research.schemas import EntailmentOut, ReportOut

log = get_logger(__name__)

# Entailment checking costs one model call per claim, so it is sampled rather
# than exhaustive. Claims are prioritised by how much a wrong one would matter.
_MAX_ENTAILMENT_CHECKS = 10


async def synthesize_report(state: ResearchState) -> ResearchState:
    """Write the report from the curated evidence package.

    The synthesiser never sees raw page text. It sees verified, attributed,
    deduplicated findings grouped by sub-question, which keeps the prompt
    affordable and makes a citable claim the path of least resistance.
    """
    sub_questions = state.get("sub_questions", [])
    sources = state.get("sources", [])
    evidence = state.get("evidence", [])
    analysis = state.get("analysis")
    question = analysis.normalized_query if analysis else state["original_query"]
    coverage = state.get("coverage")

    emit("synthesizing", evidence=len(evidence))
    store = EvidenceStore(sources, evidence)

    with stage("synthesize") as timing:
        package = store.build_package(sub_questions)
        gaps: list[str] = []
        if coverage:
            gaps = [f"no evidence for {sq_id}" for sq_id in coverage.missing[:5]]
            gaps += [f"thin evidence for {sq_id}" for sq_id in coverage.weak[:5]]
        if state.get("stop_reason"):
            gaps.append(f"research stopped early: {state['stop_reason']}")

        errors = []
        try:
            out = (
                await ctx()
                .router.get(ModelRole.SYNTHESIZER)
                .structured(
                    ReportOut,
                    SYNTHESIZER_SYSTEM,
                    synthesizer_user(
                        question,
                        analysis.output_format.value if analysis else "overview",
                        package.text or "(no evidence was gathered)",
                        "\n".join(f"- {g}" for g in gaps),
                    ),
                )
            )
            report = ResearchReport(
                title=out.title.strip() or question[:120],
                executive_summary=out.executive_summary.strip(),
                sections=[
                    ReportSection(
                        heading=section.heading,
                        claims=[_to_claim(c) for c in section.claims],
                    )
                    for section in out.sections
                ],
                key_findings=[_to_claim(c) for c in out.key_findings],
                contradictions=list(out.contradictions),
                limitations=_dedupe_limitations(list(out.limitations) + gaps),
            )
        except LLMError as exc:
            log.error("synthesis_failed", error=str(exc)[:300])
            report = _fallback_report(question, store, gaps, str(exc))
            errors = [error_from("synthesize", exc, "emitted evidence-only report")]
        timing["evidence_items"] = package.evidence_count

    log.info(
        "synthesis_completed",
        sections=len(report.sections),
        findings=len(report.key_findings),
        evidence_used=package.evidence_count,
    )
    emit("synthesized", sections=len(report.sections), findings=len(report.key_findings))
    return {"report": report, "stage_timings": [timing], "errors": errors}


def _to_claim(out: object) -> Claim:
    """Build a domain Claim from a model-produced one.

    Accepts citations from either the ``source_ids`` field or markers left in
    the prose, because some models do both and some do neither. Unknown ids
    are not filtered here; verification catches them, which keeps the
    hallucination visible in the report instead of silently swallowed.
    """
    from agentic_research.citations.verifier import extract_markers, strip_markers

    text = getattr(out, "text", "")
    ids = list(getattr(out, "source_ids", []) or [])
    for marker in extract_markers(text):
        if marker not in ids:
            ids.append(marker)
    return Claim(
        text=strip_markers(text) if ids else text.strip(),
        citation_ids=ids,
        is_interpretation=bool(getattr(out, "is_interpretation", False)),
    )


def _dedupe_limitations(items: list[str]) -> list[str]:
    """Drop near-duplicates, which appear when the model restates a gap we
    also appended from the coverage assessment."""
    seen: set[str] = set()
    kept: list[str] = []
    for item in items:
        key = " ".join(item.lower().split()).rstrip(".")
        if key and key not in seen:
            seen.add(key)
            kept.append(item.strip())
    return kept


def _fallback_report(
    question: str, store: EvidenceStore, gaps: list[str], error: str
) -> ResearchReport:
    """Emit the evidence directly when synthesis fails.

    A run that gathered forty verified findings should not return nothing
    because the final call failed. The evidence is the expensive part; listing
    it unsynthesised is degraded but genuinely useful.
    """
    findings = [
        Claim(text=item.claim, citation_ids=[item.source_id])
        for item in sorted(store.evidence, key=lambda e: -e.confidence)[:12]
        if item.quote_verified
    ]
    return ResearchReport(
        title=f"Evidence summary: {question[:100]}",
        executive_summary=(
            "Report synthesis failed, so this document lists the verified evidence "
            "gathered during the run without narrative synthesis."
        ),
        key_findings=findings,
        limitations=[f"synthesis step failed: {error[:200]}", *gaps],
    )


async def verify_citations(state: ResearchState) -> ResearchState:
    """Validate the report's citations, then repair what can be repaired."""
    report = state.get("report")
    sources = state.get("sources", [])
    evidence = state.get("evidence", [])
    if report is None:
        return {"verification": None}

    store = EvidenceStore(sources, evidence)
    known = {s.id for s in store.usable_sources()}
    emit("verifying_citations")

    with stage("verify_citations") as timing:
        result = verify_structure(report, known)

        # Repair before entailment: no point paying to check a claim whose
        # citation is about to be stripped.
        repaired_count = 0
        if any(i.type is CitationIssueType.UNKNOWN_SOURCE for i in result.issues):
            report, repaired_count = repair_report(report, known)
            result = verify_structure(report, known)
            result.repaired = True
            log.warning("citations_repaired", removed_markers=repaired_count)

        errors = await _check_entailment(report, store, result)
        timing["citations"] = result.total_citations

    log.info(
        "citation_verification_completed",
        citations=result.total_citations,
        validity_rate=result.citation_validity_rate,
        coverage_rate=result.citation_coverage_rate,
        support_rate=result.support_rate,
        checked=result.checked_claims,
        unused_sources=len(result.unused_source_ids),
        repaired=result.repaired,
    )
    emit(
        "citations_verified",
        total=result.total_citations,
        valid=result.valid_citations,
        validity_rate=result.citation_validity_rate,
        support_rate=result.support_rate,
        repaired=result.repaired,
    )
    return {
        "report": report,
        "verification": result.model_dump(mode="json"),
        "stage_timings": [timing],
        "errors": errors,
    }


async def _check_entailment(report: ResearchReport, store: EvidenceStore, result: object) -> list:
    """Ask the verifier model whether cited evidence supports each claim.

    Sampled, not exhaustive. Key findings go first because they are what a
    reader takes away, and a wrong headline claim matters more than a wrong
    aside in section four.
    """
    from agentic_research.models import CitationVerification

    assert isinstance(result, CitationVerification)

    candidates = [c for c in report.key_findings if c.citation_ids and not c.is_interpretation]
    candidates += [
        c
        for section in report.sections
        for c in section.claims
        if c.citation_ids and not c.is_interpretation
    ]
    candidates = candidates[:_MAX_ENTAILMENT_CHECKS]
    if not candidates:
        return []

    errors = []
    model = ctx().router.get(ModelRole.VERIFIER)
    for claim in candidates:
        block = _evidence_block(claim.citation_ids, store)
        if not block:
            continue
        try:
            out = await model.structured(
                EntailmentOut, VERIFIER_SYSTEM, verifier_user(strip_markers(claim.text), block)
            )
        except LLMError as exc:
            # Verification is a quality gate, not a blocker. Stop checking and
            # report how many claims were actually checked.
            log.warning("entailment_check_failed", error=str(exc)[:200])
            errors.append(error_from("verify_citations", exc, "entailment sampling stopped"))
            break

        result.checked_claims += 1
        if out.verdict == "supported":
            result.supported_claims += 1
        elif out.verdict == "unsupported":
            result.issues.append(
                CitationIssue(
                    type=CitationIssueType.UNSUPPORTED_CLAIM,
                    severity="warning",
                    claim_text=claim.text[:200],
                    citation_id=",".join(claim.citation_ids),
                    detail=out.reason[:200],
                )
            )
    return errors


def _evidence_block(source_ids: list[str], store: EvidenceStore) -> str:
    lines = []
    for source_id in source_ids[:4]:
        source = store.source(source_id)
        if source is None:
            continue
        items = [e for e in store.evidence if e.source_id == source_id][:3]
        for item in items:
            lines.append(f'[{source_id}] "{item.quote[:300]}"')
    return "\n".join(lines)


async def finalize(state: ResearchState) -> ResearchState:
    """Render the final markdown and record why research stopped."""
    from agentic_research.graph.routing import stop_reason_for
    from agentic_research.models import CitationVerification
    from agentic_research.report import render_markdown

    report = state.get("report")
    if report is None:
        return {"final_markdown": "# Research failed\n\nNo report was produced.\n"}

    raw_verification = state.get("verification")
    verification = (
        CitationVerification.model_validate(raw_verification) if raw_verification else None
    )
    markdown = render_markdown(report, state.get("sources", []), verification)
    reason = stop_reason_for(state)

    log.info("research_completed", stop_reason=reason, sources=len(state.get("sources", [])))
    emit("completed", stop_reason=reason)
    return {"final_markdown": markdown, "stop_reason": reason}
