"""Synthesis, citation verification and finalisation."""

from __future__ import annotations

from agentic_research.citations.publication import (
    ClaimKey,
    ClaimVerdict,
    filter_report_by_verification,
    key_of,
)
from agentic_research.citations.verifier import (
    resolve_report,
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
    CitationVerification,
    Claim,
    ClaimKind,
    Contradiction,
    CoverageAssessment,
    ReportSection,
    ResearchReport,
    SubQuestion,
)
from agentic_research.observability import get_logger
from agentic_research.schemas import EntailmentOut, ReportOut

log = get_logger(__name__)

# Entailment costs one model call per claim. Interactive runs sample; the
# benchmark checks everything, because a sampled number reported as if it were
# exhaustive is the kind of metric this project exists not to publish.
_DEFAULT_ENTAILMENT_SAMPLE = 10

# A report is worth writing even when almost nothing can be verified, and
# a floor keeps a tiny budget from producing an empty one.
_MIN_CLAIM_BUDGET = 4

# Headroom left when sizing the report: verification also spends a call
# resolving structure, and a structured-output repair can cost another.
_VERIFICATION_OVERHEAD = 2


async def _claim_budget() -> int | None:
    """How many substantive claims this run can afford to verify.

    Every substantive claim costs one entailment call, and an unverified
    claim is not published. Generating more than the budget allows does
    not lengthen the report; it just means the surplus is deleted after
    being paid for. A live run generated 25 claims, could check 4, and
    published 2.

    Returns None when the remaining budget is ample, so an unconstrained
    local run is not told to write a short report for no reason.
    """
    remaining = await ctx().router.tracker.remaining()
    # One for synthesis itself, which has not been reserved yet.
    affordable = remaining - 1 - _VERIFICATION_OVERHEAD
    if affordable >= _DEFAULT_ENTAILMENT_SAMPLE:
        return None
    return max(_MIN_CLAIM_BUDGET, affordable)


async def synthesize_report(state: ResearchState) -> ResearchState:
    """Write the report from the curated evidence package.

    The synthesiser never sees raw page text. It sees citable, attributed,
    deduplicated findings grouped by sub-question, each labelled with its
    evidence id, and it references those ids. The engine resolves them to
    sources afterwards.
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
        # citable_only: a claim must never rest on a quote we could not find.
        package = store.build_package(sub_questions, citable_only=True)
        gaps: list[str] = _coverage_limitations(coverage, sub_questions)
        if state.get("stop_reason"):
            gaps.append(f"research stopped early: {state['stop_reason']}")
        uncitable = len(evidence) - len(store.citable_evidence())
        if uncitable:
            gaps.append(
                f"{uncitable} extracted finding(s) were excluded because their "
                "quotes could not be located in the source text"
            )

        errors = []
        claim_budget = await _claim_budget()
        if claim_budget is not None:
            log.info("synthesis_claim_budget", claims=claim_budget)
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
                        package.text or "(no citable evidence was gathered)",
                        "\n".join(f"- {g}" for g in gaps),
                        claim_budget=claim_budget,
                    ),
                )
            )
            report = ResearchReport(
                title=out.title.strip() or _clipped(question, 80),
                summary_claims=[_to_claim(c) for c in out.summary_claims],
                sections=[
                    ReportSection(
                        heading=section.heading,
                        claims=[_to_claim(c) for c in section.claims],
                    )
                    for section in out.sections
                ],
                key_findings=[_to_claim(c) for c in out.key_findings],
                contradictions=[
                    Contradiction(
                        topic=c.topic,
                        left_summary=c.left_summary,
                        left_evidence_ids=list(c.left_evidence_ids),
                        right_summary=c.right_summary,
                        right_evidence_ids=list(c.right_evidence_ids),
                    )
                    for c in out.contradictions
                ],
                limitations=_dedupe_limitations(list(out.limitations) + gaps),
            )
        except LLMError as exc:
            log.error("synthesis_failed", error=str(exc)[:300])
            report = _fallback_report(question, store, gaps, _safe_failure_reason(exc))
            errors = [error_from("synthesize", exc, "emitted evidence-only report")]
        timing["evidence_items"] = package.evidence_count

    log.info(
        "synthesis_completed",
        sections=len(report.sections),
        summary_claims=len(report.summary_claims),
        findings=len(report.key_findings),
        evidence_offered=package.evidence_count,
    )
    emit("synthesized", sections=len(report.sections), findings=len(report.key_findings))
    return {"report": report, "stage_timings": [timing], "errors": errors}


def _to_claim(out: object) -> Claim:
    """Build a domain Claim from a model-produced one.

    Citation ids are deliberately left empty here; the engine fills them in
    during resolution by looking each evidence id up. Anything bracket-shaped
    the model left in the prose is stripped at that point too, so an invented
    marker cannot masquerade as a verified citation.
    """
    kind_value = getattr(out, "kind", "factual")
    try:
        kind = ClaimKind(kind_value)
    except ValueError:
        kind = ClaimKind.FACTUAL
    return Claim(
        text=str(getattr(out, "text", "")).strip(),
        evidence_ids=list(getattr(out, "evidence_ids", []) or []),
        citation_ids=[],
        kind=kind,
    )


def _coverage_limitations(
    coverage: CoverageAssessment | None, sub_questions: list[SubQuestion]
) -> list[str]:
    """Turn coverage gaps into sentences a reader can use.

    Two rules. Gaps are named by the sub-question's own text rather than
    its identifier, because "no evidence for SQ3" means nothing outside
    this process. And an entry that does not match a known sub-question id
    is dropped: the critic occasionally returns prose in that field, and
    that prose is its reasoning, not a finding.
    """
    if coverage is None:
        return []
    by_id = {q.id: q.text.rstrip(".?") for q in sub_questions}
    out: list[str] = []
    for sq_id in coverage.missing[:5]:
        text = by_id.get(sq_id)
        if text:
            out.append(f"The retrieved evidence did not answer: {text}.")
    for sq_id in coverage.weak[:5]:
        text = by_id.get(sq_id)
        if text:
            out.append(f"Only limited evidence was found for: {text}.")
    return out


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


def _safe_failure_reason(exc: BaseException) -> str:
    """Why a step failed, in words safe to publish.

    The raw exception is never used. A provider error body is not a
    message written for a reader: it carries the account's organisation
    id, the model name, the configured ceiling and sometimes a URL. One
    was published verbatim in a live report --

        "Rate limit reached for gpt-6-luna in organization org-..."

    -- which put an account identifier in front of every visitor. The
    same reasoning already governs the SSE error path; it simply had not
    been applied here.

    The full exception still goes to the logs, where it belongs.
    """
    from agentic_research.llm.base import (
        BudgetExceededError,
        ModelTimeoutError,
        ModelUnavailableError,
        ProviderRateLimited,
        ProviderRejectedRequest,
        StructuredOutputError,
    )

    if isinstance(exc, ProviderRateLimited):
        return "the model provider's rate limit was reached"
    if isinstance(exc, BudgetExceededError):
        return "this run reached its configured budget"
    if isinstance(exc, ModelTimeoutError):
        return "the model did not respond in time"
    if isinstance(exc, ModelUnavailableError):
        return "the configured model was unreachable"
    if isinstance(exc, StructuredOutputError):
        return "the model did not return the required structure"
    if isinstance(exc, ProviderRejectedRequest):
        return "the provider rejected the request"
    return "the model call did not complete"


def _clipped(text: str, limit: int) -> str:
    """Shorten to a word boundary, marked as shortened.

    A raw slice cuts mid-word and produces a heading that reads like the
    generation broke off -- "...neck-vessel bypass followed b" was a real
    published title. Ending on a whole word with an ellipsis reads as a
    deliberate abbreviation, which is what it is.
    """
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    clipped = collapsed[:limit].rsplit(" ", 1)[0].rstrip(",;:.-")
    # A single word longer than the limit has no boundary to fall back to.
    return f"{clipped or collapsed[:limit]}…"


def _fallback_report(
    question: str, store: EvidenceStore, gaps: list[str], error: str
) -> ResearchReport:
    """Emit the citable evidence directly when synthesis fails.

    A run that gathered forty verified findings should not return nothing
    because the final call failed. The evidence is the expensive part; listing
    it unsynthesised is degraded but genuinely useful — and it is still fully
    provenanced, because each listed finding carries its own evidence id.
    """
    findings = [
        Claim(text=item.claim, evidence_ids=[item.id], kind=ClaimKind.EXTRACTED)
        for item in sorted(store.citable_evidence(), key=lambda e: -e.confidence)[:12]
    ]
    return ResearchReport(
        # Deliberately not the question. Echoing a 160-character research
        # question into a heading and cutting it produced titles like
        # "...open total arch replacement (including redo...", which
        # reads as a generation that broke off rather than a label. The
        # question is rendered in full below the title instead.
        title="Evidence summary",
        summary_claims=[
            Claim(
                text=(
                    "Report synthesis did not run, so this lists the findings "
                    "gathered during research instead. Each one restates a single "
                    "quote that was matched verbatim against its source; unlike a "
                    "normal report, they have not been entailment-checked against "
                    "the evidence they cite."
                ),
                kind=ClaimKind.FRAMING,
            )
        ],
        key_findings=findings,
        limitations=[f"Report synthesis could not run: {error}.", *gaps],
    )


async def verify_citations(state: ResearchState) -> ResearchState:
    """Resolve the report's provenance, validate it, then check entailment."""
    report = state.get("report")
    sources = state.get("sources", [])
    evidence = state.get("evidence", [])
    if report is None:
        return {"verification": None}

    store = EvidenceStore(sources, evidence)
    emit("verifying_citations")

    with stage("verify_citations") as timing:
        # Resolution rewrites the report, so what the reader sees and what the
        # verifier measures are the same object.
        report, resolution_issues = resolve_report(report, store)
        result = verify_structure(report, store, resolution_issues=resolution_issues)
        result.repaired = bool(resolution_issues)

        errors, verdicts = await _check_entailment(report, store, result, state)

        # Publication gate. Claims the verifier could not support are
        # removed rather than rewritten; the issues explaining why stay in
        # the verification record so the removal remains auditable.
        result.generated_substantive_claims = len(report.substantive_claims())
        report, removed = filter_report_by_verification(report, verdicts)
        result.removed_after_verification = removed
        result.final_published_claims = len(report.substantive_claims())
        if removed:
            # Citation totals describe the report a reader receives, so
            # they are recomputed against the filtered one.
            result = verify_structure(
                report, store, resolution_issues=resolution_issues
            ).model_copy(
                update={
                    "checked_claims": result.checked_claims,
                    "checkable_claims": result.checkable_claims,
                    "not_checked_claims": result.not_checked_claims,
                    "supported_claims": result.supported_claims,
                    "partially_supported_claims": result.partially_supported_claims,
                    "unsupported_claims": result.unsupported_claims,
                    "entailment_exhaustive": result.entailment_exhaustive,
                    "issues": result.issues,
                    "repaired": result.repaired,
                    "generated_substantive_claims": result.generated_substantive_claims,
                    "removed_after_verification": removed,
                    "final_published_claims": result.final_published_claims,
                }
            )

        timing["citations"] = result.total_citations

    log.info(
        "citation_verification_completed",
        citations=result.total_citations,
        citation_integrity=result.citation_integrity_rate,
        evidence_integrity=result.evidence_integrity_rate,
        coverage_rate=result.citation_coverage_rate,
        support=result.support_breakdown,
        exhaustive=result.entailment_exhaustive,
        generated=result.generated_substantive_claims,
        removed=result.removed_after_verification,
        published=result.final_published_claims,
        unused_sources=len(result.unused_source_ids),
    )
    emit(
        "citations_verified",
        total=result.total_citations,
        evidence_integrity=result.evidence_integrity_rate,
        support=result.support_breakdown,
        exhaustive=result.entailment_exhaustive,
        not_checked=result.not_checked_claims,
        removed=result.removed_after_verification,
        published=result.final_published_claims,
    )
    return {
        "report": report,
        "verification": result.model_dump(mode="json"),
        "stage_timings": [timing],
        "errors": errors,
    }


async def _check_entailment(
    report: ResearchReport,
    store: EvidenceStore,
    result: CitationVerification,
    state: ResearchState,
) -> tuple[list, dict[ClaimKey, ClaimVerdict]]:
    """Ask the verifier model whether a claim's own evidence supports it.

    The evidence shown is exactly the evidence the claim references. The
    previous implementation sampled up to three items belonging to a cited
    *source*, which frequently meant judging a claim against text that played
    no part in producing it.
    """
    verdicts: dict[ClaimKey, ClaimVerdict] = {}
    candidates = [c for c in report.substantive_claims() if c.evidence_ids]
    result.checkable_claims = len(candidates)
    if not candidates:
        return [], verdicts

    exhaustive = bool(state.get("exhaustive_verification"))
    if exhaustive:
        selected = candidates
    else:
        # Summary and key findings first: they are what a reader takes
        # away, and under the publication gate an unchecked claim does
        # not survive, so check order decides what gets published.
        prominent = [c for c in report.summary_claims if c in candidates]
        prominent += [c for c in report.key_findings if c in candidates]
        rest = [c for c in candidates if c not in prominent]
        # Bounded by what the run can still pay for, not just by the
        # sample size. Attempting calls the budget cannot cover spends
        # the last of it and then fails mid-loop, which is a worse
        # outcome than checking fewer claims deliberately.
        affordable = await ctx().router.tracker.remaining()
        limit = max(1, min(_DEFAULT_ENTAILMENT_SAMPLE, affordable))
        selected = (prominent + rest)[:limit]
    result.entailment_exhaustive = len(selected) == len(candidates)

    errors = []
    model = ctx().router.get(ModelRole.VERIFIER)
    for claim in selected:
        block = _evidence_block(claim.evidence_ids, store)
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
            result.entailment_exhaustive = False
            break

        result.checked_claims += 1
        verdicts[key_of(claim)] = out.verdict
        if out.verdict == "supported":
            result.supported_claims += 1
        elif out.verdict == "partially_supported":
            result.partially_supported_claims += 1
            result.issues.append(
                CitationIssue(
                    type=CitationIssueType.PARTIALLY_SUPPORTED_CLAIM,
                    severity="warning",
                    claim_text=claim.text[:200],
                    evidence_id=",".join(claim.evidence_ids),
                    detail=out.reason[:200],
                )
            )
        else:
            result.unsupported_claims += 1
            result.issues.append(
                CitationIssue(
                    type=CitationIssueType.UNSUPPORTED_CLAIM,
                    severity="warning",
                    claim_text=claim.text[:200],
                    evidence_id=",".join(claim.evidence_ids),
                    detail=out.reason[:200],
                )
            )
    result.not_checked_claims = max(0, result.checkable_claims - result.checked_claims)
    return errors, verdicts


def _evidence_block(evidence_ids: list[str], store: EvidenceStore) -> str:
    """Render exactly the evidence a claim references."""
    lines = []
    for evidence_id in evidence_ids[:6]:
        item = store.evidence_by_id(evidence_id)
        if item is None:
            continue
        page = f" (p. {item.page})" if item.page else ""
        lines.append(f'{item.id}{page}: "{item.quote[:300]}"')
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
    markdown = render_markdown(
        report, state.get("sources", []), verification, evidence=state.get("evidence", [])
    )
    reason = stop_reason_for(state)

    log.info("research_completed", stop_reason=reason, sources=len(state.get("sources", [])))
    emit("completed", stop_reason=reason)
    return {"final_markdown": markdown, "stop_reason": reason}
