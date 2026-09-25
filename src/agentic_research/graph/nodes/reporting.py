"""Synthesis, citation verification and finalisation."""

from __future__ import annotations

from typing import Any

from agentic_research.citations.publication import (
    ClaimKey,
    ClaimVerdict,
    claim_key,
    deduplicate_claims,
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
    ClaimJudgment,
    ClaimKind,
    Contradiction,
    CoverageAssessment,
    ReportSection,
    ResearchReport,
    RunError,
    SubQuestion,
)
from agentic_research.observability import get_logger
from agentic_research.schemas import MAX_EVIDENCE_PER_CLAIM, EntailmentOut, ReportOut

log = get_logger(__name__)

# Entailment costs one model call per claim. Interactive runs sample; the
# benchmark checks everything, because a sampled number reported as if it were
# exhaustive is the kind of metric this project exists not to publish.
_DEFAULT_ENTAILMENT_SAMPLE = 10

# Headroom left when sizing the report: verification also spends a call
# resolving structure, and a structured-output repair can cost another.
_VERIFICATION_OVERHEAD = 2


async def _claim_budget() -> int | None:
    """How many substantive claims this run can afford to verify.

    Every substantive claim costs one entailment call, and under the
    publication gate an unverified claim is not published. Generating
    more than the budget allows does not lengthen the report; the surplus
    is deleted after being paid for. A live run generated 25 claims,
    could check 4, and published 2.

    Returns ``None`` when the remaining budget is ample, so an
    unconstrained local run is not told to write a short report for no
    reason, and ``0`` when the run cannot afford synthesis plus even one
    verified claim -- the caller emits the evidence listing instead of
    promising claims that would be removed on the way out.

    Deliberately no floor. An earlier version asked for at least four
    claims regardless of budget, which is a promise the run could not
    keep: those claims were generated, went unverified and were then
    dropped, so the floor bought nothing but spend.
    """
    remaining = await ctx().router.tracker.remaining()
    # One for synthesis itself, which has not been reserved yet.
    affordable = remaining - 1 - _VERIFICATION_OVERHEAD
    if affordable >= _DEFAULT_ENTAILMENT_SAMPLE:
        return None
    return max(0, affordable)


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

        errors: list[RunError] = []
        claim_budget = await _claim_budget()
        if claim_budget == 0:
            # Not enough budget left to synthesise *and* verify even one
            # claim. Every claim written here would be removed by the
            # publication gate, so the honest and cheaper answer is the
            # evidence listing, which needs no model call at all.
            log.warning("synthesis_skipped_no_verification_budget")
            report = _fallback_report(
                question, store, gaps, "the run's model-call budget was exhausted"
            )
            emit("synthesized", sections=0, findings=len(report.key_findings))
            timing["evidence_items"] = package.evidence_count
            return {"report": report, "stage_timings": [timing], "errors": errors}
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
    # The quote itself, not the extractor's paraphrase of it.
    #
    # EXTRACTED bypasses entailment because this path runs when synthesis
    # or verification is unavailable, which is only defensible if the
    # published text carries a guarantee of its own. `item.claim` does
    # not: exact quote matching proves the *quote* appears in the source,
    # and says nothing about whether the paraphrase beside it is faithful.
    # Publishing the verified span makes the fallback deterministic --
    # every character of it was matched against the source.
    findings = [
        Claim(text=item.quote, evidence_ids=[item.id], kind=ClaimKind.EXTRACTED)
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
                    "Report synthesis did not run, so this lists source excerpts "
                    "gathered during research instead. Each line is a quote "
                    "reproduced exactly from its source and matched against it; "
                    "unlike a normal report, nothing here has been written or "
                    "interpreted, and no claim has been entailment-checked."
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
        # The order below is the definition of every counter this node
        # reports, so it is written out step by step. Previously the
        # "generated" count was taken after deduplication and structural
        # totals could describe a report the reader never saw.
        #
        #   1. resolve references
        #   2. count generated substantive claims -- raw synthesiser output
        #   3. remove exact duplicates
        #   4. structural verification of the deduplicated candidate
        #   5. semantic verification
        #   6. publication gate
        #   7. structural metrics recomputed on the published report
        #
        # Resolution rewrites the report, so what the reader sees and what
        # the verifier measures are the same object.
        report, resolution_issues = resolve_report(report, store)

        # 2. Before dedup and before any filtering: what synthesis produced.
        generated = len(report.substantive_claims())

        # 3. A claim repeated three times is checked once rather than
        # spending three entailment calls on one sentence -- in a bounded
        # run, three claims' worth of budget for one finding.
        report, duplicates = deduplicate_claims(report)
        if duplicates:
            log.info("duplicate_claims_removed", count=duplicates)

        # 4. Structural verification of the candidate the reader may get.
        result = verify_structure(report, store, resolution_issues=resolution_issues)
        result.repaired = bool(resolution_issues)
        result.generated_substantive_claims = generated
        result.duplicate_claims_removed = duplicates

        # 5. Semantic verification, claims then contradiction sides.
        errors, verdicts = await _check_entailment(report, store, result, state)

        # 6. Publication gate. Claims and contradictions the verifier could
        # not support are removed rather than rewritten; the issues
        # explaining why stay in the record so removal remains auditable.
        published_report, removed = filter_report_by_verification(report, verdicts)

        # 7. Structural totals always describe the published report, even
        # when nothing was removed semantically -- deduplication alone can
        # change them.
        semantic = {
            "checked_claims": result.checked_claims,
            "checkable_claims": result.checkable_claims,
            "not_checked_claims": result.not_checked_claims,
            "supported_claims": result.supported_claims,
            "partially_supported_claims": result.partially_supported_claims,
            "unsupported_claims": result.unsupported_claims,
            "entailment_exhaustive": result.entailment_exhaustive,
            "contradiction_sides_checkable": result.contradiction_sides_checkable,
            "contradiction_sides_checked": result.contradiction_sides_checked,
            "issues": result.issues,
            "repaired": result.repaired,
            "generated_substantive_claims": generated,
            "duplicate_claims_removed": duplicates,
            "removed_after_verification": removed,
        }
        report = published_report
        result = verify_structure(report, store, resolution_issues=resolution_issues).model_copy(
            update=semantic
        )
        result.final_published_claims = len(report.substantive_claims())
        result.contradictions_semantically_supported = len(report.contradictions)

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
    candidates = []

    def judgment(claim: Claim, *, reason: str | None = None) -> ClaimJudgment:
        """Record every candidate, checked or not, with its full text."""
        record = ClaimJudgment(
            claim_text=claim.text,
            kind=claim.kind,
            evidence_ids=list(claim.evidence_ids),
            reason=reason,
        )
        result.judgments.append(record)
        return record

    for claim in report.substantive_claims():
        if not claim.evidence_ids:
            judgment(claim, reason="claim cites no evidence")
            continue
        if len(claim.evidence_ids) > MAX_EVIDENCE_PER_CLAIM:
            # Cannot be shown to the verifier in full, so it is not
            # checked and therefore not published. Recorded rather than
            # trimmed: judging a subset while reporting an all-evidence
            # check is the defect this bound exists to prevent.
            result.issues.append(
                CitationIssue(
                    type=CitationIssueType.UNSUPPORTED_CLAIM,
                    severity="warning",
                    claim_text=claim.text[:200],
                    evidence_id=",".join(claim.evidence_ids),
                    detail=(
                        f"claim cites {len(claim.evidence_ids)} evidence items, above the "
                        f"{MAX_EVIDENCE_PER_CLAIM} that can be verified together; not checked"
                    ),
                )
            )
            judgment(
                claim,
                reason=(
                    f"cites {len(claim.evidence_ids)} evidence items, above the "
                    f"{MAX_EVIDENCE_PER_CLAIM} that can be verified together"
                ),
            )
            continue
        candidates.append(claim)
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
    # Candidates the sample did not reach are still recorded, so the
    # audit trail distinguishes "judged not supported" from "never
    # looked at". Both are removed by the gate; they are not the same
    # finding.
    unreached = {id(c) for c in candidates} - {id(c) for c in selected}
    for claim in candidates:
        if id(claim) in unreached:
            judgment(claim, reason="not reached within this run's verification budget")

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
        record = judgment(claim, reason=out.reason)
        record.verdict = out.verdict
        record.checked = True
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
    errors += await _check_contradictions(report, store, result, state, verdicts, model)
    return errors, verdicts


async def _check_contradictions(
    report: ResearchReport,
    store: EvidenceStore,
    result: CitationVerification,
    state: ResearchState,
    verdicts: dict[ClaimKey, ClaimVerdict],
    model: Any,
) -> list:
    """Entailment-check both sides of every contradiction.

    A contradiction's two summaries are model-written assertions, and they
    reached the published report without any support check: structural
    resolution proved only that evidence existed on each side, which is
    what ``is_auditable`` reports. Prose asserting what a source says can
    be wrong in exactly the ways a claim can.

    Both sides must come back supported for the contradiction to survive.
    Partial, unsupported, unchecked or structurally unresolvable on either
    side removes it, and the reason stays in the issue list.
    """
    errors: list = []
    if not report.contradictions:
        return errors

    exhaustive = bool(state.get("exhaustive_verification"))
    for index, contradiction in enumerate(report.contradictions):
        sides = (
            ("left", contradiction.left_summary, contradiction.left_evidence_ids),
            ("right", contradiction.right_summary, contradiction.right_evidence_ids),
        )
        for side, summary, evidence_ids in sides:
            if not evidence_ids or len(evidence_ids) > MAX_EVIDENCE_PER_CLAIM:
                continue
            result.contradiction_sides_checkable += 1

            block = _evidence_block(evidence_ids, store)
            if not block:
                continue
            # Shares the verifier budget with claims. When nothing is left,
            # the side stays unchecked and the contradiction is dropped
            # rather than published on an unverified summary.
            if not exhaustive and await ctx().router.tracker.remaining() <= 0:
                continue
            try:
                out = await model.structured(
                    EntailmentOut, VERIFIER_SYSTEM, verifier_user(strip_markers(summary), block)
                )
            except LLMError as exc:
                log.warning("contradiction_check_failed", error=str(exc)[:200])
                errors.append(error_from("verify_citations", exc, "contradiction check stopped"))
                break

            result.contradiction_sides_checked += 1
            verdicts[claim_key(summary, evidence_ids)] = out.verdict
            if out.verdict != "supported":
                issue_type = (
                    CitationIssueType.PARTIALLY_SUPPORTED_CLAIM
                    if out.verdict == "partially_supported"
                    else CitationIssueType.UNSUPPORTED_CLAIM
                )
                result.issues.append(
                    CitationIssue(
                        type=issue_type,
                        severity="warning",
                        claim_text=summary[:200],
                        evidence_id=",".join(evidence_ids),
                        detail=f"contradiction {index} {side} side: {out.reason[:160]}",
                    )
                )
    return errors


def _evidence_block(evidence_ids: list[str], store: EvidenceStore) -> str:
    """Render the evidence a claim references, with who published it.

    The quote alone is not enough to judge an attributed claim. A vendor
    page recommending that AI governance sit with a CISO reads exactly
    like the NIST framework requiring it, once the publisher is stripped
    away -- and a human audit found the verifier passing precisely that
    substitution. Naming the source makes the attribution checkable.

    Every cited id is rendered and every quote is whole. Showing the
    first six and cutting each quote at 300 characters changed the
    material being judged while the artifact still recorded an
    all-evidence check: one NIST claim cited eight ids, and the seventh
    stated its case outright. Over-long claims are refused at the schema
    boundary instead, so nothing arrives here that cannot be shown in
    full.

    "Site category" rather than "Type", because a retrieval taxonomy is
    not authority: docs.modulos.ai classifies as official_docs without
    being an official source for NIST. Quality score is absent for the
    same reason -- it is a heuristic over document kind and rank, and a
    verifier would read it as confidence in the claim.
    """
    blocks = []
    for evidence_id in evidence_ids:
        item = store.evidence_by_id(evidence_id)
        if item is None:
            continue
        source = store.source(item.source_id)
        title = (source.title if source else "") or "unknown"
        domain = (source.domain if source else "") or "unknown"
        category = source.source_type.value if source else "unknown"
        blocks.append(
            f"Evidence: {item.id}\n"
            f"Source: {title}\n"
            f"Domain: {domain}\n"
            f"Site category: {category}\n"
            f"Page: {item.page if item.page else 'n/a'}\n"
            f'Quote: "{item.quote}"'
        )
    return "\n\n".join(blocks)


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
