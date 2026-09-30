"""Synthesis, citation verification and finalisation."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from agentic_research.answer_contract import AnswerContract
from agentic_research.answer_coverage import assess_coverage
from agentic_research.citations.guards import SourceAuthority, SourceIdentity, authority_of
from agentic_research.citations.nli import NLIUnavailable, build_verifier
from agentic_research.citations.propositions import decompose
from agentic_research.citations.publication import (
    ClaimKey,
    ClaimVerdict,
    claim_key,
    deduplicate_claims,
    filter_report_by_verification,
    key_of,
)
from agentic_research.citations.relevance import deterministic_relevance
from agentic_research.citations.repair import is_repairable, validate_rewrite
from agentic_research.citations.semantic import (
    CitedEvidence,
    Scorer,
    SemanticVerdict,
    verify_claim,
)
from agentic_research.citations.verifier import (
    resolve_report,
    verify_structure,
)
from agentic_research.comparison import (
    ComparisonPair,
    SideClaim,
    dynamically_discharging_slots,
    pairs_from_payload,
)
from agentic_research.config import ModelRole
from agentic_research.evidence.store import EvidenceStore
from agentic_research.graph.nodes.common import ctx, emit, error_from, stage
from agentic_research.graph.prompts import (
    RELEVANCE_SYSTEM,
    REPAIR_SYSTEM,
    SYNTHESIZER_SYSTEM,
    relevance_user,
    repair_user,
    synthesizer_user,
)
from agentic_research.graph.state import ResearchState
from agentic_research.models import (
    CitationIssue,
    CitationIssueType,
    CitationVerification,
    Claim,
    ClaimJudgment,
    ClaimKind,
    Contradiction,
    CoverageAssessment,
    EvidenceScoreRecord,
    PropositionRecord,
    RelevanceRecord,
    RepairRecord,
    ReportSection,
    ResearchReport,
    RunError,
    SubQuestion,
)
from agentic_research.observability import get_logger
from agentic_research.report import MAX_EXCERPTS
from agentic_research.schemas import (
    MAX_EVIDENCE_PER_CLAIM,
    RelevanceOut,
    RepairOut,
    ReportOut,
)

log = get_logger(__name__)

# What synthesis costs *after* the report is written, however many
# claims it contains: one batched relevance judgement, and at most one
# batched wording-repair pass. Neither is per-claim.
_POST_SYNTHESIS_CALLS = 2

# How many claims a contract is worth, per part it asks for.
#
# Two: one to fill the part and one in reserve, since a claim can be
# refused on wording or on evidence and the part still wants filling.
# More than that is padding -- the contract names what the answer
# requires, and a report with four claims per part is not answering
# it four times over.
_CLAIMS_PER_SLOT = 2

# The floor when no contract could be built, and the value that stood
# in for all of this before. A question with no resolvable shape still
# gets a bounded report rather than an unbounded one.
_CLAIMS_WITHOUT_A_CONTRACT = 6


async def _claim_budget(contract: AnswerContract | None = None) -> int | None:
    """How many claims this report is worth writing, and whether it can
    be written at all.

    Two separate bounds, and getting them confused cost a release.

    **Affordability.** ``0`` when the run cannot pay for synthesis and
    the two batched calls that follow it. Without the relevance
    judgement every claim is withheld, so synthesising would spend a
    call to publish nothing, and the caller emits the evidence listing
    instead.

    **Focus.** Otherwise, the contract decides: ``_CLAIMS_PER_SLOT``
    for each part the answer requires. This is a *quality* bound and
    is now labelled as one.

    It is a **request, not a ceiling**, and that is worth being exact
    about because the name invites the opposite reading. The number
    reaches the synthesiser as "write at most N" and the synthesiser
    may write more; nothing trims the surplus. A local run asked for
    six and produced eight.

    Not enforced deliberately. Every claim is gated individually and
    an extra one costs no model call, so exceeding the request is
    untidy rather than unsafe -- while truncating a report to a count
    would mean discarding claims before anything had looked at them,
    and the one filling the required part is as likely to go as any
    other. What the request buys is measured and real: unbounded, the
    same question produced thirteen claims; asked for six, eight.

    It did not used to be. The cap was derived from the remaining call
    budget, one call per claim, which was right when entailment was a
    generative call and wrong from the moment the NLI classifier
    replaced it -- the classifier is not a model call, and the
    judgement and repair are batched, so an extra claim costs nothing.
    Removing the spend justification was correct. Removing the cap with
    it was not, and the hosted runs measured the difference:

        with a cap of 7   5 claims generated, 1 published
        with no cap      13 claims generated, 0 published

    The cap had been doing a second job nobody had written down. The
    synthesiser given no bound wrote thin claims until it ran out of
    evidence, and a larger batch of thin claims fared worse at the
    relevance gate than a smaller batch of considered ones. So the
    bound stays, tied to the thing that actually says how much answer
    is wanted.
    """
    remaining = await ctx().router.tracker.remaining()
    # One for synthesis itself, which has not been reserved yet.
    if remaining - 1 - _POST_SYNTHESIS_CALLS < 0:
        return 0
    if contract is None or not contract.usable or not contract.required_slots:
        return _CLAIMS_WITHOUT_A_CONTRACT
    return _CLAIMS_PER_SLOT * len(contract.required_slots)


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
    contract = state.get("contract")
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
        claim_budget = await _claim_budget(contract)
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
                        # The slots themselves, not name/description
                        # pairs: flattening them here dropped `core`,
                        # and the synthesiser could not tell the slot
                        # the answer turns on from the optional ones.
                        answer_slots=(
                            contract.required_slots
                            if contract is not None and contract.usable
                            else None
                        ),
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
        except Exception as exc:
            # The evidence-only report exists for exactly this. A
            # non-LLMError here used to end the run instead, losing
            # every quote already extracted and paid for.
            log.error("synthesis_failed", error_type=type(exc).__name__, error=str(exc)[:300])
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
        answer_slot=str(getattr(out, "answer_slot", "") or "").strip(),
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
            # Carried explicitly. Step 7 rebuilds the record from the
            # published report, and anything not listed here is silently
            # replaced by the fresh object's default -- which is how the
            # judgments were being appended and then dropped before the
            # artifact was ever written, leaving the truncated
            # CitationIssue list as the only surviving audit trail.
            "judgments": result.judgments,
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
        # Counted separately and never folded into the claim count. When
        # every claim is withheld the report falls back to verbatim
        # excerpts, and reporting those as findings would undo the
        # withholding it just did.
        if result.final_published_claims == 0:
            result.evidence_only_excerpts = min(len(store.citable_evidence()), MAX_EXCERPTS)

        # What the published claims actually covered of what was asked.
        #
        # A report could finish with claims in it and answer nothing --
        # a comparison whose survivors are two definitions is the case
        # that prompted this. The limitations say which required part
        # is missing, by name, rather than leaving a reader to infer it
        # from the absence.
        contract = state.get("contract")
        coverage = None
        if contract is not None:
            published = report.substantive_claims()
            coverage = assess_coverage(
                contract,
                [c.answer_slot for c in published if c.answer_slot],
                # Slot and text together, per claim. A contrast is now
                # assembled from one verified atomic claim per subject
                # within a single dimension, which needs to know which
                # dimension each claim was declared against -- and the
                # two were previously passed as separate lists built
                # with different filters, so they were not aligned.
                claims=[
                    SideClaim(
                        subject="",
                        text=c.text,
                        answer_slot=c.answer_slot or "",
                        evidence_ids=tuple(c.evidence_ids),
                    )
                    for c in published
                ],
                # The sources, so a question naming a subject that no
                # source discusses is reported as such instead of as
                # an uncovered contract. Title included: a retrieval
                # that found the subject but could not extract its
                # body still mentions it.
                source_texts=[f"{src.title}\n{src.text}" for src in store.usable_sources()],
            )
            gaps = coverage.limitations()
            if gaps:
                report = report.model_copy(
                    update={"limitations": _dedupe_limitations([*report.limitations, *gaps])}
                )
            log.info(
                "answer_coverage",
                answered=coverage.answered,
                satisfied=list(coverage.satisfied),
                missing=list(coverage.missing_core),
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
        # The coverage assessment itself, not just the sentences it
        # produced. `absent_entities` is a fact about the retrieved
        # sources and cannot be re-derived from the published claims,
        # so the client has to be told rather than left to compute it.
        "answer_coverage": None if coverage is None else coverage.to_dict(),
        "stage_timings": [timing],
        "errors": errors,
    }


async def _check_entailment(
    report: ResearchReport,
    store: EvidenceStore,
    result: CitationVerification,
    state: ResearchState,
) -> tuple[list, dict[ClaimKey, ClaimVerdict]]:
    """Score every claim against its own evidence with the NLI verifier.

    The evidence scored is exactly the evidence the claim references,
    one quote at a time. A claim publishes when a single quote carries
    it; quotes are never concatenated, because assembling a broad claim
    out of several partial ones is the failure this gate exists to stop.

    No sampling and no budget arithmetic. The previous implementation
    spent one generative call per claim, so most claims in a bounded run
    were never checked and were dropped unchecked. Classification is
    local and costs no provider request, so every checkable candidate is
    checked and ``entailment_exhaustive`` is true whenever the model
    loaded at all.
    """
    verdicts: dict[ClaimKey, ClaimVerdict] = {}
    candidates = []

    def judgment(claim: Claim, *, reason: str | None = None) -> ClaimJudgment:
        """Record every candidate, checked or not, with its full text."""
        record = ClaimJudgment(
            claim_text=claim.text,
            kind=claim.kind,
            answer_slot=claim.answer_slot or "",
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
            # A claim resting on more than this is not atomic, whatever
            # else it is. Recorded rather than trimmed: scoring a subset
            # while reporting an all-evidence check is the defect this
            # bound exists to prevent.
            result.issues.append(
                CitationIssue(
                    type=CitationIssueType.UNSUPPORTED_CLAIM,
                    severity="warning",
                    claim_text=claim.text[:200],
                    evidence_id=",".join(claim.evidence_ids),
                    detail=(
                        f"claim cites {len(claim.evidence_ids)} evidence items, above the "
                        f"{MAX_EVIDENCE_PER_CLAIM} a single atomic claim may rest on; not checked"
                    ),
                )
            )
            judgment(
                claim,
                reason=(
                    f"cites {len(claim.evidence_ids)} evidence items, above the "
                    f"{MAX_EVIDENCE_PER_CLAIM} a single atomic claim may rest on"
                ),
            )
            continue
        candidates.append(claim)
    result.checkable_claims = len(candidates)
    if not candidates:
        return [], verdicts

    settings = ctx().settings
    threshold = settings.nli_support_threshold
    errors: list = []
    contract = state.get("contract")
    # The audit record travels with the claim through both later
    # gates. Relevance and repair are decided after this loop, and
    # without a handle on the record they were decided nowhere the
    # transcript could see.
    awaiting_judgement: list[tuple[Claim, SemanticVerdict, ClaimJudgment]] = []
    repairable: list[tuple[Claim, SemanticVerdict, list[CitedEvidence], str, ClaimJudgment]] = []

    try:
        scorer = ctx().nli_scorer or build_verifier(settings)
    except NLIUnavailable as exc:
        # Fail closed for the whole report. Never a fallback to the
        # generative model: unverified is unverified, and a claim nobody
        # checked must not reach a reader looking like one that passed.
        log.warning("nli_verifier_unavailable", error=str(exc)[:200])
        errors.append(error_from("verify_citations", exc, "semantic verification unavailable"))
        result.entailment_exhaustive = False
        for claim in candidates:
            judgment(claim, reason=f"semantic verification unavailable: {exc}")
        result.not_checked_claims = len(candidates)
        return errors, verdicts

    for claim in candidates:
        pairs = _scoring_pairs(claim.evidence_ids, store)
        # Classification is CPU-bound and synchronous; off-thread so a
        # long report does not stall the event loop and its progress
        # stream along with it.
        verdict = await asyncio.to_thread(
            verify_claim, claim.text, pairs, scorer, support_threshold=threshold
        )

        # Support and relevance are separate questions, and only ever
        # one of them was being asked. A claim can be entailed by its
        # quote and answer nothing that was put to the engine; a live
        # run published five of those. Relevance is checked against
        # the contract built before retrieval, and a supported but
        # irrelevant claim is withheld with its own reason rather than
        # being reported as unsupported, which it is not.
        # Created here rather than after the gates below, because the
        # gates need somewhere to write. It is appended to
        # result.judgments on construction, so the ordering of the
        # transcript is unchanged.
        record = judgment(claim)

        if verdict.publishable and contract is not None:
            # Structure only, and free. The judgement is asked once for
            # everything that survives, below.
            relevance = deterministic_relevance(
                claim.text,
                claim.answer_slot or None,
                contract,
                evidence_text=" ".join(item.quote for item in pairs),
            )
            record.relevance = RelevanceRecord(
                stage="structural",
                relevant=relevance.publishable,
                reason=relevance.reason,
            )
            if not relevance.publishable:
                verdict = replace(
                    verdict,
                    publishable=False,
                    verdict="irrelevant",
                    reason=f"does not answer the question: {relevance.reason}",
                )
            else:
                awaiting_judgement.append((claim, verdict, record))
        elif not verdict.publishable and verdict.checked:
            eligible, why = is_repairable(
                verdict.reason,
                every_proposition_supported=_propositions_supported(
                    claim.text, pairs, scorer, threshold
                ),
            )
            if eligible:
                repairable.append((claim, verdict, pairs, why, record))

        _record(claim.text, claim.evidence_ids, verdict, result, record)
        if verdict.checked:
            result.checked_claims += 1
            verdicts[key_of(claim)] = verdict.verdict
            if verdict.publishable:
                result.supported_claims += 1
            elif verdict.verdict == "partially_supported":
                result.partially_supported_claims += 1
            else:
                result.unsupported_claims += 1
        if not verdict.publishable:
            result.issues.append(
                CitationIssue(
                    type=(
                        CitationIssueType.PARTIALLY_SUPPORTED_CLAIM
                        if verdict.verdict == "partially_supported"
                        else CitationIssueType.UNSUPPORTED_CLAIM
                    ),
                    severity="warning",
                    claim_text=claim.text[:200],
                    evidence_id=",".join(claim.evidence_ids),
                    detail=verdict.reason[:200],
                )
            )

    # One repair pass, before the judgement, for claims refused on
    # wording alone.
    #
    # Eligibility is narrow by design: every proposition must already
    # be supported, and the refusal must name only wording guards.
    # Repair is not a second chance at evidence, and the rewrite is
    # checked deterministically before it is re-verified so nothing it
    # smuggled in reaches the gates that would have to notice.
    if contract is not None:
        repaired = await _repair_wording(contract, repairable, store, scorer, threshold)
        for claim, verdict, record in repaired:
            verdicts[key_of(claim)] = "supported"
            result.supported_claims += 1
            result.unsupported_claims = max(0, result.unsupported_claims - 1)
            # Re-scored on the new wording, so the stored verdict and
            # propositions must describe the text that will publish,
            # not the text that was refused.
            _record(claim.text, claim.evidence_ids, verdict, result, record)
            awaiting_judgement.append((claim, verdict, record))

    # One judgement call for everything that survived structure.
    #
    # Asked of the critic rather than the synthesiser: a model marking
    # its own homework finds it relevant. Batched into a single request
    # because a public run has twenty in total, and a gate that costs
    # one per claim would be the most expensive thing in the report.
    #
    # Fails closed. If the judgement cannot be obtained, the claims it
    # would have covered are withheld -- an unanswered relevance
    # question is not a yes.
    if awaiting_judgement and contract is not None:
        judged = await _judge_relevance(contract, [claim for claim, _, _ in awaiting_judgement])
        for index, (claim, verdict, record) in enumerate(awaiting_judgement):
            answers, why = judged.get(index, (None, "no judgement was returned"))
            # Overwrites the structural record on purpose: the
            # judgement is the decision that stood, and the structural
            # pass that preceded it is a yes by construction -- nothing
            # reaches here having failed it.
            record.relevance = RelevanceRecord(stage="judged", relevant=answers, reason=why)
            if answers is True:
                continue

            # The judge decides *core* slots. It does not veto a claim
            # that structurally fills an optional part of the answer.
            #
            # It was vetoing them, and that is why comparisons produced
            # nothing. Asked how two things differ, the synthesiser
            # writes claims about each of them and declares `dimension`;
            # the judge then refused each one for "describing neural
            # networks, not how LLMs differ" -- applying the *report's*
            # question to a single claim, which no single atomic claim
            # can answer, because a contrast asserts two things and the
            # atomicity guard refuses those. Nine runs produced one
            # `direct_contrast` claim and it was refused on atomicity.
            #
            # A prompt change telling the judge to ask whether a claim
            # fills a listed part was tried first and measurably did not
            # work: the run after it refused three `dimension` claims
            # with the same reasoning. So the authority is narrowed in
            # code rather than requested in a prompt.
            #
            # What still stops the original defect -- five supported
            # claims about one of two subjects, published as an answer
            # to how they differ -- is that it was never a claim-level
            # problem. It is a *report*-level one, and `assess_coverage`
            # refuses to call such a report answered, because the
            # published claims must between them speak about every
            # subject the question named. A one-sided report now
            # publishes its claims and states that it did not answer.
            #
            # An off-topic claim does not get through either: the
            # structural check ahead of this requires the claim or its
            # evidence to mention one of the contract's subjects, and a
            # claim about a different named subject is refused there.
            if not _discharges_a_core_slot(contract, claim.answer_slot):
                continue

            reason = (
                f"does not answer the question: {why}"
                if answers is False
                else f"relevance could not be judged: {why}"
            )
            demoted = replace(verdict, publishable=False, verdict="irrelevant", reason=reason)
            record.verdict = "irrelevant"
            record.publishable = False
            record.reason = reason
            verdicts[key_of(claim)] = "irrelevant"
            result.supported_claims = max(0, result.supported_claims - 1)
            result.unsupported_claims += 1
            result.issues.append(
                CitationIssue(
                    type=CitationIssueType.UNSUPPORTED_CLAIM,
                    severity="warning",
                    claim_text=claim.text[:200],
                    evidence_id=",".join(claim.evidence_ids),
                    detail=demoted.reason[:200],
                )
            )

    result.entailment_exhaustive = result.checked_claims == result.checkable_claims
    result.not_checked_claims = max(0, result.checkable_claims - result.checked_claims)
    errors += await _check_contradictions(report, store, result, state, verdicts, scorer)
    return errors, verdicts


def _discharges_a_core_slot(contract: AnswerContract, slot: str | None) -> bool:
    """Whether this claim's slot is one the answer turns on.

    A core slot, or one the contract lets stand in for a core slot --
    `relationship` discharging `direct_contrast`, for instance. Claims
    declaring an optional part are judged and recorded, and a negative
    judgement does not withhold them; the report's coverage is what
    decides whether the optional parts added up to an answer.

    A claim declaring no slot, or a slot this contract never asked
    for, is treated as core. That is the conservative direction:
    nothing is known about what the claim was for, so the judge's
    verdict stands. The structural check ahead of this already refuses
    an unknown slot, but narrowing an authority should not depend on
    another gate having caught the case first.
    """
    if not slot or not contract.has_slot(slot):
        return True
    if any(slot == core.name or slot in core.satisfied_by for core in contract.core_slots):
        return True
    # Slots that can discharge a core slot without saying so in the
    # contract. `relationship` no longer declares a static
    # `satisfied_by`, because whether it answers a comparison depends
    # on the kind of relationship it asserts -- but it can still be
    # the whole answer, so the judge keeps authority over it. Reading
    # only `satisfied_by` silently removed that authority when the
    # static alternative was dropped.
    return slot in dynamically_discharging_slots(contract)


def _propositions_supported(
    claim_text: str,
    pairs: list[CitedEvidence],
    scorer: Scorer,
    threshold: float,
) -> bool:
    """Whether every assertion in the claim already has its evidence.

    The precondition for repair. A claim whose evidence does not carry
    it is not eligible however its wording reads, because rewording
    an unsupported claim into a supported-looking one is the
    laundering the whole design refuses.
    """
    for part in decompose(claim_text):
        verdict = verify_claim(
            part.text, pairs, scorer, support_threshold=threshold, check_propositions=False
        )
        if not verdict.publishable and verdict.best_entailment < threshold:
            return False
    return True


async def _repair_wording(
    contract: AnswerContract,
    candidates: list[tuple[Claim, SemanticVerdict, list[CitedEvidence], str, ClaimJudgment]],
    store: EvidenceStore,
    scorer: Scorer,
    threshold: float,
) -> list[tuple[Claim, SemanticVerdict, ClaimJudgment]]:
    """One rewrite attempt each, then every gate again from the start.

    Returns only claims that passed on the second attempt. A claim
    that fails again keeps its original refusal and the rewrite is
    discarded -- there is no third try, and no partial credit.

    Every attempt is written to the claim's audit record, including
    the refused ones. A rewrite rejected for inventing a number is the
    validator doing its job, and it used to be observable only in a
    log line.
    """
    if not candidates:
        return []

    def refuse(
        record: ClaimJudgment, original: str, rewritten: str | None, guard: str, why: str
    ) -> None:
        record.repair = RepairRecord(
            original_text=original,
            repaired_text=rewritten,
            guard=guard,
            accepted=False,
            reason=why,
        )

    items = [
        (
            index,
            claim.text,
            reason,
            "\n   ".join(f"- {item.quote}" for item in pairs),
        )
        for index, (claim, _verdict, pairs, reason, _record) in enumerate(candidates)
    ]
    try:
        out = (
            await ctx()
            .router.get(ModelRole.CRITIC)
            .structured(RepairOut, REPAIR_SYSTEM, repair_user(contract.question, items))
        )
    except Exception as exc:
        # Repair is optional; its failure refuses the rewrites and
        # leaves every original refusal standing.
        log.warning("wording_repair_failed", error_type=type(exc).__name__, error=str(exc)[:200])
        for _claim, _verdict, _pairs, reason, record in candidates:
            refuse(record, record.claim_text, None, reason, f"repair call failed: {exc}"[:200])
        return []

    healed: list[tuple[Claim, SemanticVerdict, ClaimJudgment]] = []
    attempted = {rewrite.claim_index for rewrite in out.verdicts}
    for index, (_claim, _verdict, _pairs, reason, record) in enumerate(candidates):
        if index not in attempted:
            refuse(record, record.claim_text, None, reason, "no rewrite was returned")
    for rewrite in out.verdicts:
        if not 0 <= rewrite.claim_index < len(candidates):
            continue
        claim, _original, pairs, reason, record = candidates[rewrite.claim_index]
        # Captured before the claim is mutated below. Once claim.text
        # is reassigned the pre-repair wording of a published claim
        # exists nowhere else.
        original = claim.text
        text = (rewrite.rewritten or "").strip()
        if not text:
            refuse(record, original, None, reason, "the rewrite was empty")
            continue

        ok, why = validate_rewrite(claim.text, text)
        if not ok:
            log.info("wording_repair_rejected", reason=why[:120])
            refuse(record, original, text, reason, why)
            continue

        # Every gate again, from the beginning, on the new wording.
        verdict = await asyncio.to_thread(
            verify_claim, text, pairs, scorer, support_threshold=threshold
        )
        if not verdict.publishable:
            log.info("wording_repair_still_refused", reason=verdict.reason[:120])
            refuse(record, original, text, reason, f"still unsupported: {verdict.reason}")
            continue
        relevance = deterministic_relevance(
            text,
            claim.answer_slot or None,
            contract,
            evidence_text=" ".join(item.quote for item in pairs),
        )
        if not relevance.publishable:
            refuse(record, original, text, reason, f"still irrelevant: {relevance.reason}")
            continue

        claim.text = text
        record.repair = RepairRecord(
            original_text=original,
            repaired_text=text,
            guard=reason,
            accepted=True,
        )
        healed.append((claim, verdict, record))
        log.info("wording_repair_accepted", rule=reason[:60])

    log.info("wording_repair", attempted=len(candidates), accepted=len(healed))
    return healed


async def _judge_relevance(
    contract: AnswerContract, claims: list[Claim]
) -> dict[int, tuple[bool | None, str]]:
    """Ask, once, which of these claims answer the question.

    Returns nothing rather than guessing when the call fails, and the
    caller withholds what it cannot get a verdict for.
    """
    try:
        out = (
            await ctx()
            .router.get(ModelRole.CRITIC)
            .structured(
                RelevanceOut,
                RELEVANCE_SYSTEM,
                relevance_user(
                    contract.question,
                    contract.required_slots,
                    [claim.text for claim in claims],
                ),
            )
        )
    except Exception as exc:
        # Returning {} withholds every claim this would have judged,
        # which is the fail-closed behaviour the gate is specified to
        # have. It has to hold for any failure, not just an LLM one.
        log.warning(
            "relevance_judgement_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return {}

    judged: dict[int, tuple[bool | None, str]] = {}
    for verdict in out.verdicts:
        if 0 <= verdict.claim_index < len(claims):
            judged[verdict.claim_index] = (verdict.answers_question, verdict.reason)
    log.info("relevance_judged", asked=len(claims), answered=len(judged))
    return judged


def _scoring_pairs(evidence_ids: list[str], store: EvidenceStore) -> list[CitedEvidence]:
    """Each cited item that may ground a claim, with who published it.

    Unresolvable ids and non-citable quotes are dropped rather than
    scored. A quote that could not be matched to its source cannot
    support anything, and scoring it would let a claim publish on text
    the engine never verified came from the page.

    The source's domain and title travel alongside the quote for the
    attribution guard, which is the only thing that reads them. They do
    not enter the NLI premise -- see verify_claim.
    """
    cited: list[CitedEvidence] = []
    for evidence_id in evidence_ids:
        item = store.evidence_by_id(evidence_id)
        if item is None or not item.is_citable:
            continue
        source = store.source(item.source_id)
        cited.append(
            CitedEvidence(
                evidence_id=evidence_id,
                quote=item.quote,
                # Authority and quality are populated here, and were
                # not. verify_claim ranks equally-entailed quotes by
                # them, so leaving them at their defaults made every
                # source rank UNKNOWN/0.0 and collapsed that ordering
                # to entailment alone -- the selection was implemented,
                # tested against constructed identities, and inert in
                # production. Neither field ever enters the premise.
                source=SourceIdentity(
                    domain=(source.domain if source else "") or "",
                    title=(source.title if source else "") or "",
                    authority=(
                        authority_of(source.source_type.value)
                        if source
                        else SourceAuthority.UNKNOWN
                    ),
                    quality=(source.quality_score if source else 0.0),
                ),
            )
        )
    return cited


def _record(
    text: str,
    evidence_ids: list[str],
    verdict: SemanticVerdict,
    result: CitationVerification,
    record: ClaimJudgment,
) -> None:
    """Write the full semantic detail onto the audit record.

    Everything needed to re-derive the decision without rerunning it:
    which model at which revision, the threshold it was held to, and
    every pairwise score with its guard results.
    """
    record.verdict = verdict.verdict
    record.reason = verdict.reason
    record.checked = verdict.checked
    record.publishable = verdict.publishable
    record.model_id = verdict.model_id
    record.model_revision = verdict.model_revision
    record.support_threshold = verdict.support_threshold
    record.best_evidence_id = verdict.best_evidence_id
    record.best_entailment = round(verdict.best_entailment, 6) if verdict.per_evidence else None
    record.guards_passed = (
        any(s.guards_passed for s in verdict.per_evidence) if verdict.per_evidence else None
    )
    record.propositions = [
        PropositionRecord(
            text=part.text,
            supported=part.supported,
            best_entailment=round(part.best_entailment, 6),
            best_evidence_id=part.best_evidence_id,
        )
        for part in verdict.propositions
    ]
    record.evidence_scores = [
        EvidenceScoreRecord(
            evidence_id=s.evidence_id,
            entailment=round(s.entailment, 6),
            neutral=round(s.neutral, 6),
            contradiction=round(s.contradiction, 6),
            guards_passed=s.guards_passed,
            failed_guards=s.failed_guard_names,
        )
        for s in verdict.per_evidence
    ]


async def _check_contradictions(
    report: ResearchReport,
    store: EvidenceStore,
    result: CitationVerification,
    state: ResearchState,
    verdicts: dict[ClaimKey, ClaimVerdict],
    scorer: Any,
) -> list:
    """Score both sides of every contradiction the same way as a claim.

    A contradiction's two summaries are model-written assertions, and
    they used to reach the published report without any support check:
    structural resolution proved only that evidence existed on each
    side, which is what ``is_auditable`` reports. Prose asserting what a
    source says can be wrong in exactly the ways a claim can.

    Both sides must be publishable for the contradiction to survive.
    Below threshold, guard failure, unchecked or structurally
    unresolvable on either side removes it, and the reason stays in the
    issue list.
    """
    errors: list = []
    if not report.contradictions:
        return errors

    threshold = ctx().settings.nli_support_threshold
    for index, contradiction in enumerate(report.contradictions):
        sides = (
            ("left", contradiction.left_summary, contradiction.left_evidence_ids),
            ("right", contradiction.right_summary, contradiction.right_evidence_ids),
        )
        for side, summary, evidence_ids in sides:
            if not evidence_ids or len(evidence_ids) > MAX_EVIDENCE_PER_CLAIM:
                continue
            result.contradiction_sides_checkable += 1

            pairs = _scoring_pairs(evidence_ids, store)
            if not pairs:
                continue
            verdict = await asyncio.to_thread(
                verify_claim, summary, pairs, scorer, support_threshold=threshold
            )
            if not verdict.checked:
                # Unverified, so the contradiction is dropped rather
                # than published on an unchecked summary.
                continue

            result.contradiction_sides_checked += 1
            verdicts[claim_key(summary, evidence_ids)] = verdict.verdict
            if not verdict.publishable:
                result.issues.append(
                    CitationIssue(
                        type=(
                            CitationIssueType.PARTIALLY_SUPPORTED_CLAIM
                            if verdict.verdict == "partially_supported"
                            else CitationIssueType.UNSUPPORTED_CLAIM
                        ),
                        severity="warning",
                        claim_text=summary[:200],
                        evidence_id=",".join(evidence_ids),
                        detail=f"contradiction {index} {side} side: {verdict.reason[:160]}",
                    )
                )
    return errors


def _pairs_from_state(state: ResearchState) -> tuple[ComparisonPair, ...]:
    """The contrasts the coverage assessment already found."""
    return pairs_from_payload(state.get("answer_coverage"))


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
        report,
        state.get("sources", []),
        verification,
        evidence=state.get("evidence", []),
        # Rebuilt from the assessment carried in state rather than
        # reassessed here: two derivations of one run that can disagree
        # is the defect this project keeps producing.
        comparison_pairs=_pairs_from_state(state),
    )
    reason = stop_reason_for(state)

    log.info("research_completed", stop_reason=reason, sources=len(state.get("sources", [])))
    emit("completed", stop_reason=reason)
    return {"final_markdown": markdown, "stop_reason": reason}
