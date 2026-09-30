"""Coverage assessment — the node that decides whether to research again.

The assessment is deliberately split in two:

* **Counted mechanically** — how many verified evidence items each
  sub-question has, across how many distinct sources, whether any source
  contradicts another, how concentrated the domains are. These are facts
  about the evidence set and need no model.
* **Judged by a model** — whether the evidence is actually on target, which
  angle nobody looked at, what the disagreements are about.

Asking a model for "coverage: 0.72" would produce a number with no defined
meaning that nonetheless looks authoritative. The ratio reported here has a
definition: the share of sub-questions with at least two verified evidence
items drawn from at least two distinct sources.
"""

from __future__ import annotations

from agentic_research.config import ModelRole
from agentic_research.evidence.quality import domain_concentration
from agentic_research.evidence.store import EvidenceStore
from agentic_research.graph.nodes.common import ctx, emit, error_from, stage
from agentic_research.graph.prompts import CRITIC_SYSTEM, critic_user
from agentic_research.graph.state import ResearchState
from agentic_research.models import (
    CoverageAssessment,
    FetchStatus,
    Stance,
    SubQuestionCoverage,
)
from agentic_research.observability import get_logger
from agentic_research.schemas import CoverageOut

log = get_logger(__name__)

# A run is allowed to proceed to synthesis at or above this ratio. Set below
# 1.0 on purpose: insisting every sub-question be fully covered would spend
# rounds chasing the one dimension the web has little to say about, and the
# report can state that gap instead.
_SUFFICIENT_RATIO = 0.7

# Above this share from one domain, independence is worth warning about.
_CONCENTRATION_WARNING = 0.6


def _attribute_gaps(
    per_question: list[SubQuestionCoverage], state: ResearchState
) -> list[SubQuestionCoverage]:
    """Name the retrieval-side cause of a gap, where there is one.

    Ordered most-specific first. "No suitable source found" is a
    different problem from "the source we chose would not load", and
    both are different from "the source loaded and was about something
    else" -- which is the one `coverage_for` can see for itself.
    """
    diagnostics = state.get("retrieval_diagnostics") or {}
    starved = set(diagnostics.get("starved_sub_questions") or ())
    by_sub_question = diagnostics.get("by_sub_question") or {}

    failed_fetches = [
        s for s in state.get("sources", []) or [] if s.fetch_status is not FetchStatus.OK
    ]

    out: list[SubQuestionCoverage] = []
    for coverage in per_question:
        if coverage.verdict == "covered":
            out.append(coverage)
            continue
        cause = coverage.gap_cause
        sq_id = coverage.sub_question_id
        if sq_id in starved or not by_sub_question.get(sq_id):
            cause = "no suitable source found"
        elif failed_fetches and coverage.evidence_count == 0:
            # A source was chosen for this sub-question and did not
            # load, so the gap is retryable rather than a dead end.
            cause = "candidate selected but fetch failed"
        out.append(coverage.model_copy(update={"gap_cause": cause}))
    return out


async def assess_coverage(state: ResearchState) -> ResearchState:
    """Score coverage and decide whether the evidence can answer the question."""
    sub_questions = state.get("sub_questions", [])
    sources = state.get("sources", [])
    evidence = state.get("evidence", [])
    round_number = state.get("round_number", 1)
    analysis = state.get("analysis")
    question = analysis.normalized_query if analysis else state["original_query"]

    emit("assessing_coverage", round=round_number)
    store = EvidenceStore(sources, evidence)

    with stage("assess_coverage") as timing:
        per_question = [store.coverage_for(q) for q in sub_questions]
        # A gap with no diagnosis is indistinguishable from a gap with a
        # different cause, and the four causes call for four different
        # fixes: a better query, a retryable fetch, a better source, or
        # more evidence. `coverage_for` can only see the evidence it
        # was given, so the retrieval-side causes are attributed here,
        # where the selection diagnostics and fetch statuses are.
        per_question = _attribute_gaps(per_question, state)
        covered = [c.sub_question_id for c in per_question if c.verdict == "covered"]
        weak = [c.sub_question_id for c in per_question if c.verdict == "weak"]
        missing = [c.sub_question_id for c in per_question if c.verdict == "uncovered"]
        ratio = round(len(covered) / len(per_question), 4) if per_question else 0.0
        concentration = domain_concentration(store.domains())

        contradictions = [
            f"[{e.source_id}] {e.claim}"
            for e in evidence
            if e.stance is Stance.CONTRADICTS and e.quote_verified
        ][:8]

        reasoning = ""
        errors = []
        llm_weak: list[str] = []
        llm_missing: list[str] = []
        try:
            counts_block = "\n".join(
                f"{c.sub_question_id}: {c.verdict} ({c.note})" for c in per_question
            )
            package = store.build_package(sub_questions, max_items_per_question=4)
            out = (
                await ctx()
                .router.get(ModelRole.CRITIC)
                .structured(
                    CoverageOut,
                    CRITIC_SYSTEM,
                    critic_user(question, counts_block, package.text or "(no evidence yet)"),
                )
            )
            valid = {q.id for q in sub_questions}
            llm_weak = [sq_id for sq_id in out.weak_sub_question_ids if sq_id in valid]
            llm_missing = list(out.missing_angles)
            contradictions = list(dict.fromkeys(contradictions + list(out.contradictions)))[:10]
            reasoning = out.reasoning
        except Exception as exc:
            # Any exception, not only LLMError.
            #
            # This call is advice. Every number that routes the run --
            # the per-question verdicts, the ratio, the domain
            # concentration -- is computed above it from evidence
            # already in hand, and the critic only adds to the weak
            # list and names gaps. Letting a failure here end the run
            # throws away four searches and twenty-eight extracted
            # quotes that were already paid for, to lose an opinion.
            #
            # A hosted run died exactly here: it reached
            # assessing_coverage with 28 items and emitted an error
            # instead of a report. LLMError was caught and everything
            # else was not, which made the narrow catch a promise the
            # node did not keep.
            #
            # Not silent. The exception type reaches the run's error
            # list and its metrics, so a bug here shows up as a
            # degraded run rather than as nothing -- swallowing it
            # would trade a lost run for a hidden defect.
            log.warning(
                "coverage_critique_failed",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            reasoning = "critique unavailable; using mechanical coverage counts only"
            errors = [error_from("assess_coverage", exc, "counts-only assessment")]

        # A model can add to the weak list but cannot promote a sub-question
        # the counts show as uncovered.
        weak = list(dict.fromkeys(weak + [w for w in llm_weak if w not in missing]))
        covered = [sq_id for sq_id in covered if sq_id not in weak]

        # Recomputed, because the demotions above changed what `covered`
        # means. The first value was derived from the mechanical counts
        # alone; deciding sufficiency on it credited coverage the critic
        # had just withdrawn. A run whose every sub-question was demoted
        # to weak reported "coverage judged sufficient" beside 0/5.
        ratio = round(len(covered) / len(per_question), 4) if per_question else 0.0

        recommended = [f"gap: {m}" for m in llm_missing[:4]] + [
            f"weak coverage for {sq_id}" for sq_id in weak[:3]
        ]
        sufficient = ratio >= _SUFFICIENT_RATIO and not missing and bool(evidence)
        if concentration > _CONCENTRATION_WARNING and len(store.usable_sources()) > 2:
            recommended.append(f"source diversity: {concentration:.0%} of sources share one domain")

        assessment = CoverageAssessment(
            round_number=round_number,
            per_question=per_question,
            coverage_ratio=ratio,
            covered=covered,
            weak=weak,
            missing=missing + llm_missing,
            contradictions=contradictions,
            domain_concentration=concentration,
            recommended_followups=recommended,
            sufficient=sufficient,
            reasoning=reasoning,
        )
        timing["coverage_ratio"] = ratio

    log.info(
        "coverage_evaluated",
        round=round_number,
        ratio=ratio,
        covered=len(covered),
        weak=len(weak),
        missing=len(missing),
        sufficient=sufficient,
        domain_concentration=concentration,
    )
    emit(
        "coverage_evaluated",
        round=round_number,
        ratio=ratio,
        sufficient=sufficient,
        covered=len(covered),
        # Emitted explicitly rather than left to be derived. A client
        # reconstructing the denominator as covered/ratio cannot do it when
        # nothing is covered, and rendered "0/0 covered" for a run with six
        # research dimensions.
        total=len(covered) + len(weak) + len(missing),
        weak=len(weak),
        missing=len(missing),
    )
    return {
        "coverage": assessment,
        "coverage_history": [assessment],
        "stage_timings": [timing],
        "errors": errors,
    }
