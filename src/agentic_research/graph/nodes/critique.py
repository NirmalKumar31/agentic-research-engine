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
from agentic_research.llm.base import LLMError
from agentic_research.models import CoverageAssessment, Stance
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
        except LLMError as exc:
            # The counted half still stands, so routing remains sound.
            log.warning("coverage_critique_failed", error=str(exc)[:200])
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
