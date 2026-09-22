"""Planning nodes: understand the question, decompose it, write queries.

All three run once per round at most and are single-writer, which is why they
are the only place identifiers get assigned.
"""

from __future__ import annotations

from agentic_research.config import ModelRole
from agentic_research.graph.nodes.common import ctx, emit, error_from, numbered, stage
from agentic_research.graph.prompts import (
    ANALYST_SYSTEM,
    FOLLOWUP_SYSTEM,
    PLANNER_SYSTEM,
    QUERY_WRITER_SYSTEM,
    analyst_user,
    followup_user,
    planner_user,
    query_writer_user,
)
from agentic_research.graph.state import ResearchState
from agentic_research.llm.base import LLMError
from agentic_research.models import (
    OutputFormat,
    QueryAnalysis,
    ResearchPlan,
    SearchQuery,
    SubQuestion,
)
from agentic_research.observability import get_logger
from agentic_research.schemas import AnalysisOut, FollowupsOut, PlanOut, QueriesOut

log = get_logger(__name__)


async def analyze_query(state: ResearchState) -> ResearchState:
    """Turn the raw question into a structured brief.

    Degrades rather than fails: if the model cannot produce a valid analysis,
    the run continues with a minimal one built from the raw question. Losing
    intent detection costs some quality; aborting costs the user the run.
    """
    query = state["original_query"]
    emit("analyzing_query", query=query)

    with stage("analyze_query") as timing:
        try:
            out = (
                await ctx()
                .router.get(ModelRole.PLANNER)
                .structured(AnalysisOut, ANALYST_SYSTEM, analyst_user(query))
            )
            analysis = QueryAnalysis(
                original_query=query,
                normalized_query=out.normalized_query or query,
                intent=out.intent,
                entities=out.entities[:12],
                constraints=out.constraints[:8],
                output_format=OutputFormat(out.output_format),
                time_sensitive=out.time_sensitive,
                recency_horizon_months=out.recency_horizon_months if out.time_sensitive else None,
                requires_web_research=out.requires_web_research,
            )
            errors = []
        except LLMError as exc:
            log.warning("analysis_failed_using_fallback", error=str(exc)[:200])
            analysis = QueryAnalysis(
                original_query=query, normalized_query=query, intent="unclassified"
            )
            errors = [error_from("analyze_query", exc, "fell back to raw question")]

    log.info(
        "query_analyzed",
        intent=analysis.intent[:80],
        format=analysis.output_format.value,
        time_sensitive=analysis.time_sensitive,
    )
    emit("query_analyzed", intent=analysis.intent, format=analysis.output_format.value)
    return {"analysis": analysis, "stage_timings": [timing], "errors": errors}


def _analysis_block(analysis: QueryAnalysis) -> str:
    lines = [f"Question: {analysis.normalized_query}", f"Intent: {analysis.intent}"]
    if analysis.entities:
        lines.append("Key entities: " + ", ".join(analysis.entities))
    if analysis.constraints:
        lines.append("Constraints: " + "; ".join(analysis.constraints))
    if analysis.time_sensitive:
        lines.append(
            f"Time sensitive: prefer sources from the last "
            f"{analysis.recency_horizon_months or 24} months"
        )
    lines.append(f"Expected answer shape: {analysis.output_format.value}")
    return "\n".join(lines)


async def plan_research(state: ResearchState) -> ResearchState:
    """Decompose the question into sub-questions.

    Falls back to treating the original question as a single dimension. That
    is a worse plan, but it still produces a researchable run.
    """
    analysis = state.get("analysis") or QueryAnalysis(
        original_query=state["original_query"],
        normalized_query=state["original_query"],
        intent="unclassified",
    )
    emit("planning")

    with stage("plan_research") as timing:
        errors = []
        try:
            out = (
                await ctx()
                .router.get(ModelRole.PLANNER)
                .structured(PlanOut, PLANNER_SYSTEM, planner_user(_analysis_block(analysis)))
            )
            raw = out.sub_questions[:8]
            strategy = out.strategy_note
        except LLMError as exc:
            log.warning("planning_failed_using_single_dimension", error=str(exc)[:200])
            raw = []
            strategy = "planning failed; researching the question as a single dimension"
            errors = [error_from("plan_research", exc, "single-dimension fallback")]

        ids = numbered("SQ", 1, len(raw))
        sub_questions = [
            SubQuestion(
                id=sq_id,
                text=item.text,
                rationale=item.rationale,
                priority=min(max(item.priority, 1), 3),
                round_introduced=1,
            )
            for sq_id, item in zip(ids, raw, strict=True)
        ]
        if not sub_questions:
            sub_questions = [
                SubQuestion(
                    id="SQ1",
                    text=analysis.normalized_query,
                    rationale="fallback: the question itself",
                    priority=1,
                )
            ]

        plan = ResearchPlan(analysis=analysis, sub_questions=sub_questions, strategy_note=strategy)

    log.info("plan_generated", sub_questions=len(sub_questions))
    emit(
        "plan_generated",
        count=len(sub_questions),
        questions=[q.text for q in sub_questions],
    )
    return {
        "plan": plan,
        "sub_questions": sub_questions,
        "stage_timings": [timing],
        "errors": errors,
    }


def _sub_question_block(sub_questions: list[SubQuestion]) -> str:
    return "\n".join(f"{q.id}: {q.text}" for q in sub_questions)


async def generate_queries(state: ResearchState) -> ResearchState:
    """Write this round's search queries.

    Enforces the query budget here rather than at dispatch, because this is
    where the count is known and can be trimmed before anything is spent.
    """
    context = ctx()
    round_number = state.get("round_number", 0) + 1
    completed = state.get("completed_queries", [])
    all_sub_questions = state.get("sub_questions", [])

    # In a follow-up round only the newly added sub-questions need queries;
    # the earlier ones have already been searched.
    targets = (
        [q for q in all_sub_questions if q.round_introduced == round_number]
        if round_number > 1
        else all_sub_questions
    )
    if not targets:
        targets = all_sub_questions

    remaining_budget = context.budget.max_search_queries - len(completed)
    if remaining_budget <= 0:
        log.warning("query_budget_exhausted", issued=len(completed))
        return {
            "pending_queries": [],
            "round_number": round_number,
            "stop_reason": "query budget exhausted",
        }

    emit("generating_queries", round=round_number, sub_questions=len(targets))
    with stage("generate_queries") as timing:
        errors = []
        try:
            out = await context.router.get(ModelRole.RESEARCHER).structured(
                QueriesOut,
                QUERY_WRITER_SYSTEM,
                query_writer_user(_sub_question_block(targets), [q.text for q in completed]),
            )
            proposed = [(q.sub_question_id, q.text) for q in out.queries]
        except LLMError as exc:
            log.warning("query_generation_failed_using_text", error=str(exc)[:200])
            # The sub-question text is a serviceable query on its own.
            proposed = [(q.id, q.text) for q in targets]
            errors = [error_from("generate_queries", exc, "using sub-question text")]

        valid_ids = {q.id for q in targets}
        seen = {q.text.strip().lower() for q in completed}
        queries: list[SearchQuery] = []
        next_index = len(completed) + 1

        for sub_question_id, text in proposed:
            text = text.strip()
            key = text.lower()
            if not text or key in seen:
                continue
            # A model that invents a sub-question id would break provenance,
            # so unknown ids are reassigned rather than trusted.
            owner = sub_question_id if sub_question_id in valid_ids else targets[0].id
            seen.add(key)
            queries.append(
                SearchQuery(
                    id=f"Q{next_index}",
                    sub_question_id=owner,
                    text=text,
                    round_number=round_number,
                )
            )
            next_index += 1
            if len(queries) >= remaining_budget:
                break

        if not queries:
            queries = [
                SearchQuery(
                    id=f"Q{next_index + i}",
                    sub_question_id=q.id,
                    text=q.text,
                    round_number=round_number,
                )
                for i, q in enumerate(targets[:remaining_budget])
            ]

    log.info("queries_generated", round=round_number, count=len(queries))
    emit(
        "queries_generated",
        round=round_number,
        count=len(queries),
        queries=[q.text for q in queries],
    )
    return {
        "pending_queries": queries,
        "completed_queries": queries,
        "round_number": round_number,
        "stage_timings": [timing],
        "errors": errors,
    }


async def generate_followups(state: ResearchState) -> ResearchState:
    """Turn coverage gaps into new sub-questions for another round."""
    coverage = state.get("coverage")
    existing = state.get("sub_questions", [])
    round_number = state.get("round_number", 1)
    gaps = []
    if coverage:
        gaps = list(coverage.missing) + [f"weak evidence for {sq_id}" for sq_id in coverage.weak]
    if not gaps:
        return {"stop_reason": "no actionable gaps"}

    emit("generating_followups", gaps=len(gaps))
    with stage("generate_followups") as timing:
        errors = []
        try:
            out = (
                await ctx()
                .router.get(ModelRole.PLANNER)
                .structured(
                    FollowupsOut,
                    FOLLOWUP_SYSTEM,
                    followup_user(_sub_question_block(existing), "\n".join(f"- {g}" for g in gaps)),
                )
            )
            items = out.followups[:4]
        except LLMError as exc:
            log.warning("followup_generation_failed", error=str(exc)[:200])
            items = []
            errors = [error_from("generate_followups", exc)]

        start = len(existing) + 1
        followups = [
            SubQuestion(
                id=f"SQ{start + i}",
                text=item.text,
                rationale=f"follow-up: {item.gap}",
                priority=1,
                round_introduced=round_number + 1,
                is_followup=True,
                parent_gap=item.gap,
            )
            for i, item in enumerate(items)
            if item.text.strip()
        ]

    if not followups:
        return {"stop_reason": "no follow-up questions produced", "errors": errors}

    log.info("followups_generated", count=len(followups), round=round_number + 1)
    emit("followups_generated", count=len(followups), questions=[f.text for f in followups])
    return {"sub_questions": followups, "stage_timings": [timing], "errors": errors}
