"""Conditional edges: fan-out dispatchers and the research loop decision.

Two things in here are easy to get wrong and are therefore deliberate.

**Empty dispatch.** A conditional edge that returns an empty list of ``Send``
objects does not fall through to the next node — LangGraph schedules nothing,
the downstream node never runs, and the graph simply ends. Verified against
1.2.x. Every dispatcher below returns an explicit fallback node name when it
has nothing to send, because the alternative is a research round that
silently evaporates.

**Loop termination.** The loop back-edge is guarded by four independent
conditions, any one of which ends research. Termination does not depend on
the critic choosing to be satisfied; a model that always asks for more would
still be stopped by the round counter and the budgets.
"""

from __future__ import annotations

from langgraph.types import Send

from agentic_research.graph.state import ResearchState
from agentic_research.observability import get_logger

log = get_logger(__name__)


def dispatch_searches(state: ResearchState) -> list[Send] | str:
    """Fan out one search worker per pending query."""
    queries = state.get("pending_queries", [])
    if not queries:
        log.info("no_queries_to_dispatch", reason=state.get("stop_reason", "none generated"))
        return "assess_coverage"
    log.info("search_dispatched", count=len(queries), round=state.get("round_number", 1))
    return [Send("search_worker", {"query": q}) for q in queries]


def dispatch_fetches(state: ResearchState) -> list[Send] | str:
    """Fan out one fetch worker per unique source selected this round."""
    candidates = state.get("fetch_candidates", [])
    if not candidates:
        return "register_sources"

    # Pair each candidate with the stub created for it in the dedup barrier.
    by_url = {s.canonical_url: s for s in state.get("sources", [])}
    sends = [
        Send("fetch_worker", {"source": by_url[c.canonical_url], "candidate": c})
        for c in candidates
        if c.canonical_url in by_url
    ]
    if not sends:
        return "register_sources"
    log.info("fetch_dispatched", count=len(sends))
    return sends


def dispatch_extraction(state: ResearchState) -> list[Send] | str:
    """Fan out one extraction worker per source worth reading."""
    pending = state.get("pending_extraction", [])
    if not pending:
        return "assess_coverage"

    by_id = {s.id: s for s in state.get("sources", [])}
    # Only sub-questions still lacking solid evidence are put in front of the
    # extractor. Including everything would grow the prompt on every round
    # while adding nothing for questions already answered.
    coverage = state.get("coverage")
    open_ids: set[str] | None = None
    if coverage is not None:
        open_ids = set(coverage.weak) | {
            c.sub_question_id for c in coverage.per_question if c.verdict != "covered"
        }
    sub_questions = [
        q
        for q in state.get("sub_questions", [])
        if open_ids is None or q.id in open_ids or q.is_followup
    ] or state.get("sub_questions", [])

    if not sub_questions:
        return "assess_coverage"

    sends = [
        Send("extract_worker", {"source": by_id[sid], "sub_questions": sub_questions})
        for sid in pending
        if sid in by_id
    ]
    if not sends:
        return "assess_coverage"
    log.info("extraction_dispatched", sources=len(sends), sub_questions=len(sub_questions))
    return sends


def route_after_coverage(state: ResearchState) -> str:
    """Decide between another research round and writing the report.

    Returns the name of the next node. Any single stop condition ends the
    loop; the report is still produced, with its limitations recorded.
    """
    from agentic_research.graph.nodes.common import ctx

    coverage = state.get("coverage")
    round_number = state.get("round_number", 1)
    budget = ctx().budget

    if coverage is not None and coverage.sufficient:
        log.info("research_loop_exit", reason="coverage sufficient", round=round_number)
        return "synthesize"

    if round_number >= budget.max_research_rounds:
        log.info("research_loop_exit", reason="max rounds reached", round=round_number)
        return "synthesize"

    if len(state.get("completed_queries", [])) >= budget.max_search_queries:
        log.info("research_loop_exit", reason="query budget spent", round=round_number)
        return "synthesize"

    if len(state.get("sources", [])) >= budget.max_sources:
        log.info("research_loop_exit", reason="source budget spent", round=round_number)
        return "synthesize"

    if coverage is not None and not coverage.recommended_followups and not coverage.missing:
        log.info("research_loop_exit", reason="no actionable gaps", round=round_number)
        return "synthesize"

    log.info(
        "research_loop_triggered",
        round=round_number + 1,
        ratio=coverage.coverage_ratio if coverage else 0.0,
    )
    return "generate_followups"


def route_after_followups(state: ResearchState) -> str:
    """Only start another round if the follow-up step actually produced work."""
    round_number = state.get("round_number", 1)
    new_questions = [q for q in state.get("sub_questions", []) if q.round_introduced > round_number]
    if not new_questions:
        log.info("research_loop_exit", reason="no follow-up questions produced")
        return "synthesize"
    return "generate_queries"


def stop_reason_for(state: ResearchState) -> str:
    """Human-readable explanation of why research ended, for the report."""
    if state.get("stop_reason"):
        return str(state["stop_reason"])
    coverage = state.get("coverage")
    if coverage is not None and coverage.sufficient:
        return "coverage judged sufficient"
    return f"stopped after round {state.get('round_number', 1)}"
