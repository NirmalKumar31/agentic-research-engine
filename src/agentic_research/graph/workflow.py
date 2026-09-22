"""Graph assembly.

The shape of the research process, in one place::

    analyze_query -> plan_research -> generate_queries
        -> [fan-out] search_worker xN
        -> dedupe_sources          (defer: waits for every search)
        -> [fan-out] fetch_worker xM
        -> register_sources        (defer: waits for every fetch)
        -> [fan-out] extract_worker xM
        -> assess_coverage         (defer: waits for every extraction)
        -> gaps?  yes: generate_followups -> generate_queries  (bounded loop)
                  no:  synthesize -> verify_citations -> finalize -> END

Three design points worth stating:

**Why fan out per stage rather than per researcher.** A worker that did
search-then-fetch-then-extract could not see its siblings' results, so a page
surfaced by four sub-questions would be downloaded and sent to a model four
times. The barriers between stages exist so deduplication can happen in the
middle, which turns that into one download and one model call.

**Why ``defer=True`` instead of a counter.** Marking a node deferred tells
LangGraph to run it only once every task that writes to it has settled. Doing
this manually — counting completions in state and re-checking on each pass —
is the classic reimplementation of a barrier, with the classic race.

**Why workers catch their own exceptions.** ``add_node(error_handler=...)``
does not fire for ``Send``-dispatched tasks in LangGraph 1.2.x. Verified, not
assumed. An escaping exception therefore kills the whole super-step, so each
worker records failures into the ``errors`` channel instead.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from agentic_research.graph.nodes.critique import assess_coverage
from agentic_research.graph.nodes.planning import (
    analyze_query,
    generate_followups,
    generate_queries,
    plan_research,
)
from agentic_research.graph.nodes.reporting import (
    finalize,
    synthesize_report,
    verify_citations,
)
from agentic_research.graph.nodes.research import (
    dedupe_sources,
    extract_worker,
    fetch_worker,
    register_sources,
    search_worker,
)
from agentic_research.graph.routing import (
    dispatch_extraction,
    dispatch_fetches,
    dispatch_searches,
    route_after_coverage,
    route_after_followups,
)
from agentic_research.graph.state import (
    ExtractTask,
    FetchTask,
    ResearchState,
    RunContext,
    SearchTask,
)

# Transport hiccups and rate limits are worth one more attempt; a schema
# violation is not, because the structured-output layer already retried it
# with the validation error fed back in.
_TRANSIENT_RETRY = RetryPolicy(
    max_attempts=2,
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=8.0,
    jitter=True,
)


def build_graph() -> StateGraph[ResearchState, RunContext, ResearchState, ResearchState]:
    """Construct the research graph (uncompiled, so tests can inspect it)."""
    graph = StateGraph(ResearchState, context_schema=RunContext)

    # --- planning ---------------------------------------------------------
    graph.add_node("analyze_query", analyze_query, retry_policy=_TRANSIENT_RETRY)
    graph.add_node("plan_research", plan_research, retry_policy=_TRANSIENT_RETRY)
    graph.add_node("generate_queries", generate_queries, retry_policy=_TRANSIENT_RETRY)
    graph.add_node("generate_followups", generate_followups, retry_policy=_TRANSIENT_RETRY)

    # --- parallel research ------------------------------------------------
    # No retry_policy on the workers: they already handle their own failures
    # and returning a recorded error is a success from the graph's point of
    # view, so a retry here would never trigger.
    graph.add_node("search_worker", search_worker, input_schema=SearchTask)
    graph.add_node("fetch_worker", fetch_worker, input_schema=FetchTask)
    graph.add_node("extract_worker", extract_worker, input_schema=ExtractTask)

    # --- barriers ---------------------------------------------------------
    graph.add_node("dedupe_sources", dedupe_sources, defer=True)
    graph.add_node("register_sources", register_sources, defer=True)
    graph.add_node("assess_coverage", assess_coverage, defer=True)

    # --- reporting --------------------------------------------------------
    graph.add_node("synthesize", synthesize_report, retry_policy=_TRANSIENT_RETRY)
    graph.add_node("verify_citations", verify_citations, retry_policy=_TRANSIENT_RETRY)
    graph.add_node("finalize", finalize)

    # --- edges ------------------------------------------------------------
    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "plan_research")
    graph.add_edge("plan_research", "generate_queries")

    # Each dispatcher lists its fallback target as a possible destination so
    # an empty round routes forward instead of ending the graph.
    graph.add_conditional_edges(
        "generate_queries", dispatch_searches, ["search_worker", "assess_coverage"]
    )
    graph.add_edge("search_worker", "dedupe_sources")

    graph.add_conditional_edges(
        "dedupe_sources", dispatch_fetches, ["fetch_worker", "register_sources"]
    )
    graph.add_edge("fetch_worker", "register_sources")

    graph.add_conditional_edges(
        "register_sources", dispatch_extraction, ["extract_worker", "assess_coverage"]
    )
    graph.add_edge("extract_worker", "assess_coverage")

    # The loop, and the only back-edge in the graph.
    graph.add_conditional_edges(
        "assess_coverage", route_after_coverage, ["generate_followups", "synthesize"]
    )
    graph.add_conditional_edges(
        "generate_followups", route_after_followups, ["generate_queries", "synthesize"]
    )

    graph.add_edge("synthesize", "verify_citations")
    graph.add_edge("verify_citations", "finalize")
    graph.add_edge("finalize", END)

    return graph


def compile_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[ResearchState, RunContext, ResearchState, ResearchState]:
    """Compile the graph, optionally with a checkpointer."""
    return build_graph().compile(checkpointer=checkpointer)
