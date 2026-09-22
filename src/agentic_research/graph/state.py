"""Graph state, reducers and runtime context.

Two containers, split by a rule worth stating plainly:

* **State** is data the graph produces. It is a ``TypedDict`` of plain values
  and Pydantic models, and it is what gets checkpointed. Everything in it must
  survive a round trip through JSON.
* **Context** is the live machinery a run needs — open HTTP clients, the model
  router, semaphores. None of it is serialisable and none of it belongs in a
  checkpoint. LangGraph 1.x provides a typed runtime context for exactly this,
  reached from a node with ``get_runtime(RunContext)``.

State is a ``TypedDict`` rather than a Pydantic model because LangGraph merges
*partial* updates through per-channel reducers. A node returns only the keys it
touched. That is the mechanism that lets several parallel workers write to the
same channel in one super-step without overwriting each other.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from agentic_research.config import RunBudget, Settings
from agentic_research.evidence.dedup import Candidate
from agentic_research.llm.router import ModelRouter
from agentic_research.models import (
    CoverageAssessment,
    EvidenceItem,
    QueryAnalysis,
    ResearchPlan,
    ResearchReport,
    RunError,
    SearchQuery,
    SearchResult,
    SourceDocument,
    SubQuestion,
)
from agentic_research.retrieval.fetcher import PageFetcher
from agentic_research.search.service import SearchService

# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------


def merge_sources(
    existing: list[SourceDocument], incoming: list[SourceDocument]
) -> list[SourceDocument]:
    """Merge source lists by id, letting later writes win.

    A source is created as a stub when its URL is deduplicated, then filled in
    with text by whichever fetch worker handled it. Both writes land on the
    same channel, so plain concatenation would leave two copies of every
    source and break citation lookup. Insertion order is preserved because the
    ids are ordinal (S1, S2, ...) and a reordered source list would make the
    report's citation markers appear to jump around.
    """
    merged: dict[str, SourceDocument] = {s.id: s for s in existing}
    order: list[str] = [s.id for s in existing]
    for source in incoming:
        if source.id not in merged:
            order.append(source.id)
        merged[source.id] = source
    return [merged[sid] for sid in order]


def merge_sub_questions(
    existing: list[SubQuestion], incoming: list[SubQuestion]
) -> list[SubQuestion]:
    """Append new sub-questions, ignoring ids already present.

    Follow-up rounds add to this list. Deduplicating by id keeps a retried
    node from producing a second copy of the same question.
    """
    seen = {q.id for q in existing}
    return existing + [q for q in incoming if q.id not in seen]


def sum_counters(existing: dict[str, int], incoming: dict[str, int]) -> dict[str, int]:
    """Add counter dictionaries key-wise.

    Parallel workers each report their own tallies (pages fetched, provider
    content reused); summing is the only merge that makes sense.
    """
    merged = dict(existing)
    for key, value in incoming.items():
        merged[key] = merged.get(key, 0) + value
    return merged


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ResearchState(TypedDict, total=False):
    """Everything one research run accumulates.

    Channels carrying an ``Annotated[..., reducer]`` are written by more than
    one node, or by parallel copies of one node. Channels without a reducer are
    written by exactly one node and are replaced outright.
    """

    # --- identity -----------------------------------------------------
    run_id: str
    original_query: str

    # --- planning (single writer each) --------------------------------
    analysis: QueryAnalysis | None
    plan: ResearchPlan | None
    sub_questions: Annotated[list[SubQuestion], merge_sub_questions]

    # --- query lifecycle ----------------------------------------------
    pending_queries: list[SearchQuery]
    """This round's dispatch list. Replaced wholesale each round."""
    completed_queries: Annotated[list[SearchQuery], operator.add]
    """Every query ever issued. Used for budget checks and to stop the
    follow-up rounds from re-asking something already asked."""

    # --- fan-in channels ----------------------------------------------
    round_results: Annotated[list[SearchResult], operator.add]
    """Written concurrently by the parallel search workers, then cleared by
    the deduplication barrier via Overwrite."""
    sources: Annotated[list[SourceDocument], merge_sources]
    evidence: Annotated[list[EvidenceItem], operator.add]

    # --- work queues between stages (single writer each) --------------
    fetch_candidates: list[Candidate]
    pending_extraction: list[str]

    # --- loop control --------------------------------------------------
    round_number: int
    coverage: CoverageAssessment | None
    coverage_history: Annotated[list[CoverageAssessment], operator.add]
    stop_reason: str

    # --- output ---------------------------------------------------------
    report: ResearchReport | None
    verification: dict[str, Any] | None
    final_markdown: str

    # --- diagnostics ------------------------------------------------------
    errors: Annotated[list[RunError], operator.add]
    """Non-fatal failures. Parallel workers cannot use the graph's node-level
    error handler (it does not fire for Send-dispatched tasks), so they record
    failures here and the run continues degraded."""
    stage_timings: Annotated[list[dict[str, Any]], operator.add]
    counters: Annotated[dict[str, int], sum_counters]


# ---------------------------------------------------------------------------
# Worker payloads
# ---------------------------------------------------------------------------
#
# A Send-dispatched worker does not receive the graph state; it receives the
# payload the dispatcher handed it. Declaring those shapes explicitly (and
# passing them as `input_schema` when the node is registered) keeps the
# worker signatures honest and lets the type checker catch a dispatcher that
# sends the wrong thing.


class SearchTask(TypedDict):
    query: SearchQuery


class FetchTask(TypedDict):
    source: SourceDocument
    candidate: Candidate


class ExtractTask(TypedDict):
    source: SourceDocument
    sub_questions: list[SubQuestion]


# ---------------------------------------------------------------------------
# Runtime context
# ---------------------------------------------------------------------------


ProgressFn = Callable[[str, dict[str, Any]], None]


@dataclass
class RunContext:
    """Live objects shared by every node in one run.

    Held here rather than in state for two reasons: an ``httpx.AsyncClient``
    cannot be checkpointed, and the concurrency semaphores must be created
    inside the running event loop. A module-level semaphore would bind to
    whichever loop happened to import it first and would silently leak its
    limit across runs.
    """

    settings: Settings
    router: ModelRouter
    search: SearchService
    fetcher: PageFetcher
    budget: RunBudget
    run_id: str
    started_at: float = 0.0
    warnings: list[str] = field(default_factory=list)


def initial_state(run_id: str, query: str) -> ResearchState:
    """Seed state. Every accumulating channel starts as an empty container."""
    return ResearchState(
        run_id=run_id,
        original_query=query,
        analysis=None,
        plan=None,
        sub_questions=[],
        pending_queries=[],
        completed_queries=[],
        round_results=[],
        sources=[],
        evidence=[],
        fetch_candidates=[],
        pending_extraction=[],
        round_number=0,
        coverage=None,
        coverage_history=[],
        stop_reason="",
        report=None,
        verification=None,
        final_markdown="",
        errors=[],
        stage_timings=[],
        counters={},
    )
