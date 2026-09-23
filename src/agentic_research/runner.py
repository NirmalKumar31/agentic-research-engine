"""Run orchestration.

Owns everything around a single research run that is not the graph itself:
building the runtime context, opening and closing the HTTP clients, choosing a
checkpointer, streaming progress to a callback, assembling metrics and writing
artifacts.

Keeping this out of the graph means the graph file stays a readable
description of the research process, and means the CLI, the Streamlit app and
the evaluation harness all drive runs through one code path.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig

from agentic_research.config import Settings
from agentic_research.environment import capture as capture_environment
from agentic_research.graph.state import RunContext, initial_state
from agentic_research.graph.workflow import compile_graph
from agentic_research.llm.base import UsageTracker
from agentic_research.llm.router import ModelRouter
from agentic_research.metrics import RunMetrics, build_metrics
from agentic_research.models import CitationVerification
from agentic_research.observability import bind_run, clear_run, get_logger
from agentic_research.report import render_markdown
from agentic_research.retrieval.fetcher import PageFetcher
from agentic_research.search.service import SearchService, build_provider

log = get_logger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class RunResult:
    run_id: str
    markdown: str
    metrics: RunMetrics
    state: dict[str, Any]
    output_dir: Path | None = None
    warnings: list[str] | None = None


def new_run_id() -> str:
    """Sortable, human-readable, unique enough for a local tool.

    Time-prefixed so ``ls outputs/`` is chronological without extra tooling.
    """
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


async def _make_checkpointer(settings: Settings, stack: AsyncExitStack) -> Any:
    """Choose a checkpointer.

    SQLite by default and nothing heavier. A research run is a single-process,
    minutes-long job whose state is a few hundred kilobytes; Postgres would
    add an operational dependency purely for appearances. In-memory is used by
    tests, and 'none' disables checkpointing entirely for throwaway runs.
    """
    backend = settings.checkpoint_backend
    if backend == "none":
        return None
    if backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return await stack.enter_async_context(
        AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_path))
    )


async def stream_research(
    query: str,
    settings: Settings,
    *,
    run_id: str | None = None,
    exhaustive_verification: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Run research, yielding progress events as they happen.

    The final event is ``{"event": "result", "result": RunResult}``. Streaming
    rather than returning lets a caller show progress during a run that
    legitimately takes minutes, without the runner knowing anything about how
    it will be displayed.
    """
    run_id = run_id or new_run_id()
    bind_run(run_id)
    started = time.perf_counter()

    router = ModelRouter(settings, UsageTracker(settings.max_llm_calls))
    log.info(
        "research_started",
        query=query[:200],
        mode=settings.llm_mode.value,
        models=router.describe(),
    )
    yield {"event": "started", "run_id": run_id, "query": query, "models": router.describe()}

    try:
        # Fail before spending anything if a configured model is unreachable.
        warnings = await router.preflight()
        for warning in warnings:
            yield {"event": "warning", "message": warning}

        async with AsyncExitStack() as stack:
            provider = build_provider(settings)
            search = await stack.enter_async_context(SearchService(provider, settings))
            fetcher = await stack.enter_async_context(PageFetcher(settings))
            checkpointer = await _make_checkpointer(settings, stack)

            context = RunContext(
                settings=settings,
                router=router,
                search=search,
                fetcher=fetcher,
                budget=settings.budget,
                run_id=run_id,
                started_at=started,
                warnings=warnings,
            )
            app = compile_graph(checkpointer=checkpointer)
            # recursion_limit bounds total super-steps. The research loop is
            # already bounded by round and budget checks; this is a backstop
            # against a routing bug turning into an unbounded run.
            config: RunnableConfig = {
                "configurable": {"thread_id": run_id},
                "recursion_limit": 100,
            }

            final_state: dict[str, Any] = {}
            async for mode, chunk in app.astream(
                initial_state(run_id, query, exhaustive_verification=exhaustive_verification),
                config=config,
                context=context,
                # Must be a list: LangGraph switches on isinstance(..., list)
                # to decide whether to yield (mode, chunk) pairs. A tuple is
                # accepted by the type checker and yields bare chunks instead.
                stream_mode=["custom", "values"],
            ):
                # With multiple stream modes the parts arrive as
                # (mode, payload); the payload type varies per mode, so it is
                # narrowed here rather than in the annotation.
                if mode == "custom":
                    yield cast("dict[str, Any]", chunk)
                elif mode == "values":
                    final_state = cast("dict[str, Any]", chunk)

            duration = time.perf_counter() - started
            metrics = build_metrics(
                run_id=run_id,
                query=query,
                mode=settings.llm_mode.value,
                model_assignments=router.describe(),
                duration_s=duration,
                state=final_state,
                usage=router.tracker,
                search_stats=search.stats,
                fetch_stats=fetcher.stats,
                environment=capture_environment(settings),
            )

            markdown = _render(final_state, metrics)
            output_dir = (
                _write_artifacts(settings, run_id, final_state, metrics, markdown)
                if settings.persist_runs
                else None
            )

            log.info(
                "research_completed",
                duration_s=round(duration, 2),
                rounds=metrics.research_rounds,
                sources=metrics.unique_sources,
                evidence=metrics.evidence_items,
                llm_calls=metrics.llm_calls,
                cost=metrics.cost_display,
            )
            yield {
                "event": "result",
                "result": RunResult(
                    run_id=run_id,
                    markdown=markdown,
                    metrics=metrics,
                    state=final_state,
                    output_dir=output_dir,
                    warnings=warnings,
                ),
            }
    finally:
        clear_run()


async def run_research(
    query: str,
    settings: Settings,
    *,
    run_id: str | None = None,
    on_progress: ProgressCallback | None = None,
    exhaustive_verification: bool = False,
) -> RunResult:
    """Convenience wrapper for callers that just want the finished result."""
    result: RunResult | None = None
    async for event in stream_research(
        query,
        settings,
        run_id=run_id,
        exhaustive_verification=exhaustive_verification,
    ):
        if event.get("event") == "result":
            result = event["result"]
        elif on_progress is not None:
            on_progress(event)
    if result is None:
        raise RuntimeError("research run produced no result")
    return result


def _render(state: dict[str, Any], metrics: RunMetrics) -> str:
    report = state.get("report")
    if report is None:
        return "# Research failed\n\nNo report was produced.\n"
    raw = state.get("verification")
    verification = CitationVerification.model_validate(raw) if raw else None
    return render_markdown(
        report,
        state.get("sources", []),
        verification,
        metrics,
        evidence=state.get("evidence", []),
    )


def _write_artifacts(
    settings: Settings,
    run_id: str,
    state: dict[str, Any],
    metrics: RunMetrics,
    markdown: str,
) -> Path:
    """Persist the run so it can be inspected without re-running anything.

    Everything needed to audit a claim is here: the report, every source, every
    evidence item with its quote and verification flag, and the metrics.
    """
    directory = settings.output_dir / run_id
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "report.md").write_text(markdown, encoding="utf-8")
    _dump(directory / "metrics.json", metrics.model_dump(mode="json"))
    _dump(
        directory / "sources.json",
        [s.model_dump(mode="json", exclude={"text"}) for s in state.get("sources", [])],
    )
    _dump(
        directory / "evidence.json",
        [e.model_dump(mode="json") for e in state.get("evidence", [])],
    )
    _dump(
        directory / "run.json",
        {
            "run_id": run_id,
            "query": metrics.query,
            "plan": state["plan"].model_dump(mode="json") if state.get("plan") else None,
            "queries": [q.model_dump(mode="json") for q in state.get("completed_queries", [])],
            "coverage_history": [
                c.model_dump(mode="json") for c in state.get("coverage_history", [])
            ],
            "verification": state.get("verification"),
            "errors": [e.model_dump(mode="json") for e in state.get("errors", [])],
            "stop_reason": state.get("stop_reason", ""),
        },
    )
    log.info("artifacts_written", path=str(directory))
    return directory


def _dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
