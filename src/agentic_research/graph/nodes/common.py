"""Helpers shared by graph nodes."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from langgraph.runtime import get_runtime

from agentic_research.graph.state import RunContext
from agentic_research.models import RunError
from agentic_research.observability import get_logger

log = get_logger(__name__)


def ctx() -> RunContext:
    """The live run context for the current node."""
    return get_runtime(RunContext).context


def emit(event: str, **fields: Any) -> None:
    """Push a progress event to anything consuming ``stream_mode="custom"``.

    Progress reporting is a stream concern, not a state concern: writing it
    into state would checkpoint every UI tick and make replay noisy.
    """
    try:
        writer = get_runtime(RunContext).stream_writer
    except Exception:
        return
    if writer is not None:
        writer({"event": event, **fields})


@contextmanager
def stage(name: str) -> Iterator[dict[str, Any]]:
    """Time a stage and yield a dict the node can add detail to."""
    record: dict[str, Any] = {"stage": name}
    started = time.perf_counter()
    try:
        yield record
    finally:
        record["seconds"] = round(time.perf_counter() - started, 3)


def error_from(stage_name: str, exc: BaseException, context: str = "") -> RunError:
    return RunError(
        stage=stage_name,
        kind=type(exc).__name__,
        message=str(exc)[:500],
        context=context[:200],
    )


def numbered(prefix: str, start: int, count: int) -> list[str]:
    """Generate ordinal ids such as SQ1..SQ4.

    Ids are assigned by the engine in single-writer nodes, never by a model
    and never inside a parallel worker, so they stay stable and race-free.
    """
    return [f"{prefix}{start + i}" for i in range(count)]


def first_sentence(text: str, limit: int = 160) -> str:
    cut = text.strip().split("\n", 1)[0]
    return cut[:limit] + ("..." if len(cut) > limit else "")


Progress = Callable[[str], None]
