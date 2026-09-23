"""HTTP API for the hosted demo.

Thin by design. It drives the same ``stream_research`` generator the CLI
uses and forwards the graph's own events over Server-Sent Events, so there
is no second copy of the research logic and no invented progress: every
event the browser renders was emitted by a node that actually ran.

SSE rather than WebSockets because the traffic is one-directional and SSE
survives proxies and free-tier hosting without special configuration.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentic_research.config import Settings, get_settings
from agentic_research.observability import configure_logging, get_logger
from agentic_research.runner import RunResult, new_run_id, stream_research
from agentic_research.web.limits import (
    CapacityError,
    DemoLimits,
    RateLimiter,
    apply_demo_limits,
    demo_mode_summary,
    validate_query,
)

log = get_logger(__name__)

_FRONTEND_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


class ResearchRequest(BaseModel):
    """A demo research request.

    Only the question is honoured in demo mode. The optional fields exist so
    a local operator can narrow a run; they can never widen one, because the
    server clamps afterwards.
    """

    query: str = Field(min_length=1, max_length=2_000)
    max_rounds: int | None = Field(default=None, ge=1, le=5)
    max_sources: int | None = Field(default=None, ge=1, le=40)


@dataclass
class AppState:
    settings: Settings
    limits: DemoLimits
    limiter: RateLimiter
    demo_mode: bool


def _client_key(request: Request) -> str:
    """Identify the caller for rate limiting.

    Behind Render's proxy the peer address is the proxy, so the first entry
    of X-Forwarded-For is used when present. It is spoofable by a determined
    caller; the global daily cap and the per-run spend ceiling are the
    limits that actually bound cost, and this one bounds accidents.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


def _serialise_result(result: RunResult) -> dict[str, Any]:
    """Everything the UI needs to show provenance, and nothing more.

    Deliberately assembled field by field rather than dumping state: state
    holds full source text and would leak configuration if serialised
    wholesale.
    """
    state = result.state
    report = state.get("report")
    evidence = state.get("evidence", []) or []
    sources = state.get("sources", []) or []
    plan = state.get("plan")

    def claim(c: Any) -> dict[str, Any]:
        return {
            "text": c.text,
            "kind": c.kind.value,
            "evidence_ids": list(c.evidence_ids),
            "citation_ids": list(c.citation_ids),
        }

    return {
        "run_id": result.run_id,
        "report": None
        if report is None
        else {
            "title": report.title,
            "summary_claims": [claim(c) for c in report.summary_claims],
            "key_findings": [claim(c) for c in report.key_findings],
            "sections": [
                {"heading": s.heading, "claims": [claim(c) for c in s.claims]}
                for s in report.sections
            ],
            "contradictions": [
                {
                    "topic": c.topic,
                    "left_summary": c.left_summary,
                    "left_evidence_ids": list(c.left_evidence_ids),
                    "left_citation_ids": list(c.left_citation_ids),
                    "right_summary": c.right_summary,
                    "right_evidence_ids": list(c.right_evidence_ids),
                    "right_citation_ids": list(c.right_citation_ids),
                    "auditable": c.is_auditable,
                }
                for c in report.contradictions
            ],
            "limitations": list(report.limitations),
        },
        "plan": None
        if plan is None
        else {
            "strategy": plan.strategy_note,
            "sub_questions": [
                {
                    "id": q.id,
                    "text": q.text,
                    "rationale": q.rationale,
                    "is_followup": q.is_followup,
                }
                for q in plan.sub_questions
            ],
        },
        "evidence": [
            {
                "id": e.id,
                "source_id": e.source_id,
                "sub_question_id": e.sub_question_id,
                "claim": e.claim,
                "quote": e.quote,
                "quote_match": e.quote_match.value,
                "page": e.page,
                "stance": e.stance.value,
                "confidence": e.confidence,
                "citable": e.is_citable,
                "query_id": e.query_id,
                "cross_attributed": e.cross_attributed,
            }
            for e in evidence
        ],
        "sources": [
            {
                "id": s.id,
                "url": s.url,
                "title": s.title,
                "domain": s.domain,
                "source_type": s.source_type.value,
                "content_origin": s.content_origin.value,
                "quality_score": s.quality_score,
                "page_count": s.page_count,
                "usable": s.is_usable,
                "fetch_status": s.fetch_status.value,
            }
            for s in sources
        ],
        "verification": state.get("verification"),
        "metrics": result.metrics.model_dump(mode="json"),
        "markdown": result.markdown,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level, resolved.log_format)
    limits = DemoLimits()
    state = AppState(
        settings=resolved,
        limits=limits,
        limiter=RateLimiter(limits),
        demo_mode=resolved.demo_mode,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info(
            "web_started",
            demo_mode=state.demo_mode,
            mode=resolved.llm_mode.value,
            frontend_present=_FRONTEND_DIST.is_dir(),
        )
        yield

    app = FastAPI(
        title="Agentic Research Engine",
        version="0.2.0",
        lifespan=lifespan,
        # No interactive docs in demo mode: they invite poking at an endpoint
        # that spends money, and add nothing for a portfolio visitor.
        docs_url=None if state.demo_mode else "/docs",
        redoc_url=None,
    )
    app.state.research = state

    origins = [o.strip() for o in resolved.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["content-type"],
        )

    api = APIRouter(prefix="/api")

    @api.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness probe. Deliberately cheap and credential-free."""
        return {
            "status": "ok",
            "demo_mode": state.demo_mode,
            "capacity": await state.limiter.snapshot(),
        }

    @api.get("/config")
    async def config() -> dict[str, Any]:
        """What the client may know. Never any secret or raw environment."""
        return demo_mode_summary(state.settings, state.limits)

    @api.post("/research")
    async def research(payload: ResearchRequest, request: Request) -> Any:
        try:
            query = validate_query(payload.query, state.limits)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        client = _client_key(request)
        try:
            await state.limiter.acquire(client)
        except CapacityError as exc:
            headers = (
                {"Retry-After": str(exc.retry_after_seconds)} if exc.retry_after_seconds else {}
            )
            return JSONResponse(
                {"error": exc.reason, "capacity_reached": True},
                status_code=429,
                headers=headers,
            )

        run_settings = state.settings
        # Client values may only narrow a run.
        narrowing: dict[str, Any] = {}
        if payload.max_rounds is not None:
            narrowing["max_research_rounds"] = min(
                run_settings.max_research_rounds, payload.max_rounds
            )
        if payload.max_sources is not None:
            narrowing["max_sources"] = min(run_settings.max_sources, payload.max_sources)
        if narrowing:
            run_settings = run_settings.model_copy(update=narrowing)
        if state.demo_mode:
            run_settings = apply_demo_limits(run_settings, state.limits)

        run_id = new_run_id()
        return StreamingResponse(
            _event_stream(state, run_settings, query, run_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Render and most proxies buffer by default, which would
                # hold every event until the run finished.
                "X-Accel-Buffering": "no",
            },
        )

    app.include_router(api)

    if _FRONTEND_DIST.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=_FRONTEND_DIST / "assets"),
            name="assets",
        )

        @app.get("/{full_path:path}")
        async def spa(full_path: str) -> Any:
            """Serve the built frontend, falling back to index for routes."""
            candidate = (_FRONTEND_DIST / full_path).resolve()
            if full_path and candidate.is_file() and _FRONTEND_DIST.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(_FRONTEND_DIST / "index.html")

    return app


async def _event_stream(
    state: AppState,
    settings: Settings,
    query: str,
    run_id: str,
    request: Request,
) -> AsyncIterator[str]:
    """Forward graph events to the browser, then release the slot.

    Wrapped in a timeout and a disconnect check: a hosted demo must reclaim
    capacity when a visitor closes the tab mid-run, or the concurrency slot
    leaks and the next visitor is told the demo is busy.
    """
    started = time.monotonic()
    deadline = state.limits.max_runtime_seconds
    try:
        yield _sse("started", {"run_id": run_id, "query": query})

        stream = stream_research(query, settings, run_id=run_id)
        while True:
            if await request.is_disconnected():
                log.info("web_client_disconnected", run_id=run_id)
                break
            remaining = deadline - (time.monotonic() - started)
            if remaining <= 0:
                yield _sse(
                    "error",
                    {
                        "error": "This demo run exceeded its time limit and was stopped.",
                        "timeout": True,
                    },
                )
                break
            try:
                event = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            except TimeoutError:
                yield _sse(
                    "error",
                    {
                        "error": "This demo run exceeded its time limit and was stopped.",
                        "timeout": True,
                    },
                )
                break

            if event.get("event") == "result":
                yield _sse("result", _serialise_result(event["result"]))
            else:
                yield _sse("progress", event)
    except Exception as exc:
        log.warning("web_run_failed", run_id=run_id, error=str(exc)[:200])
        # The message is deliberately generic: an exception string can carry
        # a URL, a model name or a provider error that reveals configuration.
        yield _sse("error", {"error": "The research run failed. Please try again."})
    finally:
        await state.limiter.release()
        yield _sse("done", {"run_id": run_id})


app = None  # populated by create_app() in the ASGI entry point below


def get_asgi_app() -> FastAPI:
    """Entry point for uvicorn: ``agentic_research.web.api:get_asgi_app``."""
    return create_app()
