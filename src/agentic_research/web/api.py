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
import ipaddress
import json
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentic_research.config import Settings, get_settings
from agentic_research.llm.base import ProviderRateLimited
from agentic_research.observability import configure_logging, get_logger
from agentic_research.runner import new_run_id, stream_research
from agentic_research.web.durable_quota import (
    AtomicCounter,
    DurableRunQuota,
    build_counter,
)
from agentic_research.web.limits import (
    CapacityError,
    DemoLimits,
    RateLimiter,
    apply_demo_limits,
    demo_mode_summary,
    limits_from_settings,
    validate_query,
)
from agentic_research.web.recordings import RecordingNotFound
from agentic_research.web.recordings import available as available_recordings
from agentic_research.web.recordings import load as load_recording
from agentic_research.web.recordings import serialise_result as _serialise_result

log = get_logger(__name__)


def _find_frontend_dist() -> Path:
    """Locate the built frontend in both layouts this runs in.

    A source checkout has ``<repo>/web/dist`` three levels above this file.
    An installed wheel does not: the same walk lands in the interpreter's
    lib directory, so the container served JSON 404s at ``/`` while
    ``/api/health`` stayed green -- which is exactly what CI was probing.

    Candidates, in order of how explicit they are. The env var wins so a
    deployment can state the path outright; the working directory covers
    the container, whose WORKDIR holds ``web/dist``; the repo-relative walk
    covers development.
    """
    override = os.environ.get("FRONTEND_DIST")
    candidates = [
        *([Path(override)] if override else []),
        Path.cwd() / "web" / "dist",
        Path(__file__).resolve().parents[3] / "web" / "dist",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    # Nothing found: return the development path so the value is stable and
    # `is_dir()` stays false, which the 404 handler already copes with.
    return candidates[-1]


_FRONTEND_DIST = _find_frontend_dist()

# Recorded traces are replayed faster than they happened. An 18-minute local
# run is not worth watching in real time, and the events are real either
# way; only the spacing is cosmetic.
_REPLAY_EVENT_DELAY_S = 0.35

# An SSE comment. Standard, ignored by every conforming client, and
# deliberately not a fake event: a synthetic "still working" event would
# advance the pipeline UI and be counted as engine output.
_SSE_HEARTBEAT = ": keepalive\n\n"

# Comfortably inside the 30-60s idle window proxies typically enforce,
# and unrelated to uvicorn's --timeout-keep-alive, which governs idle
# connections *between* requests rather than an active response.
_HEARTBEAT_SECONDS = 15.0

_TIMED_OUT = {
    "error": "This demo run exceeded its time limit and was stopped.",
    "timeout": True,
}


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
    # Built during startup, because connecting to the store is I/O and
    # because a module imported at build time should not open sockets.
    # None until then, which reads as "not ready" rather than "no limit".
    quota: DurableRunQuota | None = None

    def live_research_available(self) -> bool:
        """Whether a live run could actually be admitted right now.

        Distinct from ``live_research_enabled``, which is a statement of
        intent. A deployment configured for live research whose quota
        store is unreachable is enabled and unavailable at once, and
        reporting only the intent is how a deploy looks healthy while
        refusing every run.
        """
        if not self.settings.live_research_enabled:
            return False
        if self.quota is None:
            return False
        return self.quota.usable()


def _peer(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _client_key(request: Request, trusted_hops: int) -> str:
    """Identify the caller for rate limiting.

    X-Forwarded-For is *appended to* by each proxy it passes through, so
    the leftmost entry is whatever the caller chose to send and the
    rightmost entries were added by infrastructure closer to us. Reading
    the first entry therefore lets a caller name itself: sending
    ``X-Forwarded-For: 1.2.3.4`` bought a fresh per-client allowance on
    every request, and the header cost nothing to change.

    With ``trusted_hops`` proxies in front of this service, the last
    ``trusted_hops`` entries are the ones those proxies contributed, and
    the caller's real address is the one immediately before them --
    index ``len(parts) - trusted_hops``. For Render's single proxy that
    is the last entry, which a spoofed prefix cannot displace.

    Anything unexpected falls back to the socket peer: zero configured
    hops (trust nothing), fewer entries than hops (the header did not
    come through the expected path), or a value that is not an IP
    address. The peer address is the one thing a caller cannot forge.

    This bounds accidents and casual abuse. It is not the financial
    boundary -- the durable global quota is.
    """
    if trusted_hops <= 0:
        return _peer(request)

    forwarded = request.headers.get("x-forwarded-for", "")
    parts = [p.strip() for p in forwarded.split(",") if p.strip()]
    if len(parts) < trusted_hops:
        return _peer(request)

    candidate = parts[len(parts) - trusted_hops]
    try:
        # Parsed, not merely trimmed: a hostname, a port suffix or junk
        # is refused, and so is 192.000.002.044, because leading zeros
        # are octal to some stacks and decimal to others. Returned in
        # normalised form so one client cannot hold two allowances by
        # varying the spelling of its own address.
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return _peer(request)


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


def create_app(
    settings: Settings | None = None,
    *,
    counter_factory: Callable[[str | None], AtomicCounter | None] = build_counter,
) -> FastAPI:
    """Build the application.

    ``counter_factory`` is injectable so tests can give two app
    instances one shared store and assert that the cap is genuinely
    shared. Testing the quota object directly would prove only that the
    class works, not that the route calls it.
    """
    resolved = settings or get_settings()
    configure_logging(resolved.log_level, resolved.log_format)
    limits = limits_from_settings(resolved)
    state = AppState(
        settings=resolved,
        limits=limits,
        limiter=RateLimiter(limits),
        demo_mode=resolved.demo_mode,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Connecting here rather than at import: this opens a socket, and
        # a failure has to be visible as "not ready" rather than as an
        # import error with no HTTP surface to report it on.
        counter = await asyncio.to_thread(counter_factory, resolved.demo_quota_url)
        state.quota = DurableRunQuota(
            counter,
            limits.global_runs_per_day,
            required=resolved.demo_quota_required,
        )
        log.info(
            "web_started",
            demo_mode=state.demo_mode,
            mode=resolved.llm_mode.value,
            frontend_present=_FRONTEND_DIST.is_dir(),
            durable_quota=counter is not None,
            live_research_available=state.live_research_available(),
        )
        yield

    from agentic_research import __version__

    app = FastAPI(
        title="Agentic Research Engine",
        # Read from the package rather than restated here. The two drifted
        # once already (package 0.1.0, API 0.2.0) and a hardcoded string is
        # guaranteed to drift again.
        version=__version__,
        lifespan=lifespan,
        # No interactive docs in demo mode: they invite poking at an endpoint
        # that spends money, and add nothing for a demo visitor.
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
        """Liveness probe. Deliberately cheap and credential-free.

        Carries the version and commit so a deployed instance can be
        matched to the code that produced it without guessing from the
        deploy timestamp.
        """
        from agentic_research.provenance import git_state

        return {
            "status": "ok",
            "version": __version__,
            "commit": git_state().get("short_commit", "unavailable"),
            "demo_mode": state.demo_mode,
            "capacity": await state.limiter.snapshot(),
        }

    @api.get("/readiness")
    async def readiness() -> JSONResponse:
        """What this instance can actually do, not merely that it is up.

        Liveness passes on a deployment that refuses every run, which is
        why it is the wrong signal for acceptance. These four facts come
        apart in practice and each has a different remedy:

        ``replay_available``
            Recordings are present. Replay costs nothing and must keep
            working even when live research cannot.
        ``live_research_enabled``
            What the configuration asks for.
        ``durable_quota_available``
            Whether the shared counter can be reached. Without it a live
            deployment has no cap that survives a restart.
        ``live_research_available``
            Whether a run could actually be admitted. A deployment can
            be enabled and unavailable at the same time.

        503 when the instance cannot do what it is configured to do, so
        a deploy check fails rather than reporting a healthy service
        that turns every visitor away.
        """
        quota = state.quota
        replay_available = bool(available_recordings())
        enabled = state.settings.live_research_enabled
        live_available = state.live_research_available()
        body: dict[str, Any] = {
            "alive": True,
            "replay_available": replay_available,
            "live_research_enabled": enabled,
            "live_research_available": live_available,
            "durable_quota_available": quota is not None and quota.usable(),
            "quota_initialised": quota is not None,
        }
        ready = replay_available and (live_available or not enabled)
        body["ready"] = ready
        return JSONResponse(body, status_code=200 if ready else 503)

    @api.get("/config")
    async def config() -> dict[str, Any]:
        """What the client may know. Never any secret or raw environment."""
        # service_mode and live_research_enabled come from the summary, so
        # the UI can describe what it offers rather than discovering the
        # refusal after the visitor has typed a question.
        summary = demo_mode_summary(state.settings, state.limits)
        summary["recorded_examples"] = len(available_recordings())
        return summary

    @api.get("/examples")
    async def examples() -> dict[str, Any]:
        """Recorded runs. Needs no credentials and spends nothing."""
        return {"examples": [s.to_dict() for s in available_recordings()]}

    @api.get("/examples/{example_id}")
    async def example(example_id: str) -> Any:
        """One recorded run, in the same shape a live run returns.

        The UI renders it identically apart from the recording label, which
        is the point: the provenance explorer is the thing worth showing.
        """
        try:
            payload = load_recording(example_id)
        except RecordingNotFound:
            return JSONResponse({"error": "No such example."}, status_code=404)
        return {
            "recorded": True,
            "meta": payload.get("meta", {}),
            "result": payload.get("result", {}),
        }

    @api.get("/examples/{example_id}/stream")
    async def example_stream(example_id: str, request: Request) -> Any:
        """Replay a recorded run's own progress events.

        Every event here was emitted by a node that really ran. Timing is
        compressed for watchability; nothing is invented. A recording with
        no captured trace simply yields its result.
        """
        try:
            payload = load_recording(example_id)
        except RecordingNotFound:
            return JSONResponse({"error": "No such example."}, status_code=404)

        async def replay() -> AsyncIterator[str]:
            meta = payload.get("meta", {})
            yield _sse(
                "started",
                {
                    "run_id": f"recorded-{example_id}",
                    "query": meta.get("question", ""),
                    "recorded": True,
                },
            )
            for event in payload.get("trace", []) or []:
                if await request.is_disconnected():
                    return
                await asyncio.sleep(_REPLAY_EVENT_DELAY_S)
                yield _sse("progress", event)
            yield _sse("result", payload.get("result", {}))
            yield _sse("done", {"run_id": f"recorded-{example_id}", "recorded": True})

        return StreamingResponse(
            replay(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @api.post("/research")
    async def research(payload: ResearchRequest, request: Request) -> Any:
        # Enforced here, on the server, before anything is validated or
        # dispatched. Hiding the button in React would leave the endpoint
        # open to anyone with curl, and this endpoint spends money.
        if not state.settings.live_research_enabled:
            return JSONResponse(
                {
                    "error": (
                        "Live research is disabled on this public demo. "
                        "Explore a recorded run, or run the project "
                        "locally for live research."
                    ),
                    "live_disabled": True,
                },
                status_code=403,
            )
        try:
            query = validate_query(payload.query, state.limits)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        client = _client_key(request, state.settings.trusted_proxy_hops)
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

        # The financial boundary, and the only counter that outlives this
        # process. Taken after the in-memory limiter so concurrency is
        # already bounded, and before the router, the search provider and
        # the verifier -- every one of which costs money.
        #
        # A refused reservation must hand back the concurrency slot it
        # just took. The streaming path releases it on exit, and this
        # path never reaches the streaming path.
        quota = state.quota
        if quota is None:
            await state.limiter.release()
            return JSONResponse(
                {
                    "error": "The service is still starting. Try again shortly.",
                    "error_code": "not_ready",
                },
                status_code=503,
            )
        decision = await asyncio.to_thread(quota.reserve)
        if not decision.allowed:
            await state.limiter.release()
            log.info(
                "live_run_refused",
                reason=decision.detail,
                used=decision.used,
                limit=decision.limit,
            )
            return JSONResponse(
                {
                    # decision.detail names the mechanism, not the store,
                    # the URL or any credential.
                    "error": (
                        "The demo has reached its daily budget. It resets within 24 hours."
                        if decision.used >= decision.limit > 0
                        else "Live research is unavailable right now."
                    ),
                    "capacity_reached": True,
                },
                status_code=429,
                headers={"Retry-After": "3600"},
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

    @app.exception_handler(404)
    async def not_found(request: Request, exc: Exception) -> Response:
        """Client-side routes get the app shell; unknown API paths get JSON.

        Deliberately a 404 handler rather than a catch-all ``/{path:path}``
        route. A catch-all *matches* every request, which suppresses
        Starlette's trailing-slash redirect -- so ``/api/examples/`` became
        a 404 when the frontend was built and a 307 to the collection when
        it was not. Same service, two behaviours, decided by whether a
        build directory happened to exist. Running after routing has
        already failed keeps redirects intact and keeps this path identical
        in both cases.
        """
        path = request.url.path
        if path == "/api" or path.startswith("/api/"):
            # A missing endpoint must not answer with the SPA shell, or a
            # caller cannot tell it apart from a working one.
            return JSONResponse({"error": "Not found."}, status_code=404)
        if not _FRONTEND_DIST.is_dir():
            return JSONResponse({"error": "Not found."}, status_code=404)

        root = _FRONTEND_DIST.resolve()
        candidate = (_FRONTEND_DIST / path.lstrip("/")).resolve()
        if path.strip("/") and candidate.is_file() and root in candidate.parents:
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

    Three things this has to get right, all of them cleanup rather than
    happy path.

    *The slot is released exactly once, on every exit.* A hosted demo that
    leaks a concurrency slot when a visitor closes the tab tells the next
    visitor it is busy, forever.

    *The underlying generator is closed explicitly.* Breaking out of the
    loop on disconnect or timeout leaves ``stream_research`` suspended
    mid-run, holding its HTTP clients and graph state until the collector
    happens to run. ``aclose()`` unwinds it deterministically.

    *Nothing is yielded from a cancellation path.* ``done`` used to be
    emitted from ``finally``, which runs while the generator is being
    closed; yielding there raises "async generator ignored GeneratorExit"
    and turns a clean disconnect into an error. It is emitted after
    cleanup instead, and only if the client is still there to read it.
    """
    started = time.monotonic()
    deadline = state.limits.max_runtime_seconds
    stream: AsyncGenerator[dict[str, Any], None] | None = None
    pending: asyncio.Task[dict[str, Any]] | None = None
    disconnected = False

    try:
        yield _sse("started", {"run_id": run_id, "query": query})

        # Two names on purpose: `events` is what the loop iterates and is
        # never None, `stream` is what `finally` closes and may be None if
        # construction itself raised.
        events = stream_research(query, settings, run_id=run_id)
        stream = events
        while True:
            if await request.is_disconnected():
                log.info("web_client_disconnected", run_id=run_id)
                disconnected = True
                break

            remaining = deadline - (time.monotonic() - started)
            if remaining <= 0:
                yield _sse("error", _TIMED_OUT)
                break

            # Wake at the heartbeat interval even when the engine is
            # silent. A cloud model can take tens of seconds, and a
            # connection with no bytes on it looks dead to proxies and to
            # the browser alike.
            #
            # The pending step is a task held *across* heartbeats, not a
            # fresh `wait_for` each time. `wait_for` cancels its awaitable
            # on timeout, so re-awaiting `__anext__` would abandon the
            # engine's in-flight work at every heartbeat and restart it --
            # a run that never finishes, and an async generator left in a
            # broken state.
            if pending is None:
                pending = asyncio.ensure_future(events.__anext__())
            finished, _ = await asyncio.wait({pending}, timeout=min(_HEARTBEAT_SECONDS, remaining))
            if not finished:
                if time.monotonic() - started >= deadline:
                    yield _sse("error", _TIMED_OUT)
                    break
                # Silence, not expiry: keep the connection warm and let the
                # same step carry on. A comment is not an event, so the
                # client parses nothing and the pipeline does not advance.
                yield _SSE_HEARTBEAT
                continue

            step, pending = pending, None
            try:
                event = step.result()
            except StopAsyncIteration:
                break

            if event.get("event") == "result":
                yield _sse("result", _serialise_result(event["result"]))
            else:
                yield _sse("progress", event)
    except ProviderRateLimited:
        # Distinguished from a generic failure on purpose. "Please try
        # again" is wrong advice here: the next attempt fails the same way
        # and spends another provider request doing it. The provider's own
        # message is not forwarded -- it names models and limits.
        log.warning("web_run_rate_limited", run_id=run_id)
        yield _sse(
            "error",
            {
                "error": (
                    "The demo has reached its provider quota for now. The "
                    "recorded runs below still work, and the quota resets "
                    "within a day."
                ),
                "capacity_reached": True,
            },
        )
    except Exception as exc:
        log.warning("web_run_failed", run_id=run_id, error=str(exc)[:200])
        # The message is deliberately generic: an exception string can carry
        # a URL, a model name or a provider error that reveals configuration.
        yield _sse("error", {"error": "The research run failed. Please try again."})
    finally:
        # Cleanup only. GeneratorExit and CancelledError pass straight
        # through this block and skip the `done` below, which is what
        # keeps a cancelled stream from yielding.
        if pending is not None:
            # An in-flight step outlives the response otherwise, holding
            # the graph and its clients open with nobody reading it.
            pending.cancel()
            with suppress(BaseException):
                await pending
        if stream is not None:
            with suppress(Exception):
                await stream.aclose()
        await state.limiter.release()

    if not disconnected:
        yield _sse("done", {"run_id": run_id})


app = None  # populated by create_app() in the ASGI entry point below


def get_asgi_app() -> FastAPI:
    """Entry point for uvicorn: ``agentic_research.web.api:get_asgi_app``."""
    return create_app()
