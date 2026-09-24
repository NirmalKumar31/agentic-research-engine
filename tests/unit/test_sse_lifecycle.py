"""SSE streaming: cleanup on every exit, and bytes while the engine is quiet.

A hosted demo has one concurrency slot on the free tier. A slot leaked by
a visitor closing a tab is a slot the next visitor is refused, for the
life of the process, so release has to happen on every path out --
completion, timeout, disconnect, provider failure and cancellation.

Two defects motivated these tests. The underlying `stream_research`
generator was never closed, so breaking out of the loop left a run
suspended mid-flight holding its HTTP clients. And `done` was yielded
from `finally`, which runs while the generator is being closed: yielding
there raises "async generator ignored GeneratorExit" and turns a clean
disconnect into an error.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentic_research.config import Settings
from agentic_research.llm.base import ProviderRateLimited
from agentic_research.web import api as api_module
from agentic_research.web.api import AppState, _event_stream
from agentic_research.web.limits import RateLimiter, limits_from_settings


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_mode": "local",
        "demo_mode": True,
        "live_research_enabled": True,
        "tavily_api_key": "tvly-test-key",
        "_env_file": None,
    }
    base.update(over)
    return Settings(**base)


class FakeRequest:
    """Minimal stand-in for the disconnect probe the stream polls."""

    def __init__(self, disconnect_after: int | None = None) -> None:
        self._polls = 0
        self._after = disconnect_after

    async def is_disconnected(self) -> bool:
        self._polls += 1
        return self._after is not None and self._polls > self._after


class TrackedStream:
    """An async generator whose closure can be observed."""

    def __init__(self, events: list[dict[str, Any]], *, stall: float = 0.0) -> None:
        self.events = events
        self.stall = stall
        self.closed = False
        self._index = 0

    def __aiter__(self) -> TrackedStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self.stall:
            await asyncio.sleep(self.stall)
        if self._index >= len(self.events):
            raise StopAsyncIteration
        event = self.events[self._index]
        self._index += 1
        return event

    async def aclose(self) -> None:
        self.closed = True


def _state(settings: Settings) -> AppState:
    limits = limits_from_settings(settings)
    return AppState(
        settings=settings,
        limits=limits,
        limiter=RateLimiter(limits),
        demo_mode=True,
    )


async def _drain(stream: AsyncIterator[str], limit: int = 200) -> list[str]:
    out: list[str] = []
    async for chunk in stream:
        out.append(chunk)
        if len(out) >= limit:
            break
    return out


async def _active(state: AppState) -> int:
    return (await state.limiter.snapshot())["active_runs"]


class TestTheSlotIsReleasedExactlyOnce:
    async def test_on_normal_completion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _settings()
        state = _state(settings)
        tracked = TrackedStream([{"event": "planning"}])
        monkeypatch.setattr(api_module, "stream_research", lambda *a, **k: tracked)

        await state.limiter.acquire("client")
        assert await _active(state) == 1

        chunks = await _drain(_event_stream(state, settings, "q", "r1", FakeRequest()))

        assert await _active(state) == 0
        assert any("event: done" in c for c in chunks)
        assert tracked.closed, "the underlying generator was left open"

    async def test_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _settings(demo_max_runtime_seconds=0.05)
        state = _state(settings)
        tracked = TrackedStream([{"event": "planning"}], stall=5.0)
        monkeypatch.setattr(api_module, "stream_research", lambda *a, **k: tracked)

        await state.limiter.acquire("client")
        chunks = await _drain(_event_stream(state, settings, "q", "r1", FakeRequest()))

        assert await _active(state) == 0
        assert any('"timeout": true' in c for c in chunks)
        assert tracked.closed

    async def test_on_client_disconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _settings()
        state = _state(settings)
        tracked = TrackedStream([{"event": "a"}, {"event": "b"}, {"event": "c"}])
        monkeypatch.setattr(api_module, "stream_research", lambda *a, **k: tracked)

        await state.limiter.acquire("client")
        request = FakeRequest(disconnect_after=1)
        chunks = await _drain(_event_stream(state, settings, "q", "r1", request))

        assert await _active(state) == 0
        assert tracked.closed
        # Nothing is sent to a client that has gone away.
        assert not any("event: done" in c for c in chunks)

    async def test_on_provider_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _settings()
        state = _state(settings)

        def explode(*a: Any, **k: Any) -> Any:
            raise ProviderRateLimited("quota gone")

        monkeypatch.setattr(api_module, "stream_research", explode)

        await state.limiter.acquire("client")
        chunks = await _drain(_event_stream(state, settings, "q", "r1", FakeRequest()))

        assert await _active(state) == 0
        assert any("capacity_reached" in c for c in chunks)
        assert any("event: done" in c for c in chunks)

    async def test_a_generic_failure_does_not_leak_detail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _settings()
        state = _state(settings)

        def explode(*a: Any, **k: Any) -> Any:
            raise RuntimeError("https://internal.example/secret-model-name")

        monkeypatch.setattr(api_module, "stream_research", explode)

        await state.limiter.acquire("client")
        chunks = await _drain(_event_stream(state, settings, "q", "r1", FakeRequest()))

        body = "".join(chunks)
        assert await _active(state) == 0
        assert "internal.example" not in body
        assert "The research run failed" in body


class TestCancellationDoesNotYield:
    async def test_closing_the_response_generator_cleans_up_without_yielding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`done` used to be emitted from `finally`. Closing a generator
        mid-flight then raises "async generator ignored GeneratorExit"."""
        settings = _settings()
        state = _state(settings)
        tracked = TrackedStream([{"event": "a"}, {"event": "b"}], stall=0.01)
        monkeypatch.setattr(api_module, "stream_research", lambda *a, **k: tracked)

        await state.limiter.acquire("client")
        stream = _event_stream(state, settings, "q", "r1", FakeRequest())

        # Past `started`, so `stream_research` has been called and there
        # is something to clean up. Closing at `started` would correctly
        # have nothing to close.
        assert "event: started" in await stream.__anext__()
        assert "event: progress" in await stream.__anext__()
        await stream.aclose()  # must not raise

        assert await _active(state) == 0
        assert tracked.closed


class TestHeartbeat:
    async def test_a_silent_engine_still_produces_bytes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cloud model can be quiet for tens of seconds; the connection
        must not look dead to the browser or an intermediary."""
        monkeypatch.setattr(api_module, "_HEARTBEAT_SECONDS", 0.05)
        settings = _settings(demo_max_runtime_seconds=10.0)
        state = _state(settings)
        # One event, arriving well after several heartbeat intervals.
        tracked = TrackedStream([{"event": "planning"}], stall=0.22)
        monkeypatch.setattr(api_module, "stream_research", lambda *a, **k: tracked)

        await state.limiter.acquire("client")
        chunks = await _drain(_event_stream(state, settings, "q", "r1", FakeRequest()))

        assert any(c == ": keepalive\n\n" for c in chunks), "no heartbeat was emitted"
        # The run continued and the real event still arrived afterwards.
        assert any("event: progress" in c for c in chunks)
        assert any("event: done" in c for c in chunks)

    async def test_a_heartbeat_is_a_comment_not_an_event(self) -> None:
        """A synthetic event would advance the pipeline UI and be counted
        as engine output. A comment carries no event or data field."""
        assert api_module._SSE_HEARTBEAT.startswith(":")
        assert "event:" not in api_module._SSE_HEARTBEAT
        assert "data:" not in api_module._SSE_HEARTBEAT

    async def test_heartbeats_stop_at_the_run_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keeping the connection warm must not keep a run alive past its
        wall-clock ceiling."""
        monkeypatch.setattr(api_module, "_HEARTBEAT_SECONDS", 0.02)
        settings = _settings(demo_max_runtime_seconds=0.1)
        state = _state(settings)
        tracked = TrackedStream([{"event": "planning"}], stall=5.0)
        monkeypatch.setattr(api_module, "stream_research", lambda *a, **k: tracked)

        await state.limiter.acquire("client")
        chunks = await _drain(_event_stream(state, settings, "q", "r1", FakeRequest()))

        assert any('"timeout": true' in c for c in chunks)
        assert tracked.closed
        assert await _active(state) == 0
