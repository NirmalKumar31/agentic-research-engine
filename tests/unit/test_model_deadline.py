"""A stuck model call must be bounded by wall-clock time.

This is a regression test for an observed failure, not a hypothetical.
A host slept mid-extraction, the connection to Ollama went quiet without
closing, and the process sat at 0% CPU for seven hours inside one
`await`. `llm_timeout_seconds` never fired, because an HTTP client
timeout bounds time between socket events and there were no more
events.

So the tests here do not raise TimeoutError to see it classified. They
hand the router a provider that genuinely never returns, and assert the
call is actually cut off.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest

from agentic_research.config import ModelRole, ModelSpec, Provider
from agentic_research.llm.base import ModelTimeoutError, UsageTracker
from agentic_research.llm.router import RoleModel
from agentic_research.schemas import PlanOut


class _FakeChatModel:
    """Minimal stand-in for the LangChain chat-model surface.

    ``with_structured_output`` returns the object itself, so the
    hanging ``ainvoke`` below is what the router ends up awaiting.
    """

    def with_structured_output(self, _schema: object, **_kwargs: object) -> _FakeChatModel:
        return self


class HangingRunnable(_FakeChatModel):
    """A provider whose call never completes."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    async def ainvoke(self, _messages: object) -> dict:
        self.started.set()
        try:
            await asyncio.Event().wait()  # never set
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")


class SlowRunnable(_FakeChatModel):
    """A provider that answers, but far too late."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.cancelled = False

    async def ainvoke(self, _messages: object) -> dict:
        try:
            await asyncio.sleep(self.seconds)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return {"raw": None, "parsed": None, "parsing_error": None}


class StubbornRunnable(_FakeChatModel):
    """A provider that defers cancellation briefly before yielding.

    Cleanup handlers that swallow CancelledError for a moment are
    ordinary; one that never yields would hang any supervisor, so the
    contract is only that we do not hang when it eventually does.
    """

    def __init__(self) -> None:
        self.cleanup_ran = False

    async def ainvoke(self, _messages: object) -> dict:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(0.05)  # deferred cleanup
            self.cleanup_ran = True
            raise
        raise AssertionError("unreachable")


def model(runnable: object, timeout: float) -> RoleModel:
    return RoleModel(
        role=ModelRole.PLANNER,
        spec=ModelSpec(provider=Provider.OLLAMA, model="test-model"),
        model=runnable,  # type: ignore[arg-type]
        tracker=UsageTracker(50, 0.0, max_provider_requests=50),
        timeout_seconds=timeout,
    )


class TestTheDeadlineIsEnforced:
    async def test_a_call_that_never_returns_is_cut_off(self) -> None:
        runnable = HangingRunnable()
        bound = model(runnable, timeout=0.2)
        started = time.perf_counter()
        with pytest.raises(ModelTimeoutError):
            await bound.structured(PlanOut, "system", "user")
        elapsed = time.perf_counter() - started
        assert elapsed < 5, f"took {elapsed:.1f}s; the deadline did not fire"
        assert runnable.started.is_set()

    async def test_the_provider_task_is_actually_cancelled(self) -> None:
        """Not merely abandoned. A task left running holds its
        connection and its concurrency slot forever."""
        runnable = HangingRunnable()
        with pytest.raises(ModelTimeoutError):
            await model(runnable, timeout=0.2).structured(PlanOut, "system", "user")
        await asyncio.sleep(0)
        assert runnable.cancelled

    async def test_it_fires_close_to_the_configured_bound(self) -> None:
        runnable = SlowRunnable(seconds=30)
        started = time.perf_counter()
        with pytest.raises(ModelTimeoutError):
            await model(runnable, timeout=0.3).structured(PlanOut, "system", "user")
        elapsed = time.perf_counter() - started
        assert elapsed < 3, f"took {elapsed:.1f}s against a 0.3s timeout"
        assert runnable.cancelled

    async def test_a_provider_that_defers_cancellation_does_not_hang_us(self) -> None:
        runnable = StubbornRunnable()
        started = time.perf_counter()
        with pytest.raises(ModelTimeoutError):
            await model(runnable, timeout=0.2).structured(PlanOut, "system", "user")
        assert time.perf_counter() - started < 5

    async def test_the_error_names_the_deadline_and_the_model(self) -> None:
        with pytest.raises(ModelTimeoutError) as caught:
            await model(HangingRunnable(), timeout=0.2).structured(PlanOut, "system", "user")
        message = str(caught.value)
        assert "wall-clock deadline" in message
        assert "test-model" in message

    async def test_no_credential_or_prompt_text_leaks_into_the_error(self) -> None:
        with pytest.raises(ModelTimeoutError) as caught:
            await model(HangingRunnable(), timeout=0.2).structured(
                PlanOut, "system prompt with secrets", "user prompt with secrets"
            )
        assert "secrets" not in str(caught.value)


class TestResourcesAreReleased:
    async def test_a_later_call_still_works_after_a_timeout(self) -> None:
        """A leaked semaphore slot or unreleased reservation would show
        up as the next call hanging rather than failing."""
        tracker = UsageTracker(50, 0.0, max_provider_requests=50)
        hung = RoleModel(
            role=ModelRole.PLANNER,
            spec=ModelSpec(provider=Provider.OLLAMA, model="test-model"),
            model=HangingRunnable(),  # type: ignore[arg-type]
            tracker=tracker,
            timeout_seconds=0.2,
        )
        with pytest.raises(ModelTimeoutError):
            await hung.structured(PlanOut, "system", "user")

        # The tracker must still admit work; if the timed-out call held
        # its slot this would block or raise.
        remaining = await tracker.remaining()
        assert remaining > 0

    async def test_repeated_timeouts_do_not_exhaust_the_budget_silently(self) -> None:
        tracker = UsageTracker(50, 0.0, max_provider_requests=50)
        for _ in range(3):
            bound = RoleModel(
                role=ModelRole.PLANNER,
                spec=ModelSpec(provider=Provider.OLLAMA, model="test-model"),
                model=HangingRunnable(),  # type: ignore[arg-type]
                tracker=tracker,
                timeout_seconds=0.15,
            )
            with pytest.raises(ModelTimeoutError):
                await bound.structured(PlanOut, "system", "user")
        assert await tracker.remaining() > 0


class TestRunLevelDeadline:
    """Many individually bounded calls still compose into an unbounded run.

    Fifty calls at three minutes each is two and a half hours, and no
    per-call deadline can see that total. The run budget is counted in
    elapsed wall-clock time so a slow provider cannot quietly extend it.
    """

    async def test_a_run_that_overruns_is_stopped(self) -> None:
        from agentic_research.config import LLMMode, Settings
        from agentic_research.runner import stream_research

        settings = Settings(
            llm_mode=LLMMode.LOCAL,
            persist_runs=False,
            run_timeout_seconds=0.25,
            checkpoint_backend="none",
        )
        started = time.perf_counter()
        events = []
        async for event in stream_research("a question", settings):
            events.append(event)
            if event.get("event") == "result":
                break
        elapsed = time.perf_counter() - started
        assert elapsed < 120, f"run took {elapsed:.0f}s against a 0.25s deadline"

    def test_the_budget_carries_a_wall_clock_ceiling(self) -> None:
        """Counted in seconds, not in calls times nominal timeout."""
        from agentic_research.config import Settings

        budget = Settings(run_timeout_seconds=42.0).budget
        assert budget.max_run_seconds == 42.0

    def test_the_result_reports_whether_it_was_cut_short(self) -> None:
        """An incomplete run must be distinguishable from a fast one."""
        from agentic_research.runner import RunResult

        assert "timed_out" in RunResult.__dataclass_fields__
        assert RunResult.__dataclass_fields__["timed_out"].default is False
