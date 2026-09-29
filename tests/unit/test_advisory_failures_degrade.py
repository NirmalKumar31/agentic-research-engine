"""A run must not die because an advisory model call did.

A hosted run on v1.4.0 reached `assessing_coverage` with six sources
and twenty-eight extracted quotes, then emitted an error instead of a
report. The coverage critique is advice: every number that routes the
run is computed from evidence already in hand before the model is
called, and the node's own comment said "the counted half still
stands, so routing remains sound".

It caught `LLMError`. Anything else ended the run -- four searches and
twenty-eight extractions paid for, discarded, to lose an opinion.

Six other calls had the same shape: a fallback that fires only for the
failure type someone happened to anticipate. That is a promise the
code does not keep, and these tests are the promise written down.

What must NOT be swallowed is the wall-clock deadline. `asyncio.timeout`
cancels the inner task with `CancelledError`, which derives from
`BaseException`, so `except Exception` cannot catch it. That is the
property that makes widening these safe, and it is asserted here
rather than assumed.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from agentic_research.graph.nodes import critique, planning, reporting

# Each entry: the function whose degradation path must be unconditional.
ADVISORY = [
    (critique.assess_coverage, "coverage critique"),
    (planning.analyze_query, "question analysis"),
    (planning.plan_research, "research planning"),
    (planning.generate_queries, "query generation"),
    (planning.generate_followups, "follow-up generation"),
    (reporting.synthesize_report, "synthesis"),
    (reporting._repair_wording, "wording repair"),
    (reporting._judge_relevance, "relevance judgement"),
]


class TestEveryFallbackFiresForAnyFailure:
    @pytest.mark.parametrize(("func", "label"), ADVISORY, ids=[label for _, label in ADVISORY])
    def test_it_does_not_catch_only_llmerror(self, func, label: str) -> None:
        """Read from the source, because the alternative is driving
        seven nodes through seven different failure injections to
        assert one property about all of them."""
        body = inspect.getsource(func)
        assert "except LLMError" not in body, (
            f"{label} degrades only for LLMError; any other failure ends the run"
        )
        assert "except Exception" in body, f"{label} has no unconditional degradation"

    @pytest.mark.parametrize(("func", "label"), ADVISORY, ids=[label for _, label in ADVISORY])
    def test_the_exception_type_is_recorded(self, func, label: str) -> None:
        """A broad catch that logged nothing would trade a lost run for
        a hidden defect. The type reaches the log and the run's errors,
        so a bug here shows up as a degraded run rather than silence."""
        assert "error_type=type(exc).__name__" in inspect.getsource(func), label


class TestTheDeadlineStillStopsARun:
    def test_cancellation_is_not_an_exception(self) -> None:
        """The property the widening depends on. If CancelledError
        derived from Exception, every one of these handlers would
        swallow the wall-clock ceiling."""
        assert not issubclass(asyncio.CancelledError, Exception)
        assert issubclass(asyncio.CancelledError, BaseException)

    async def test_a_broad_handler_does_not_survive_a_timeout(self) -> None:
        """End to end on the real construct the runner uses."""
        caught = False

        async def work() -> None:
            nonlocal caught
            try:
                await asyncio.sleep(10)
            except Exception:  # what the nodes now do
                caught = True

        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await work()
        assert not caught, "an `except Exception` handler swallowed the deadline"

    def test_the_runner_still_uses_that_construct(self) -> None:
        """Pins the assumption to the code rather than to this file."""
        from agentic_research import runner

        assert "asyncio.timeout(" in inspect.getsource(runner.stream_research)
