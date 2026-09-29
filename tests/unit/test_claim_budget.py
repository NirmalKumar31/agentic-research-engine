"""The report is bounded by what claims cost, and they cost nothing.

This file used to assert the opposite, correctly, for a design that no
longer exists. When entailment was a generative call, every
substantive claim cost one, and a live run generated 25 claims, could
afford to check 4, and published 2. Sizing the report to the
verification budget was the fix for that.

The NLI classifier replaced the generative verifier and is not a model
call. The two calls that follow synthesis -- the relevance judgement
and the wording repair -- are each batched over the whole report. So
an extra claim costs no additional call, and the budget kept capping
the report against a cost that had been removed: on the hosted demo it
told the synthesiser to write at most 7 claims from 32 evidence items.

What bounds the report now is the prompt and the gates. A spend
ceiling does not stand in for them.
"""

from __future__ import annotations

import inspect

import pytest

from agentic_research.graph.nodes.reporting import _POST_SYNTHESIS_CALLS, _claim_budget
from agentic_research.graph.prompts import synthesizer_user


class FakeTracker:
    def __init__(self, remaining: int) -> None:
        self._remaining = remaining

    async def remaining(self) -> int:
        return self._remaining


class FakeRouter:
    def __init__(self, remaining: int) -> None:
        self.tracker = FakeTracker(remaining)


class FakeContext:
    def __init__(self, remaining: int) -> None:
        self.router = FakeRouter(remaining)


@pytest.fixture
def budget_with(monkeypatch: pytest.MonkeyPatch):
    def _set(remaining: int):
        monkeypatch.setattr(
            "agentic_research.graph.nodes.reporting.ctx", lambda: FakeContext(remaining)
        )

    return _set


class TestClaimsAreNotRationed:
    async def test_the_demo_budget_no_longer_caps_the_report(self, budget_with) -> None:
        """The measured case. MAX_LLM_CALLS is 20 and research spends
        about ten before synthesis, which used to yield "write at most
        7" while 32 evidence items sat in the prompt."""
        budget_with(10)
        assert await _claim_budget() is None

    @pytest.mark.parametrize("remaining", [3, 5, 10, 50])
    async def test_any_run_that_can_finish_is_unbounded(self, budget_with, remaining: int) -> None:
        budget_with(remaining)
        assert await _claim_budget() is None

    async def test_the_exact_boundary_is_synthesis_plus_its_followers(self, budget_with) -> None:
        budget_with(1 + _POST_SYNTHESIS_CALLS)
        assert await _claim_budget() is None


class TestARunThatCannotFinishSynthesisesNothing:
    @pytest.mark.parametrize("remaining", [0, 1, 2])
    async def test_it_returns_zero(self, budget_with, remaining: int) -> None:
        """Zero is not "a short report". Without the relevance
        judgement every claim is withheld, so synthesising would spend
        a call to publish nothing; the caller emits the evidence
        listing instead."""
        budget_with(remaining)
        assert await _claim_budget() == 0

    async def test_zero_and_unbounded_are_the_only_answers(self, budget_with) -> None:
        """Non-vacuity for the rewrite: any intermediate number would
        mean something still rations claims."""
        seen = set()
        for remaining in range(0, 40):
            budget_with(remaining)
            seen.add(await _claim_budget())
        assert seen == {0, None}, seen


class TestTheReasonIsRecorded:
    def test_the_cost_it_prices_is_named(self) -> None:
        """A future reader has to be able to tell whether this is
        stale again. It was stale for three releases because the
        docstring described a cost that had been removed."""
        doc = _claim_budget.__doc__ or ""
        assert "not a model call" in doc
        assert "batched" in doc

    def test_nothing_reintroduces_a_per_claim_divisor(self) -> None:
        body = inspect.getsource(_claim_budget)
        assert "//" not in body and "/" not in body.split('"""')[-1]


class TestThePromptStillBoundsTheReport:
    def test_an_unbounded_budget_omits_the_cap_sentence(self) -> None:
        assert "Write at most" not in synthesizer_user("q", "overview", "e", "")

    def test_a_budget_is_stated_when_there_is_one(self) -> None:
        assert "Write at most 3 substantive claims" in synthesizer_user(
            "q", "overview", "e", "", claim_budget=3
        )

    def test_quality_over_quantity_survives(self) -> None:
        """The bound that remains. Removing the spend cap must not
        remove the instruction that fewer well-evidenced claims beat
        more thin ones."""
        text = synthesizer_user("q", "overview", "e", "", claim_budget=3)
        assert "fewer well-evidenced claims beat more thinly-evidenced ones" in text
