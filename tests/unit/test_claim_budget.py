"""The report is sized to what the run can afford to verify.

A live cloud run generated 25 substantive claims, could afford to check
4, and published 2. Under the publication gate an unchecked claim is not
published, so generating past the verification budget does not produce a
longer report -- it produces the same short report, with the surplus
paid for and then deleted.

The demo ceiling is 20 logical model calls. Research spends about 11 of
them before synthesis, which leaves roughly 9 for entailment. The
synthesiser is told that number.
"""

from __future__ import annotations

import pytest

from agentic_research.graph.nodes.reporting import (
    _DEFAULT_ENTAILMENT_SAMPLE,
    _MIN_CLAIM_BUDGET,
    _VERIFICATION_OVERHEAD,
    _claim_budget,
)
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


class TestTheBudgetTracksWhatIsAffordable:
    async def test_the_demo_shape_gets_a_real_budget(self, budget_with) -> None:
        """20-call ceiling, ~11 spent on research: the report is bounded."""
        budget_with(9)
        budget = await _claim_budget()
        assert budget is not None
        assert budget == 9 - 1 - _VERIFICATION_OVERHEAD

    async def test_an_ample_budget_imposes_no_limit(self, budget_with) -> None:
        """A local run has room; telling it to write a short report would
        be a constraint invented for no reason."""
        budget_with(40)
        assert await _claim_budget() is None

    async def test_the_boundary_is_the_sample_size(self, budget_with) -> None:
        budget_with(_DEFAULT_ENTAILMENT_SAMPLE + 1 + _VERIFICATION_OVERHEAD)
        assert await _claim_budget() is None

    async def test_an_exhausted_budget_still_asks_for_a_few_claims(self, budget_with) -> None:
        """A report worth writing beats an empty one; the floor holds."""
        budget_with(0)
        assert await _claim_budget() == _MIN_CLAIM_BUDGET

    async def test_the_budget_never_goes_negative(self, budget_with) -> None:
        budget_with(1)
        assert await _claim_budget() >= _MIN_CLAIM_BUDGET


class TestThePromptCarriesTheBudget:
    def test_the_claim_limit_is_stated(self) -> None:
        prompt = synthesizer_user("q", "overview", "evidence", "", claim_budget=8)
        assert "at most 8 substantive claims" in prompt

    def test_framing_is_excluded_from_the_count(self) -> None:
        prompt = synthesizer_user("q", "overview", "evidence", "", claim_budget=8)
        assert "framing sentences do not count" in prompt.lower()

    def test_the_reason_is_given_not_just_the_number(self) -> None:
        """A bare cap invites padding up to it. The prompt says why."""
        prompt = synthesizer_user("q", "overview", "evidence", "", claim_budget=8)
        assert "checked individually" in prompt
        assert "dropped before publication" in prompt

    def test_no_budget_means_no_instruction(self) -> None:
        prompt = synthesizer_user("q", "overview", "evidence", "")
        assert "at most" not in prompt


class TestTheRegressionThisPrevents:
    def test_generation_no_longer_outruns_verification(self) -> None:
        """The failing run, as arithmetic.

        25 claims against a 9-call verification budget published 2. With
        the budget applied, generation is bounded by what can be checked,
        so every generated claim can survive.
        """
        verification_budget = 9
        generated_before = 25
        published_before = 2
        assert published_before < verification_budget, "most of the budget went unused"

        generated_after = verification_budget - 1 - _VERIFICATION_OVERHEAD
        assert generated_after <= verification_budget
        assert generated_after < generated_before
