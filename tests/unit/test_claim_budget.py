"""The report is bounded by the contract, not by the call budget.

Two bounds, and confusing them cost a release each way.

When entailment was a generative call, every claim cost one, and a run
generated 25 claims, could afford 4, and published 2. Sizing the report
to the verification budget was the fix. The NLI classifier replaced
that verifier and is not a model call; the judgement and repair are
batched. So the spend justification evaporated, and the cap went on
capping against a cost that no longer existed -- on the hosted demo,
"write at most 7" with 32 evidence items in the prompt.

Removing the justification was right. Removing the cap with it was not.
Measured on the deployment, same question, same everything else:

    cap of 7    5 claims generated, 1 published
    no cap     13 claims generated, 0 published

The cap had a second job nobody had written down. Unbounded, the
synthesiser wrote thin claims until the evidence ran out, and a larger
batch of thin claims fared worse at the relevance gate than a smaller
batch of considered ones.

So the bound stays and is tied to the contract, which is the thing that
says how much answer was asked for. It is a quality bound now, and
labelled as one.
"""

from __future__ import annotations

import pytest

from agentic_research.graph.nodes.reporting import (
    _CLAIMS_PER_SLOT,
    _CLAIMS_WITHOUT_A_CONTRACT,
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


class TestTheContractSizesTheReport:
    @staticmethod
    def comparison():
        from agentic_research.answer_contract import QuestionType, build_contract

        return build_contract(
            "How does a large language model differ from a neural network?",
            QuestionType.COMPARISON,
            entities=("large language model", "neural network"),
        )

    async def test_it_allows_two_claims_per_part(self, budget_with) -> None:
        budget_with(20)
        contract = self.comparison()
        assert await _claim_budget(contract) == _CLAIMS_PER_SLOT * len(contract.required_slots)

    async def test_a_smaller_contract_gets_a_smaller_report(self, budget_with) -> None:
        """Non-vacuity: the bound has to vary with the contract, or it
        is a constant wearing a contract's clothes."""
        from agentic_research.answer_contract import QuestionType, build_contract

        budget_with(20)
        definition = build_contract(
            "What is a vector database?", QuestionType.DEFINITION, entities=("vector database",)
        )
        assert await _claim_budget(definition) < await _claim_budget(self.comparison())

    async def test_it_does_not_grow_with_the_call_budget(self, budget_with) -> None:
        """The defect this replaced. How many calls remain says nothing
        about how much answer was asked for."""
        contract = self.comparison()
        budget_with(8)
        small = await _claim_budget(contract)
        budget_with(400)
        assert await _claim_budget(contract) == small

    async def test_an_unusable_contract_falls_back_to_a_bounded_report(self, budget_with) -> None:
        """Still bounded. Thirteen thin claims published nothing."""
        from agentic_research.answer_contract import build_contract

        budget_with(20)
        unusable = build_contract("compare them", "comparison", entities=["only one"])
        assert not unusable.usable
        assert await _claim_budget(unusable) == _CLAIMS_WITHOUT_A_CONTRACT

    async def test_no_contract_is_bounded_too(self, budget_with) -> None:
        budget_with(20)
        assert await _claim_budget(None) == _CLAIMS_WITHOUT_A_CONTRACT

    async def test_the_report_is_never_unbounded(self, budget_with) -> None:
        """The measured regression, asserted directly."""
        from agentic_research.answer_contract import build_contract

        for remaining in (3, 10, 50, 500):
            budget_with(remaining)
            for contract in (None, self.comparison(), build_contract("q", "definition")):
                assert await _claim_budget(contract) is not None


class TestARunThatCannotFinishSynthesisesNothing:
    @pytest.mark.parametrize("remaining", [0, 1, 2])
    async def test_it_returns_zero(self, budget_with, remaining: int) -> None:
        """Zero is not "a short report". Without the relevance
        judgement every claim is withheld, so synthesising would spend
        a call to publish nothing; the caller emits the evidence
        listing instead."""
        budget_with(remaining)
        assert await _claim_budget(None) == 0

    async def test_affordability_outranks_the_contract(self, budget_with) -> None:
        from agentic_research.answer_contract import QuestionType, build_contract

        budget_with(0)
        rich = build_contract(
            "How does A differ from B?", QuestionType.COMPARISON, entities=("A", "B")
        )
        assert await _claim_budget(rich) == 0


class TestItIsARequestNotACeiling:
    """The name invites the wrong reading, so the docstring has to
    correct it and this has to hold the docstring to it.

    The number reaches the synthesiser as "write at most N" and the
    synthesiser may write more. A local run asked for six and produced
    eight. Nothing trims the surplus, on purpose: every claim is gated
    individually and an extra one costs no model call, so exceeding
    the request is untidy rather than unsafe -- while truncating to a
    count would discard claims before anything had looked at them, and
    the one filling the required part is as likely to go as any other.
    """

    def test_the_docstring_does_not_promise_enforcement(self) -> None:
        doc = _claim_budget.__doc__ or ""
        assert "request, not a ceiling" in doc
        assert "may write more" in doc

    def test_the_measurement_is_recorded(self) -> None:
        """Thirteen unbounded, eight when asked for six. A request that
        changes nothing would be worth removing; this one is not."""
        doc = _claim_budget.__doc__ or ""
        assert "thirteen" in doc and "eight" in doc

    def test_nothing_truncates_the_report(self) -> None:
        """Non-vacuity for the reasoning above: if a trim were added,
        this is where it would surface."""
        import inspect

        from agentic_research.graph.nodes import reporting

        body = inspect.getsource(reporting.synthesize_report)
        assert "[:claim_budget]" not in body
        assert "claim_budget]" not in body


class TestTheReasonIsRecorded:
    def test_both_bounds_are_named(self) -> None:
        """This was stale for three releases because the docstring
        described a cost that had been deleted. It now has to say
        which bound is which."""
        doc = _claim_budget.__doc__ or ""
        assert "quality" in doc
        assert "not a model call" in doc or "batched" in doc

    def test_the_measurement_is_recorded(self) -> None:
        doc = _claim_budget.__doc__ or ""
        assert "13" in doc and "0 published" in doc


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
