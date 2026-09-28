"""What a run is allowed to believe about its own cost.

Two ways the cost figure could be wrong while looking authoritative.

A dated snapshot inherited its family's price by prefix, so
`gpt-6-sol-2027-01-01` would have been billed at today's `gpt-6-sol`
rate with nobody having checked whether that is what the provider
charges. Reporting may still resolve prefixes -- a historical artifact
names models that are no longer listed -- but spending may not.

Detailed usage categories were read from the response and discarded, so
a reply billed partly at a cached-input rate was reconciled as if every
token were charged at the full one.
"""

from __future__ import annotations

import pytest

from agentic_research.config import Provider
from agentic_research.llm.pricing import (
    ModelPrice,
    PriceUnavailable,
    get_price,
    require_price,
)

PRICED = "gpt-6-luna"
DATED = "gpt-6-luna-2027-01-01"


class TestSpendingNeedsAReviewedPrice:
    def test_an_exact_model_is_priced(self) -> None:
        assert require_price(Provider.OPENAI, PRICED).input_per_mtok == 0.1

    def test_a_dated_snapshot_is_refused(self) -> None:
        """The defect: a model nobody has priced must stop the run, not
        quietly borrow a rate from a name it happens to start with."""
        with pytest.raises(PriceUnavailable, match="no reviewed price"):
            require_price(Provider.OPENAI, DATED)

    def test_the_refusal_says_how_to_fix_it(self) -> None:
        with pytest.raises(PriceUnavailable) as caught:
            require_price(Provider.OPENAI, DATED)
        assert "aliases.openai" in str(caught.value)

    def test_reporting_still_resolves_a_prefix(self) -> None:
        """A stored artifact may name a model no longer in the table.
        Reporting its cost is fine; authorising spend on it is not."""
        assert get_price(Provider.OPENAI, DATED) is not None

    def test_an_unknown_model_is_refused(self) -> None:
        with pytest.raises(PriceUnavailable):
            require_price(Provider.OPENAI, "not-a-model-at-all")

    def test_a_reviewed_alias_resolves(self, tmp_path, monkeypatch) -> None:
        """The escape hatch: one line, written by a person who looked."""
        from agentic_research.llm import pricing

        table = tmp_path / "pricing.toml"
        table.write_text(
            '[openai]\n"base-model" = { input = 1.0, output = 2.0 }\n'
            '[aliases.openai]\n"base-model-2027-01-01" = "base-model"\n'
        )
        monkeypatch.chdir(tmp_path)
        pricing.reset_cache()
        try:
            assert require_price(Provider.OPENAI, "base-model-2027-01-01").input_per_mtok == 1.0
        finally:
            monkeypatch.undo()
            pricing.reset_cache()


class TestReconcilingDetailedUsage:
    FLAT = ModelPrice(input_per_mtok=10.0, output_per_mtok=20.0)
    WITH_CACHE = ModelPrice(input_per_mtok=10.0, output_per_mtok=20.0, cached_input_per_mtok=1.0)

    def test_a_plain_response_is_priced_and_complete(self) -> None:
        usd, complete = self.FLAT.reconcile(1_000_000, 1_000_000)
        assert usd == pytest.approx(30.0)
        assert complete is True

    def test_cached_tokens_use_the_cached_rate_when_known(self) -> None:
        usd, complete = self.WITH_CACHE.reconcile(1_000_000, 0, cached_input_tokens=500_000)
        # Half at 10, half at 1.
        assert usd == pytest.approx(5.5)
        assert complete is True

    def test_cached_tokens_without_a_rate_charge_full_and_flag_incomplete(self) -> None:
        """Charging the higher rate cannot understate the bill; saying
        the figure is complete would."""
        usd, complete = self.FLAT.reconcile(1_000_000, 0, cached_input_tokens=500_000)
        assert usd == pytest.approx(10.0)
        assert complete is False

    def test_cache_writes_without_a_rate_flag_incomplete(self) -> None:
        _usd, complete = self.FLAT.reconcile(100, 100, cache_write_tokens=50)
        assert complete is False

    def test_cached_cannot_exceed_input(self) -> None:
        """A provider reporting more cached tokens than input tokens is
        malformed; it must not produce a negative charge."""
        usd, _ = self.WITH_CACHE.reconcile(100, 0, cached_input_tokens=10_000)
        assert usd >= 0

    def test_the_reservation_rate_is_the_higher_one(self) -> None:
        """Cache status is unknowable before dispatch, so reserving must
        assume no discount."""
        reserved = self.WITH_CACHE.cost(1_000_000, 0)
        best_case, _ = self.WITH_CACHE.reconcile(1_000_000, 0, cached_input_tokens=1_000_000)
        assert reserved >= best_case


class TestTotalsRefuseToLookComplete:
    def _attempt(self, **over):
        from agentic_research.config import ModelRole
        from agentic_research.llm.base import AttemptKind, ProviderAttempt

        kwargs: dict = {
            "role": ModelRole.PLANNER,
            "provider": Provider.OPENAI,
            "model": PRICED,
            "schema": "S",
            "kind": AttemptKind.INITIAL,
            "latency_s": 0.1,
            "input_tokens": 1000,
            "output_tokens": 100,
        }
        kwargs.update(over)
        return ProviderAttempt(**kwargs)  # type: ignore[arg-type]

    def test_a_plain_attempt_is_complete(self) -> None:
        assert self._attempt().cost_is_complete is True

    def test_an_attempt_with_unpriced_cached_tokens_is_not(self) -> None:
        """gpt-6-luna has no recorded cached rate, so a response using
        the cache makes the run's cost unverified."""
        assert self._attempt(cached_input_tokens=400).cost_is_complete is False

    def test_reasoning_tokens_are_not_charged_twice(self) -> None:
        """Providers report them inside output_tokens."""
        plain = self._attempt().cost_usd
        with_reasoning = self._attempt(reasoning_tokens=50).cost_usd
        assert plain == with_reasoning
