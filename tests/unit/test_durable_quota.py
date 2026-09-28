"""The live-research run cap has to outlive the process.

The in-memory counter bounds a process, not a day: a Render free
instance sleeps when idle and wakes with the count at zero. For replay
that is harmless, because a run costs nothing. For live research each
run spends at three providers, so the cap must be durable and atomic --
two replicas must not both see the last free slot.
"""

from __future__ import annotations

import pytest

from agentic_research.web.durable_quota import DurableRunQuota, build_counter


class FakeCounter:
    """An atomic counter, shared the way Redis would be."""

    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.expiries: dict[str, int] = {}

    def incr(self, key: str) -> int:
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    def expire(self, key: str, seconds: int) -> None:
        self.expiries[key] = seconds


class BrokenCounter:
    def incr(self, key: str) -> int:
        raise ConnectionError("store unreachable")

    def expire(self, key: str, seconds: int) -> None:  # pragma: no cover
        raise ConnectionError("store unreachable")


class TestTheCapIsEnforced:
    def test_runs_up_to_the_limit_are_allowed(self) -> None:
        quota = DurableRunQuota(FakeCounter(), limit=3, required=True)
        assert [quota.reserve().allowed for _ in range(3)] == [True, True, True]

    def test_the_run_after_the_limit_is_refused(self) -> None:
        quota = DurableRunQuota(FakeCounter(), limit=2, required=True)
        for _ in range(2):
            quota.reserve()
        decision = quota.reserve()
        assert not decision.allowed
        assert decision.used == 2 and decision.limit == 2

    def test_two_instances_share_one_budget(self) -> None:
        """The property the in-memory counter cannot provide: a second
        replica sees the first replica's spend."""
        counter = FakeCounter()
        a = DurableRunQuota(counter, limit=2, required=True)
        b = DurableRunQuota(counter, limit=2, required=True)
        assert a.reserve().allowed
        assert b.reserve().allowed
        assert not a.reserve().allowed, "the second instance had its own budget"

    def test_it_increments_before_comparing(self) -> None:
        """Read-then-write would let two callers both observe the last
        free slot. Exactly one of many concurrent callers may win."""
        counter = FakeCounter()
        quotas = [DurableRunQuota(counter, limit=1, required=True) for _ in range(10)]
        granted = sum(q.reserve().allowed for q in quotas)
        assert granted == 1

    def test_the_key_expires_so_the_day_resets(self) -> None:
        counter = FakeCounter()
        DurableRunQuota(counter, limit=5, required=True).reserve()
        assert counter.expiries, "the counter would accumulate forever"

    def test_the_key_is_dated(self) -> None:
        assert DurableRunQuota.key().endswith(
            __import__("datetime").datetime.now(__import__("datetime").UTC).strftime("%Y-%m-%d")
        )


class TestItFailsClosed:
    def test_a_missing_store_refuses_when_durability_is_required(self) -> None:
        """An unbounded fallback is the failure this exists to prevent."""
        decision = DurableRunQuota(None, limit=5, required=True).reserve()
        assert not decision.allowed
        assert "unavailable" in decision.detail

    def test_an_unreachable_store_refuses_when_required(self) -> None:
        decision = DurableRunQuota(BrokenCounter(), limit=5, required=True).reserve()
        assert not decision.allowed
        assert "unreachable" in decision.detail

    def test_a_zero_limit_refuses_everything(self) -> None:
        assert not DurableRunQuota(FakeCounter(), limit=0, required=True).reserve().allowed

    def test_replay_does_not_require_a_store(self) -> None:
        """Replay spends nothing, so it must not need infrastructure."""
        assert DurableRunQuota(None, limit=5, required=False).reserve().allowed


class TestCounterConstruction:
    def test_no_url_means_no_counter(self) -> None:
        assert build_counter(None) is None
        assert build_counter("") is None

    def test_an_unreachable_url_yields_no_counter_rather_than_raising(self) -> None:
        """Construction must not crash boot; the caller decides whether
        the absence is fatal."""
        assert build_counter("redis://127.0.0.1:1/0") is None


@pytest.mark.parametrize("limit", [1, 2, 5, 20])
def test_exactly_the_limit_is_granted(limit: int) -> None:
    counter = FakeCounter()
    quota = DurableRunQuota(counter, limit=limit, required=True)
    granted = sum(quota.reserve().allowed for _ in range(limit * 3))
    assert granted == limit
