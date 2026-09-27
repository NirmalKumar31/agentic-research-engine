"""A run counter that survives a restart.

The public demo's daily cap lives in process memory. A Render free
instance sleeps when idle and starts with a fresh counter, so the cap
bounds a *process*, not a day -- which is fine for replay, where a run
costs nothing, and not fine for live research, where each run spends
money at three providers.

This is the durable version: one atomic INCR against a shared store,
with an expiry so the key disappears on its own. INCR is atomic across
processes and replicas, so two instances cannot both read "3 used" and
both proceed.

It fails closed. If live research requires a durable counter and the
store is unreachable, the run is refused rather than admitted on an
in-memory guess -- an unbounded fallback is exactly the failure this
exists to prevent.

The in-process limiter stays as a second control for concurrency and
per-client rate. This replaces only the global daily cap, which is the
one that must outlive the process.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from agentic_research.observability import get_logger

log = get_logger(__name__)

_KEY_PREFIX = "are:live-runs"
_TTL_SECONDS = 60 * 60 * 36  # comfortably past a UTC day boundary


class AtomicCounter(Protocol):
    """The two operations a durable counter needs."""

    def incr(self, key: str) -> int: ...
    def expire(self, key: str, seconds: int) -> None: ...


@dataclass(frozen=True)
class QuotaDecision:
    allowed: bool
    used: int
    limit: int
    detail: str = ""


class DurableRunQuota:
    """Global daily run cap backed by an external atomic counter."""

    def __init__(self, counter: AtomicCounter | None, limit: int, *, required: bool) -> None:
        self._counter = counter
        self._limit = limit
        self._required = required

    @staticmethod
    def key(now: datetime | None = None) -> str:
        day = (now or datetime.now(UTC)).strftime("%Y-%m-%d")
        return f"{_KEY_PREFIX}:{day}"

    def usable(self) -> bool:
        """Whether a reservation could succeed, without making one.

        Readiness needs to answer "could a live run start?" and must not
        answer it by consuming a run from the day's allowance.
        """
        if self._limit <= 0:
            return False
        if self._counter is None:
            return not self._required
        return True

    def reserve(self) -> QuotaDecision:
        """Claim one run, atomically, before any provider is called.

        Increments first and compares afterwards. Reading then writing
        would let two replicas both observe the last free slot.
        """
        if self._limit <= 0:
            return QuotaDecision(False, 0, self._limit, "live research is disabled")

        if self._counter is None:
            if self._required:
                # No durable store and one is required: refuse rather
                # than fall back to a counter a restart would reset.
                log.error("durable_quota_unavailable", required=True)
                return QuotaDecision(False, 0, self._limit, "durable quota storage is unavailable")
            return QuotaDecision(True, 0, self._limit, "durable quota not configured")

        key = self.key()
        try:
            # int() inside the try on purpose. A store that answers with
            # something other than a number is as broken as one that
            # cannot be reached, and comparing it outside would raise
            # TypeError straight out of the reservation -- past every
            # fail-closed branch below and into the request handler.
            used = int(self._counter.incr(key))
            if used == 1:
                self._counter.expire(key, _TTL_SECONDS)
        except Exception as exc:
            log.error("durable_quota_error", error=type(exc).__name__)
            if self._required:
                return QuotaDecision(False, 0, self._limit, "quota store unreachable")
            return QuotaDecision(True, 0, self._limit, "quota store unreachable, not required")

        if used > self._limit:
            return QuotaDecision(False, used - 1, self._limit, "daily run limit reached")
        return QuotaDecision(True, used, self._limit)


def build_counter(url: str | None) -> AtomicCounter | None:
    """Connect to the configured store, or return None.

    Import is local so the dependency is optional: a replay deployment
    needs no quota store and should not need the client library either.
    """
    if not url:
        return None
    try:
        import redis
    except ImportError:
        log.warning("redis_client_missing")
        return None
    try:
        client: Any = redis.Redis.from_url(url, socket_timeout=3, socket_connect_timeout=3)
        client.ping()
    except Exception as exc:
        log.error("redis_unreachable", error=type(exc).__name__)
        return None
    return client
