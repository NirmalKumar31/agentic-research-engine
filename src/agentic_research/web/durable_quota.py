"""The run counter that admits or refuses a public live run.

The global daily cap is held in an external atomic counter rather than
in process memory. In memory it bounded a *process* rather than a day:
a Render instance that sleeps when idle wakes with a fresh count, so
the cap reset itself every time the service went quiet.

One atomic INCR against a shared store is the whole mechanism. INCR is
atomic across processes and replicas, so two web instances cannot both
read "the last slot is free" and both take it. Increment first, compare
afterwards -- read-then-write is the race this exists to avoid.

What the free Render Key Value plan gives, precisely: a counter that is
atomic, shared across every web replica, and unaffected by a web-service
cold start. What it does not give is persistence. Render states that
data persistence is unavailable on free Key Value instances, so a
restart of the *store itself* returns the day's allowance to zero used.
Call this counter shared or distributed; do not call it durable across
datastore restarts unless it is running on a paid plan with persistence
enabled. The hard-enforced spend limit on the provider account is the
financial backstop, not this.

It fails closed. If a durable counter is required and the store is
unreachable, the run is refused rather than admitted on an in-memory
guess -- an unbounded fallback is exactly the failure this exists to
prevent. Readiness asks the same store whether it is still reachable,
without consuming a run.

The in-process limiter stays as a second control for concurrency and
per-client rate. Those remain process-local, deliberately: they bound
accidents and abuse, and the money is bounded here.

The key is namespaced. Two deployments built from this image compute
the same key, so a staging service sharing a store with production
would spend production's day -- quietly, since nothing errors. The
namespace comes from ``DEMO_QUOTA_NAMESPACE``, defaults to empty so an
existing deployment's key is unchanged, and cannot contain the key
separator, so no namespace can address another's counter.
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
    """What a shared counter has to be able to do.

    ``healthy`` exists so readiness can ask whether the store is still
    there without spending a run to find out. Reachability at startup
    proves nothing about reachability now, and a readiness probe that
    incremented the counter would answer the question by consuming the
    thing it was asked about.
    """

    def incr(self, key: str) -> int: ...
    def expire(self, key: str, seconds: int) -> None: ...
    def healthy(self) -> bool: ...


@dataclass(frozen=True)
class QuotaDecision:
    allowed: bool
    used: int
    limit: int
    detail: str = ""


class DurableRunQuota:
    """Global daily run cap backed by an external atomic counter."""

    def __init__(
        self,
        counter: AtomicCounter | None,
        limit: int,
        *,
        required: bool,
        namespace: str = "",
    ) -> None:
        self._counter = counter
        self._limit = limit
        self._required = required
        self._namespace = namespace

    @staticmethod
    def key(now: datetime | None = None, namespace: str = "") -> str:
        """The counter key for one UTC day in one namespace.

        The date goes last, always. Two deployments sharing a store
        must not be able to address each other's counter, and putting
        the namespace in the middle makes that structural: a namespace
        cannot contain a colon (Settings enforces it), and no date
        contains one either, so no namespaced key can ever spell an
        un-namespaced key or another namespace's key.

        An empty namespace yields the un-namespaced key deliberately.
        It is what a deployment that predates this setting is already
        counting against, so adding the setting does not reset a live
        counter halfway through a day.
        """
        day = (now or datetime.now(UTC)).strftime("%Y-%m-%d")
        if namespace:
            return f"{_KEY_PREFIX}:{namespace}:{day}"
        return f"{_KEY_PREFIX}:{day}"

    def usable(self) -> bool:
        """Whether a reservation could succeed, without making one.

        Readiness needs to answer "could a live run start?" and must not
        answer it by consuming a run from the day's allowance, so this
        pings rather than increments.

        It asks the store every time. Holding on to the fact that the
        store answered at startup is how an instance keeps reporting
        itself ready while every live request fails closed against a
        Key Value service that died an hour ago.

        Only a literal ``True`` counts as reachable. A store that
        answers with something else is as unusable as one that does not
        answer, and treating a truthy string as health is how a broken
        adapter passes for a working one.
        """
        if self._limit <= 0:
            return False
        if self._counter is None:
            return not self._required

        try:
            reachable = self._counter.healthy() is True
        except Exception as exc:
            log.warning("quota_store_health_failed", error=type(exc).__name__)
            reachable = False

        if reachable:
            return True
        # Unreachable and not required means reserve() would admit the
        # run anyway, so reporting unavailable here would contradict it.
        return not self._required

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

        key = self.key(namespace=self._namespace)
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


# Short on purpose. Readiness is answered on a request path, and a
# readiness probe that hangs for the client timeout is its own outage.
_HEALTH_TIMEOUT_SECONDS = 2.0


class RedisCounter:
    """Adapts a Redis client to :class:`AtomicCounter`.

    The adapter exists so nothing outside this module has to know the
    store is Redis. The API layer asks a counter whether it is healthy;
    it does not reach into a client object and call PING itself.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def incr(self, key: str) -> int:
        return int(self._client.incr(key))

    def expire(self, key: str, seconds: int) -> None:
        self._client.expire(key, seconds)

    def healthy(self) -> bool:
        """Bounded, non-mutating reachability check.

        PING changes nothing and costs nothing, so readiness can ask it
        as often as it likes without touching the day's allowance.
        """
        try:
            response = self._client.ping()
        except Exception as exc:
            # The URL is never included: it carries the password.
            log.warning("quota_store_ping_failed", error=type(exc).__name__)
            return False
        return response is True or response in (b"PONG", "PONG")


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
        client: Any = redis.Redis.from_url(
            url,
            socket_timeout=_HEALTH_TIMEOUT_SECONDS,
            socket_connect_timeout=_HEALTH_TIMEOUT_SECONDS,
        )
        client.ping()
    except Exception as exc:
        # Deliberately only the exception type. The connection string
        # contains the password and must never reach a log line.
        log.error("quota_store_unreachable", error=type(exc).__name__)
        return None
    return RedisCounter(client)
