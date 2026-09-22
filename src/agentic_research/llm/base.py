"""Call records, usage accounting and LLM-layer errors.

Everything the engine knows about what a run cost is produced here. Because
every model call goes through one wrapper, no call site can forget to report
its tokens, and adding a new node cannot silently create an unmetered cost.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field

from agentic_research.config import ModelRole, ModelSpec, Provider
from agentic_research.llm.pricing import get_price


class LLMError(Exception):
    """Base class for model-layer failures."""


class ModelUnavailableError(LLMError):
    """A configured model cannot be reached or does not exist.

    Carries a remediation hint because the usual cause is a local model that
    was never pulled, which is fixable in one command.
    """

    def __init__(self, spec: ModelSpec, reason: str, hint: str = "") -> None:
        self.spec = spec
        self.reason = reason
        self.hint = hint
        message = f"Model {spec} unavailable: {reason}"
        if hint:
            message = f"{message}\n  -> {hint}"
        super().__init__(message)


class ModelTimeoutError(LLMError):
    """A model did not respond within the configured timeout.

    Distinct from ModelUnavailableError: the server is up and answering, it is
    just slow. Common with local models, where several concurrent requests to
    one Ollama instance queue behind each other.
    """


class StructuredOutputError(LLMError):
    """A model could not be coaxed into producing schema-valid output."""

    def __init__(self, spec: ModelSpec, schema_name: str, attempts: int, last_error: str) -> None:
        self.spec = spec
        self.schema_name = schema_name
        self.attempts = attempts
        super().__init__(
            f"{spec} failed to produce valid {schema_name} after {attempts} attempt(s): "
            f"{last_error}"
        )


class BudgetExceededError(LLMError):
    """The run hit its hard ceiling on model calls."""


@dataclass(slots=True)
class LLMCallRecord:
    """One model invocation, successful or not."""

    role: ModelRole
    provider: Provider
    model: str
    schema: str
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 1
    ok: bool = True
    error: str | None = None
    fell_back: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float | None:
        """``None`` means pricing for this model is unknown, not that it is free."""
        price = get_price(self.provider, self.model)
        if price is None:
            return None
        return price.cost(self.input_tokens, self.output_tokens)


@dataclass
class UsageTotals:
    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    known_cost_usd: float = 0.0
    unpriced_calls: int = 0
    by_provider: dict[str, int] = field(default_factory=dict)
    by_role: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_is_complete(self) -> bool:
        """False when at least one call used a model with no known price."""
        return self.unpriced_calls == 0


class UsageTracker:
    """Accumulates call records and enforces the call ceiling.

    Parallel workers call this concurrently. ``reserve`` is the only
    read-modify-write path, so it is the only one that needs the lock;
    ``list.append`` is already atomic.
    """

    def __init__(self, max_calls: int) -> None:
        self._max_calls = max_calls
        self._reserved = 0
        self._lock = asyncio.Lock()
        self.records: list[LLMCallRecord] = []

    async def reserve(self) -> None:
        """Claim one call slot, or refuse."""
        async with self._lock:
            if self._reserved >= self._max_calls:
                raise BudgetExceededError(
                    f"LLM call budget exhausted ({self._max_calls} calls). "
                    "Raise MAX_LLM_CALLS or narrow the question."
                )
            self._reserved += 1

    async def remaining(self) -> int:
        async with self._lock:
            return max(0, self._max_calls - self._reserved)

    def record(self, record: LLMCallRecord) -> None:
        self.records.append(record)

    def totals(self) -> UsageTotals:
        totals = UsageTotals()
        by_provider: dict[str, int] = defaultdict(int)
        by_role: dict[str, int] = defaultdict(int)
        for rec in self.records:
            totals.calls += 1
            if not rec.ok:
                totals.failed_calls += 1
            totals.input_tokens += rec.input_tokens
            totals.output_tokens += rec.output_tokens
            totals.latency_s += rec.latency_s
            cost = rec.cost_usd
            if cost is None:
                totals.unpriced_calls += 1
            else:
                totals.known_cost_usd += cost
            by_provider[rec.provider.value] += 1
            by_role[rec.role.value] += 1
        totals.by_provider = dict(by_provider)
        totals.by_role = dict(by_role)
        totals.known_cost_usd = round(totals.known_cost_usd, 6)
        totals.latency_s = round(totals.latency_s, 3)
        return totals
