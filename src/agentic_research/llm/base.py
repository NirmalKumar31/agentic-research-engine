"""Call records, usage accounting and LLM-layer errors.

Everything the engine knows about what a run cost is produced here. Because
every model call goes through one wrapper, no call site can forget to report
its tokens, and adding a new node cannot silently create an unmetered cost.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

from agentic_research.config import CloudBudget, ModelRole, ModelSpec, Provider
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


class ProviderRateLimited(LLMError):
    """The provider refused the request for quota or rate reasons.

    Its own type because the remedy is unlike the others: not a smaller
    request, not a different model, just less traffic or a later time. The
    message carries the provider's own wording, which usually names the
    exact limit that was hit.
    """

    def __init__(self, detail: str, retry_after_seconds: float | None = None) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(detail)


class ProviderRejectedRequest(LLMError):
    """The provider refused the request as malformed or unsupported.

    Distinct from a transport failure: retrying the identical request will
    fail identically, so only a changed request is worth attempting.
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
    """The run hit a hard ceiling on model calls or provider requests."""


class CloudBudgetExceededError(BudgetExceededError):
    """A paid-provider ceiling would be exceeded by the next call.

    Raised *before* dispatch, using the worst case that call could cost, so a
    configured spend limit is never silently passed. Distinct from the plain
    call ceiling because the remedy differs: one is a throughput limit, the
    other is money.
    """

    def __init__(self, dimension: str, spent: float, ceiling: float) -> None:
        self.dimension = dimension
        self.spent = spent
        self.ceiling = ceiling
        super().__init__(
            f"cloud {dimension} budget would be exceeded: {spent:g} of {ceiling:g} "
            f"already committed. Raise MAX_CLOUD_{dimension.upper()} or use "
            "LLM_MODE=local."
        )


class AttemptKind(StrEnum):
    """Why a particular provider request was issued.

    Recorded per request because the distinction is the whole point: a
    repair and a compatibility retry are extra provider requests that the
    old per-logical-call accounting could not see.
    """

    INITIAL = "initial"
    STRUCTURED_REPAIR = "structured_repair"
    COMPATIBILITY_RETRY = "compatibility_retry"
    TRANSPORT_RETRY = "transport_retry"


@dataclass(slots=True)
class ProviderAttempt:
    """One HTTP request to a provider.

    This is the unit budgets are enforced in. A logical model call may
    produce several of these, and the provider counts every one against its
    rate limit whether or not it bills tokens -- the live validation counted
    50 provider requests against 42 billable ones.
    """

    role: ModelRole
    provider: Provider
    model: str
    schema: str
    kind: AttemptKind
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    ok: bool = True
    error: str | None = None
    billable: bool = True
    """False for requests the provider rejected before doing work (4xx),
    which still consume rate-limit quota but produce no tokens."""

    @property
    def cost_usd(self) -> float | None:
        price = get_price(self.provider, self.model)
        if price is None:
            return None
        return price.cost(self.input_tokens, self.output_tokens)


@dataclass(slots=True)
class LLMCallRecord:
    """One *logical* model call, aggregating its provider attempts."""

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
    provider_requests: int = 0
    billable_provider_requests: int = 0
    structured_repairs: int = 0
    compatibility_retries: int = 0
    transport_retries: int = 0
    rate_limit_refusals: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    known_cost_usd: float = 0.0
    unpriced_calls: int = 0
    by_provider: dict[str, int] = field(default_factory=dict)
    by_role: dict[str, int] = field(default_factory=dict)
    provider_requests_by_model: dict[str, int] = field(default_factory=dict)
    reserved_worst_case_usd: float = 0.0
    """Sum of the worst case each attempt was allowed to cost. Actual spend
    must never exceed it; the gap is how conservative the ceiling was."""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_is_complete(self) -> bool:
        """False when at least one call used a model with no known price."""
        return self.unpriced_calls == 0

    @property
    def requests_per_logical_call(self) -> float:
        if self.calls == 0:
            return 0.0
        return round(self.provider_requests / self.calls, 3)

    @property
    def reservation_was_sufficient(self) -> bool:
        """Actual spend stayed within what was reserved before dispatch.

        The guarantee the README is allowed to state rests on this being
        true; it is asserted in tests rather than assumed.
        """
        return self.known_cost_usd <= self.reserved_worst_case_usd + 1e-9


class UsageTracker:
    """Accumulates usage and enforces ceilings at the provider-request level.

    The unit matters. An earlier version reserved once per *logical* model
    call, but one logical call can emit several HTTP requests: a structured
    -output repair, a temperature-compatibility retry, a transport retry.
    Providers count every one of those against rate limits, and each
    billable one consumes tokens. Reserving per logical call therefore
    under-counted, which is exactly what the live run showed -- the provider
    counted 50 requests where our accounting saw 42.

    Every request now passes ``reserve_provider_request`` immediately before
    it is emitted, and reserves its own worst-case token and cost allowance.

    Parallel workers call this concurrently. The reservation paths are the
    only read-modify-write sections, so they hold the lock; ``list.append``
    is already atomic.
    """

    def __init__(
        self,
        max_calls: int,
        cloud_budget: CloudBudget | None = None,
        max_provider_requests: int = 0,
    ) -> None:
        self._max_calls = max_calls
        self._max_provider_requests = max_provider_requests
        self._cloud = cloud_budget
        self._reserved_calls = 0
        self._reserved_requests = 0
        self._reserved_cloud_requests = 0
        self._reserved_input_tokens = 0
        self._reserved_output_tokens = 0
        self._reserved_cost_usd = 0.0
        self._lock = asyncio.Lock()
        self.records: list[LLMCallRecord] = []
        self.attempts: list[ProviderAttempt] = []

    # -- reservations ------------------------------------------------------

    async def reserve(self) -> None:
        """Claim one *logical* call slot.

        Bounds how much work the graph does. It does not bound money; that
        is :meth:`reserve_provider_request`.
        """
        async with self._lock:
            if self._reserved_calls >= self._max_calls:
                raise BudgetExceededError(
                    f"LLM call budget exhausted ({self._max_calls} calls). "
                    "Raise MAX_LLM_CALLS or narrow the question."
                )
            self._reserved_calls += 1

    async def reserve_provider_request(
        self,
        provider: Provider,
        *,
        role: ModelRole,
        model: str,
        estimated_input_tokens: int = 0,
    ) -> float:
        """Claim one provider HTTP request, or refuse.

        Returns the worst-case cost reserved for it, so the caller can
        reconcile actuals against reservations afterwards.

        Called immediately before the request is emitted -- including for
        repairs and retries, which are separate requests and reserve
        separately. Refusing here is the only way a ceiling can be a
        guarantee rather than a hope.
        """
        async with self._lock:
            if (
                self._max_provider_requests
                and self._reserved_requests >= self._max_provider_requests
            ):
                raise BudgetExceededError(
                    f"provider request budget exhausted "
                    f"({self._max_provider_requests} requests). "
                    "Raise MAX_PROVIDER_REQUESTS."
                )
            worst_case = 0.0
            if provider is Provider.OPENAI and self._cloud is not None:
                worst_case = self._check_cloud_locked(role, model, estimated_input_tokens)
                self._reserved_cloud_requests += 1
                self._reserved_input_tokens += estimated_input_tokens
                self._reserved_output_tokens += self._output_cap_for(role)
                self._reserved_cost_usd += worst_case
            self._reserved_requests += 1
            return worst_case

    def _output_cap_for(self, role: ModelRole) -> int:
        if self._cloud is None:
            return 0
        return self._cloud.output_cap_for(role.value) or 2_000

    def _check_cloud_locked(
        self, role: ModelRole, model: str, estimated_input_tokens: int
    ) -> float:
        """Caller must hold the lock. Returns the worst-case cost reserved.

        Checks against *reserved* totals rather than recorded ones, so
        concurrent workers cannot each see a stale "already spent" figure
        and collectively overshoot.
        """
        budget = self._cloud
        assert budget is not None
        output_cap = self._output_cap_for(role)

        if budget.max_cloud_calls and self._reserved_cloud_requests >= budget.max_cloud_calls:
            raise CloudBudgetExceededError(
                "calls", self._reserved_cloud_requests, budget.max_cloud_calls
            )
        if budget.max_cloud_input_tokens and (
            self._reserved_input_tokens + estimated_input_tokens > budget.max_cloud_input_tokens
        ):
            raise CloudBudgetExceededError(
                "input_tokens", self._reserved_input_tokens, budget.max_cloud_input_tokens
            )
        if budget.max_cloud_output_tokens and (
            self._reserved_output_tokens + output_cap > budget.max_cloud_output_tokens
        ):
            raise CloudBudgetExceededError(
                "output_tokens", self._reserved_output_tokens, budget.max_cloud_output_tokens
            )

        worst_case = 0.0
        price = get_price(Provider.OPENAI, model)
        if price is not None:
            worst_case = price.cost(estimated_input_tokens, output_cap)
            if (
                budget.max_cloud_cost_usd
                and self._reserved_cost_usd + worst_case > budget.max_cloud_cost_usd
            ):
                raise CloudBudgetExceededError(
                    "cost_usd", self._reserved_cost_usd, budget.max_cloud_cost_usd
                )
        return worst_case

    async def remaining(self) -> int:
        async with self._lock:
            return max(0, self._max_calls - self._reserved_calls)

    async def reserved_worst_case_usd(self) -> float:
        async with self._lock:
            return self._reserved_cost_usd

    # -- recording ---------------------------------------------------------

    def record(self, record: LLMCallRecord) -> None:
        self.records.append(record)

    def record_attempt(self, attempt: ProviderAttempt) -> None:
        self.attempts.append(attempt)

    def totals(self) -> UsageTotals:
        totals = UsageTotals()
        by_provider: dict[str, int] = defaultdict(int)
        by_role: dict[str, int] = defaultdict(int)
        by_model: dict[str, int] = defaultdict(int)

        for rec in self.records:
            totals.calls += 1
            if not rec.ok:
                totals.failed_calls += 1
            by_provider[rec.provider.value] += 1
            by_role[rec.role.value] += 1

        for attempt in self.attempts:
            totals.provider_requests += 1
            by_model[f"{attempt.provider.value}:{attempt.model}"] += 1
            if attempt.billable:
                totals.billable_provider_requests += 1
            if attempt.kind is AttemptKind.STRUCTURED_REPAIR:
                totals.structured_repairs += 1
            elif attempt.kind is AttemptKind.COMPATIBILITY_RETRY:
                totals.compatibility_retries += 1
            elif attempt.kind is AttemptKind.TRANSPORT_RETRY:
                totals.transport_retries += 1
            totals.input_tokens += attempt.input_tokens
            totals.output_tokens += attempt.output_tokens
            totals.latency_s += attempt.latency_s
            cost = attempt.cost_usd
            if cost is None:
                totals.unpriced_calls += 1
            else:
                totals.known_cost_usd += cost

        totals.by_provider = dict(by_provider)
        totals.by_role = dict(by_role)
        totals.provider_requests_by_model = dict(by_model)
        totals.reserved_worst_case_usd = round(self._reserved_cost_usd, 6)
        totals.known_cost_usd = round(totals.known_cost_usd, 6)
        totals.latency_s = round(totals.latency_s, 3)
        return totals
