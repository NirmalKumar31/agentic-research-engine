"""Provider-request accounting.

Budgets are enforced per provider HTTP request, not per logical model call.
The distinction is not pedantic: the live validation counted 50 provider
requests against 42 billable ones, because a structured-output repair, a
temperature-compatibility retry and an SDK-internal retry are each separate
requests that a per-logical-call reservation could not see.

These tests pin the guarantee the README is allowed to state.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from agentic_research.config import ModelRole, ModelSpec, Provider, Settings
from agentic_research.llm.base import (
    BudgetExceededError,
    CloudBudgetExceededError,
    ProviderRateLimited,
    ProviderRejectedRequest,
    StructuredOutputError,
    UsageTracker,
)
from agentic_research.llm.router import ModelRouter, RoleModel


class Shape(BaseModel):
    value: str


class ScriptedModel:
    """Returns a scripted sequence of include_raw envelopes."""

    def __init__(self, envelopes: list) -> None:
        self._envelopes = envelopes
        self.requests = 0

    def with_structured_output(self, schema, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.requests += 1
        item = self._envelopes[min(self.requests - 1, len(self._envelopes) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def envelope(parsed=None, error=None, tokens_in=100, tokens_out=50):
    class Raw:
        usage_metadata = {"input_tokens": tokens_in, "output_tokens": tokens_out}

    return {"raw": Raw(), "parsed": parsed, "parsing_error": error}


def cloud_budget(**overrides: object):
    return Settings(llm_mode="local", _env_file=None, **overrides).cloud_budget  # type: ignore[arg-type]


def role_model(model, tracker, *, rebuild=None, spec=None):
    return RoleModel(
        role=ModelRole.SYNTHESIZER,
        spec=spec or ModelSpec(provider=Provider.OPENAI, model="gpt-6-luna"),
        model=model,
        tracker=tracker,
        rebuild_without_temperature=rebuild,
    )


class TestAttemptsAreCountedIndividually:
    async def test_one_successful_call_is_one_provider_request(self) -> None:
        tracker = UsageTracker(10, cloud_budget())
        chat = ScriptedModel([envelope(parsed=Shape(value="ok"))])
        await role_model(chat, tracker).structured(Shape, "s", "u")

        totals = tracker.totals()
        assert chat.requests == 1
        assert totals.calls == 1
        assert totals.provider_requests == 1
        assert totals.structured_repairs == 0
        assert totals.requests_per_logical_call == 1.0

    async def test_a_structured_repair_is_a_second_provider_request(self) -> None:
        """One logical call, two HTTP requests, both billed."""
        tracker = UsageTracker(10, cloud_budget())
        chat = ScriptedModel(
            [
                envelope(parsed=None, error="missing field"),
                envelope(parsed=Shape(value="recovered")),
            ]
        )
        result = await role_model(chat, tracker).structured(Shape, "s", "u")

        totals = tracker.totals()
        assert result.value == "recovered"
        assert chat.requests == 2
        assert totals.calls == 1, "still one logical call"
        assert totals.provider_requests == 2, "but two provider requests"
        assert totals.structured_repairs == 1
        assert totals.billable_provider_requests == 2
        # Tokens from both attempts are counted, not just the successful one.
        assert totals.input_tokens == 200

    async def test_a_compatibility_retry_is_a_second_provider_request(self) -> None:
        """The rejected 400 bills nothing but still consumes rate limit."""
        rejection = RuntimeError(
            "Error code: 400 - Unsupported value: 'temperature' does not support 0.2"
        )
        tracker = UsageTracker(10, cloud_budget())
        rebuilt = ScriptedModel([envelope(parsed=Shape(value="ok"))])
        first = ScriptedModel([rejection])
        model = role_model(first, tracker, rebuild=lambda: rebuilt)

        await model.structured(Shape, "s", "u")

        totals = tracker.totals()
        assert totals.provider_requests == 2
        assert totals.compatibility_retries == 1
        # The 400 is counted as a request but not as a billable one.
        assert totals.billable_provider_requests == 1

    async def test_requests_are_attributed_per_model(self) -> None:
        tracker = UsageTracker(10, cloud_budget())
        chat = ScriptedModel([envelope(parsed=Shape(value="ok"))])
        await role_model(chat, tracker).structured(Shape, "s", "u")
        assert tracker.totals().provider_requests_by_model == {"openai:gpt-6-luna": 1}


class TestCeilingsCannotBeBypassed:
    async def test_a_repair_cannot_bypass_the_request_ceiling(self) -> None:
        """The repair must be refused, not waved through because the
        logical call was already reserved."""
        tracker = UsageTracker(10, cloud_budget(), max_provider_requests=1)
        chat = ScriptedModel(
            [envelope(parsed=None, error="bad"), envelope(parsed=Shape(value="x"))]
        )
        with pytest.raises(BudgetExceededError, match="MAX_PROVIDER_REQUESTS"):
            await role_model(chat, tracker).structured(Shape, "s", "u")
        assert chat.requests == 1, "the second request must never be emitted"

    async def test_a_repair_cannot_bypass_the_cloud_request_ceiling(self) -> None:
        tracker = UsageTracker(10, cloud_budget(max_cloud_calls=1))
        chat = ScriptedModel(
            [envelope(parsed=None, error="bad"), envelope(parsed=Shape(value="x"))]
        )
        with pytest.raises(CloudBudgetExceededError):
            await role_model(chat, tracker).structured(Shape, "s", "u")
        assert chat.requests == 1

    async def test_a_repair_cannot_bypass_the_cost_ceiling(self) -> None:
        """Each attempt reserves its own worst case, so two attempts need
        twice the headroom. Reserving once per logical call did not."""
        one_attempt_worst_case = 0.0031  # 6k output tokens on luna, plus input
        tracker = UsageTracker(10, cloud_budget(max_cloud_cost_usd=one_attempt_worst_case))
        chat = ScriptedModel(
            [envelope(parsed=None, error="bad"), envelope(parsed=Shape(value="x"))]
        )
        with pytest.raises(CloudBudgetExceededError) as info:
            await role_model(chat, tracker).structured(Shape, "s", "u")
        assert info.value.dimension == "cost_usd"
        assert chat.requests == 1

    async def test_a_compatibility_retry_cannot_bypass_the_ceiling(self) -> None:
        rejection = RuntimeError(
            "Error code: 400 - Unsupported value: 'temperature' does not support 0.2"
        )
        tracker = UsageTracker(10, cloud_budget(), max_provider_requests=1)
        rebuilt = ScriptedModel([envelope(parsed=Shape(value="ok"))])
        model = role_model(ScriptedModel([rejection]), tracker, rebuild=lambda: rebuilt)

        with pytest.raises(BudgetExceededError):
            await model.structured(Shape, "s", "u")
        assert rebuilt.requests == 0


class TestConcurrency:
    async def test_parallel_workers_cannot_race_past_the_request_ceiling(self) -> None:
        """Reservations check reserved totals, not recorded ones, so
        concurrent workers cannot each read a stale 'already spent'."""
        tracker = UsageTracker(100, cloud_budget(), max_provider_requests=5)
        granted = 0
        refused = 0

        async def attempt() -> None:
            nonlocal granted, refused
            try:
                await tracker.reserve_provider_request(
                    Provider.OPENAI, role=ModelRole.VERIFIER, model="gpt-6-luna"
                )
                granted += 1
            except BudgetExceededError:
                refused += 1

        await asyncio.gather(*(attempt() for _ in range(40)))
        assert granted == 5, f"ceiling breached: {granted} granted"
        assert refused == 35

    async def test_parallel_workers_cannot_race_past_the_cost_ceiling(self) -> None:
        # Room for exactly three worst-case verifier attempts.
        per_attempt = 0.0002
        tracker = UsageTracker(100, cloud_budget(max_cloud_cost_usd=per_attempt * 3))
        granted = 0

        async def attempt() -> None:
            nonlocal granted
            try:
                await tracker.reserve_provider_request(
                    Provider.OPENAI, role=ModelRole.VERIFIER, model="gpt-6-luna"
                )
                granted += 1
            except CloudBudgetExceededError:
                pass

        await asyncio.gather(*(attempt() for _ in range(30)))
        assert granted == 3, f"cost ceiling breached: {granted} granted"


class TestReconciliation:
    async def test_actual_spend_never_exceeds_what_was_reserved(self) -> None:
        """The guarantee itself. Reserved worst case is an upper bound on
        recorded actuals, checked rather than assumed."""
        tracker = UsageTracker(20, cloud_budget())
        for _ in range(5):
            chat = ScriptedModel(
                [
                    envelope(parsed=None, error="bad", tokens_in=500, tokens_out=400),
                    envelope(parsed=Shape(value="x"), tokens_in=700, tokens_out=300),
                ]
            )
            await role_model(chat, tracker).structured(Shape, "s", "u")

        totals = tracker.totals()
        assert totals.provider_requests == 10
        assert totals.structured_repairs == 5
        assert totals.known_cost_usd > 0
        assert totals.reserved_worst_case_usd >= totals.known_cost_usd
        assert totals.reservation_was_sufficient

    async def test_repair_prompts_are_re_estimated_not_reused(self) -> None:
        """A repair carries the failed response and correction back into the
        prompt, so its input estimate must grow rather than reuse the first."""
        tracker = UsageTracker(10, cloud_budget())
        chat = ScriptedModel(
            [envelope(parsed=None, error="bad"), envelope(parsed=Shape(value="x"))]
        )
        await role_model(chat, tracker).structured(Shape, "s", "u" * 4_000)
        # Two reservations, the second for a strictly larger prompt.
        assert tracker.totals().provider_requests == 2
        assert await tracker.reserved_worst_case_usd() > 0


class TestFailuresStayObservable:
    """A refused request is still a request. Losing it from the counters
    is how a quota disappears with no record of where it went."""

    REAL_429 = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached for "
        "gpt-6-luna on requests per day (RPD): Limit 50, Used 50.'}}"
    )

    async def test_a_rate_limit_refusal_is_counted_as_an_attempt(self) -> None:
        tracker = UsageTracker(10, cloud_budget())
        chat = ScriptedModel([RuntimeError(self.REAL_429)])
        # Narrow on purpose: a bare Exception would also pass if the router
        # stopped recognising a 429, which is the thing being tested.
        with pytest.raises(ProviderRateLimited):
            await role_model(chat, tracker).structured(Shape, "s", "u")

        totals = tracker.totals()
        assert totals.provider_requests == 1, "the refused request still happened"
        assert totals.rate_limit_refusals == 1
        assert totals.failed_provider_requests == 1
        # A 429 does no work, so it bills nothing but consumes quota.
        assert totals.billable_provider_requests == 0

    async def test_a_400_is_a_failed_attempt_but_not_a_rate_limit(self) -> None:
        tracker = UsageTracker(10, cloud_budget())
        chat = ScriptedModel(
            [
                RuntimeError(
                    "Error code: 400 - {'error': {'type': 'invalid_request_error', "
                    "'message': 'Invalid schema'}}"
                )
            ]
        )
        with pytest.raises(ProviderRejectedRequest):
            await role_model(chat, tracker).structured(Shape, "s", "u")

        totals = tracker.totals()
        assert totals.provider_requests == 1
        assert totals.failed_provider_requests == 1
        assert totals.rate_limit_refusals == 0

    async def test_a_successful_request_is_not_counted_as_failed(self) -> None:
        tracker = UsageTracker(10, cloud_budget())
        chat = ScriptedModel([envelope(parsed=Shape(value="ok"))])
        await role_model(chat, tracker).structured(Shape, "s", "u")

        totals = tracker.totals()
        assert totals.failed_provider_requests == 0
        assert totals.rate_limit_refusals == 0
        assert totals.billable_provider_requests == 1

    async def test_an_unparsable_response_is_a_failed_attempt(self) -> None:
        """It reached the model and billed tokens, so it is billable, but
        the attempt did not succeed."""
        tracker = UsageTracker(10, cloud_budget())
        chat = ScriptedModel([envelope(parsed=None, error="bad")])
        with pytest.raises(StructuredOutputError):
            await role_model(chat, tracker).structured(Shape, "s", "u", repair=False)

        totals = tracker.totals()
        assert totals.provider_requests == 1
        assert totals.failed_provider_requests == 1
        assert totals.billable_provider_requests == 1


class TestNoInvisibleRequests:
    def test_openai_clients_disable_sdk_retries(self) -> None:
        """An SDK-internal retry is a provider request our accounting never
        sees, which is the gap this design closes."""
        settings = Settings(
            llm_mode="cloud",
            openai_api_key="sk-test",
            openai_model="gpt-6-luna",
            llm_max_retries=5,
            _env_file=None,
        )
        client = ModelRouter(settings).get(ModelRole.SYNTHESIZER)._model
        assert client.max_retries == 0

    def test_ollama_keeps_its_configured_retries(self) -> None:
        """Local retries cost nothing and hit no provider quota."""
        settings = Settings(llm_mode="local", llm_max_retries=3, _env_file=None)
        ModelRouter(settings).get(ModelRole.RESEARCHER)
