"""Model routing, structured-output repair, usage accounting and fallback."""

from __future__ import annotations

import httpx
import pytest
import respx
from pydantic import BaseModel

from agentic_research.config import ModelRole, ModelSpec, Provider, Settings
from agentic_research.llm.base import (
    AttemptKind,
    BudgetExceededError,
    LLMCallRecord,
    ModelTimeoutError,
    ModelUnavailableError,
    ProviderAttempt,
    ProviderRateLimited,
    ProviderRejectedRequest,
    StructuredOutputError,
    UsageTracker,
)
from agentic_research.llm.pricing import get_price
from agentic_research.llm.router import ModelRouter, RoleModel

TAGS_URL = "http://localhost:11434/api/tags"


class Shape(BaseModel):
    value: str


class FakeChatModel:
    """Minimal stand-in for a LangChain chat model.

    Returns the {raw, parsed, parsing_error} envelope that include_raw=True
    produces, which is what the router is written against.
    """

    def __init__(self, envelopes: list[dict]) -> None:
        self._envelopes = envelopes
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        index = min(self.calls - 1, len(self._envelopes) - 1)
        envelope = self._envelopes[index]
        if isinstance(envelope, Exception):
            raise envelope
        return envelope


def envelope(parsed=None, error=None, in_tokens=10, out_tokens=5):
    class Raw:
        usage_metadata = {"input_tokens": in_tokens, "output_tokens": out_tokens}

    return {"raw": Raw(), "parsed": parsed, "parsing_error": error}


def role_model(model, tracker=None):
    return RoleModel(
        role=ModelRole.PLANNER,
        spec=ModelSpec(provider=Provider.OLLAMA, model="test:1b"),
        model=model,  # type: ignore[arg-type]
        tracker=tracker or UsageTracker(max_calls=10),
    )


class TestStructuredOutput:
    async def test_returns_parsed_object_and_records_usage(self) -> None:
        tracker = UsageTracker(max_calls=5)
        model = role_model(FakeChatModel([envelope(parsed=Shape(value="ok"))]), tracker)
        result = await model.structured(Shape, "sys", "user")

        assert result.value == "ok"
        totals = tracker.totals()
        assert totals.calls == 1
        assert totals.input_tokens == 10
        assert totals.output_tokens == 5

    async def test_repairs_a_schema_violation_on_the_second_attempt(self) -> None:
        """include_raw=True turns a parse failure into data, so the validation
        error can be fed back instead of the attempt being lost."""
        chat = FakeChatModel(
            [
                envelope(parsed=None, error="missing field 'value'"),
                envelope(parsed=Shape(value="recovered")),
            ]
        )
        tracker = UsageTracker(max_calls=5)
        result = await role_model(chat, tracker).structured(Shape, "s", "u")

        assert result.value == "recovered"
        assert chat.calls == 2
        record = tracker.records[0]
        assert record.attempts == 2
        # Tokens from the failed attempt are still billed and still counted.
        assert record.input_tokens == 20

    async def test_gives_up_after_the_repair_attempt(self) -> None:
        chat = FakeChatModel([envelope(parsed=None, error="still broken")])
        tracker = UsageTracker(max_calls=5)
        with pytest.raises(StructuredOutputError, match="Shape"):
            await role_model(chat, tracker).structured(Shape, "s", "u")

        assert chat.calls == 2
        assert tracker.totals().failed_calls == 1

    async def test_failed_calls_are_still_accounted_for(self) -> None:
        tracker = UsageTracker(max_calls=5)
        chat = FakeChatModel([RuntimeError("provider exploded")])
        with pytest.raises(RuntimeError):
            await role_model(chat, tracker).structured(Shape, "s", "u")
        assert tracker.totals().calls == 1
        assert tracker.totals().failed_calls == 1


class TestErrorMapping:
    """Anything a caller might catch must arrive as an LLMError. A raw
    httpx.ReadTimeout escaping here once killed a whole parallel super-step."""

    async def test_read_timeout_becomes_a_domain_error(self) -> None:
        chat = FakeChatModel([httpx.ReadTimeout("too slow")])
        with pytest.raises(ModelTimeoutError):
            await role_model(chat).structured(Shape, "s", "u")

    async def test_connect_error_becomes_model_unavailable_with_a_hint(self) -> None:
        chat = FakeChatModel([httpx.ConnectError("refused")])
        with pytest.raises(ModelUnavailableError, match="ollama serve"):
            await role_model(chat).structured(Shape, "s", "u")


class TestBudget:
    async def test_call_ceiling_is_enforced(self) -> None:
        tracker = UsageTracker(max_calls=2)
        await tracker.reserve()
        await tracker.reserve()
        with pytest.raises(BudgetExceededError, match="MAX_LLM_CALLS"):
            await tracker.reserve()

    async def test_reservation_happens_before_the_call(self) -> None:
        tracker = UsageTracker(max_calls=0)
        chat = FakeChatModel([envelope(parsed=Shape(value="x"))])
        with pytest.raises(BudgetExceededError):
            await role_model(chat, tracker).structured(Shape, "s", "u")
        assert chat.calls == 0, "budget must be checked before spending"


class TestUsageTotals:
    def test_unknown_pricing_is_reported_not_guessed(self) -> None:
        """Cost comes from provider attempts, which is where tokens are
        actually spent; a logical call is just their container."""
        tracker = UsageTracker(max_calls=10)
        tracker.record_attempt(
            ProviderAttempt(
                ModelRole.PLANNER,
                Provider.OPENAI,
                "gpt-6-sol",
                "S",
                AttemptKind.INITIAL,
                1.0,
                1000,
                500,
            )
        )
        tracker.record_attempt(
            ProviderAttempt(
                ModelRole.CRITIC,
                Provider.OPENAI,
                "gpt-unknown-99",
                "S",
                AttemptKind.INITIAL,
                1.0,
                100,
                100,
            )
        )
        totals = tracker.totals()
        assert totals.unpriced_calls == 1
        assert not totals.cost_is_complete
        assert totals.known_cost_usd > 0

    def test_local_calls_cost_nothing_but_still_count(self) -> None:
        tracker = UsageTracker(max_calls=10)
        tracker.record(
            LLMCallRecord(ModelRole.RESEARCHER, Provider.OLLAMA, "qwen3:4b", "S", 9.0, 5000, 900)
        )
        tracker.record_attempt(
            ProviderAttempt(
                ModelRole.RESEARCHER,
                Provider.OLLAMA,
                "qwen3:4b",
                "S",
                AttemptKind.INITIAL,
                9.0,
                5000,
                900,
            )
        )
        totals = tracker.totals()
        assert totals.known_cost_usd == 0.0
        assert totals.cost_is_complete
        assert totals.total_tokens == 5900
        assert totals.by_provider == {"ollama": 1}
        assert totals.provider_requests == 1

    def test_pricing_prefix_match_handles_dated_snapshots(self) -> None:
        assert get_price(Provider.OPENAI, "gpt-6-sol-2026-09-01") == get_price(
            Provider.OPENAI, "gpt-6-sol"
        )


class TestPreflightAndFallback:
    @respx.mock
    async def test_missing_local_model_errors_with_the_pull_command(self) -> None:
        respx.get(TAGS_URL).mock(
            return_value=httpx.Response(200, json={"models": [{"name": "other:1b"}]})
        )
        settings = Settings(llm_mode="local", ollama_model="qwen3:4b", _env_file=None)
        with pytest.raises(ModelUnavailableError, match="ollama pull qwen3:4b"):
            await ModelRouter(settings).preflight()

    @respx.mock
    async def test_unreachable_server_errors_with_the_serve_command(self) -> None:
        respx.get(TAGS_URL).mock(side_effect=httpx.ConnectError("down"))
        settings = Settings(llm_mode="local", _env_file=None)
        with pytest.raises(ModelUnavailableError, match="ollama serve"):
            await ModelRouter(settings).preflight()

    @respx.mock
    async def test_implicit_latest_tag_is_accepted(self) -> None:
        respx.get(TAGS_URL).mock(
            return_value=httpx.Response(200, json={"models": [{"name": "qwen3:latest"}]})
        )
        settings = Settings(llm_mode="local", ollama_model="qwen3", _env_file=None)
        assert await ModelRouter(settings).preflight() == []

    @respx.mock
    async def test_fallback_is_off_by_default(self) -> None:
        """An unreachable local server must not quietly start spending money."""
        respx.get(TAGS_URL).mock(side_effect=httpx.ConnectError("down"))
        settings = Settings(
            llm_mode="hybrid",
            openai_api_key="sk-x",
            allow_cloud_fallback=False,
            _env_file=None,
        )
        with pytest.raises(ModelUnavailableError):
            await ModelRouter(settings).preflight()

    @respx.mock
    async def test_fallback_when_enabled_swaps_provider_and_warns(self) -> None:
        respx.get(TAGS_URL).mock(side_effect=httpx.ConnectError("down"))
        settings = Settings(
            llm_mode="hybrid",
            openai_api_key="sk-x",
            allow_cloud_fallback=True,
            _env_file=None,
        )
        router = ModelRouter(settings)
        warnings = await router.preflight()

        assert len(warnings) == 1 and "falling back" in warnings[0]
        assert router.assignments()[ModelRole.RESEARCHER].provider is Provider.OPENAI

    @respx.mock
    async def test_cloud_only_mode_skips_the_ollama_check(self) -> None:
        route = respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json={"models": []}))
        settings = Settings(llm_mode="cloud", openai_api_key="sk-x", _env_file=None)
        assert await ModelRouter(settings).preflight() == []
        assert route.call_count == 0


class TestProviderQuirks:
    """Provider parameter quirks belong in the router, not in every caller."""

    async def test_model_rejecting_temperature_is_retried_without_it(self) -> None:
        """Measured against gpt-6-luna, which accepts only its default
        temperature and 400s on any explicit value. Which models behave this
        way is not discoverable without asking, so the router absorbs it."""
        rejection = RuntimeError(
            "Error code: 400 - {'error': {'message': \"Unsupported value: "
            "'temperature' does not support 0.2 with this model. Only the "
            "default (1) value is supported.\", 'type': 'invalid_request_error'}}"
        )
        rebuilt = FakeChatModel([envelope(parsed=Shape(value="ok"))])
        first = FakeChatModel([rejection])
        calls = {"rebuilds": 0}

        def rebuild() -> FakeChatModel:
            calls["rebuilds"] += 1
            return rebuilt

        model = RoleModel(
            role=ModelRole.PLANNER,
            spec=ModelSpec(provider=Provider.OPENAI, model="gpt-6-luna"),
            model=first,  # type: ignore[arg-type]
            tracker=UsageTracker(max_calls=10),
            rebuild_without_temperature=rebuild,  # type: ignore[arg-type]
        )
        result = await model.structured(Shape, "s", "u")

        assert result.value == "ok"
        assert calls["rebuilds"] == 1
        assert rebuilt.calls == 1

    async def test_other_bad_requests_are_not_silently_retried(self) -> None:
        """Only the temperature quirk is absorbed. A genuinely malformed
        request must surface rather than be retried into the same failure."""
        rejection = RuntimeError(
            "Error code: 400 - {'error': {'message': 'Invalid schema', "
            "'type': 'invalid_request_error'}}"
        )
        rebuilds = {"n": 0}

        def rebuild() -> FakeChatModel:
            rebuilds["n"] += 1
            return FakeChatModel([envelope(parsed=Shape(value="x"))])

        model = RoleModel(
            role=ModelRole.PLANNER,
            spec=ModelSpec(provider=Provider.OPENAI, model="gpt-6-luna"),
            model=FakeChatModel([rejection]),  # type: ignore[arg-type]
            tracker=UsageTracker(max_calls=10),
            rebuild_without_temperature=rebuild,  # type: ignore[arg-type]
        )
        with pytest.raises(ProviderRejectedRequest):
            await model.structured(Shape, "s", "u")
        assert rebuilds["n"] == 0

    def test_local_models_never_get_the_rebuild_hook(self) -> None:
        settings = Settings(llm_mode="local", ollama_model="qwen3:4b", _env_file=None)
        role_model = ModelRouter(settings).get(ModelRole.PLANNER)
        assert role_model._rebuild_without_temperature is None

    def test_a_model_known_to_reject_temperature_is_built_without_it(self) -> None:
        """Remembered per model, so the retry happens once rather than on
        every call for the rest of the run."""
        settings = Settings(
            llm_mode="cloud",
            openai_api_key="sk-test",
            openai_model="gpt-6-luna",
            _env_file=None,
        )
        router = ModelRouter(settings)
        spec = ModelSpec(provider=Provider.OPENAI, model="gpt-6-luna")
        rebuilt = router._without_temperature(spec, 2_000)
        assert "gpt-6-luna" in router._no_temperature
        assert getattr(rebuilt, "temperature", None) is None


class TestRateLimitHandling:
    """A provider quota refusal must degrade the run, not crash it.

    Found live: a real 429 from gpt-6-luna ("requests per day (RPD): Limit
    50, Used 50") was not an LLMError, so every planning, critique and
    reporting node -- all of which catch LLMError -- let it escape and
    killed the run. Only the Send-dispatched workers survived, because they
    catch broadly.
    """

    REAL_429 = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached for "
        "gpt-6-luna in organization org-44eb on requests per day (RPD): "
        "Limit 50, Used 50, Requested 1.', 'type': 'requests'}}"
    )

    async def test_a_provider_429_becomes_an_llm_error(self) -> None:
        chat = FakeChatModel([RuntimeError(self.REAL_429)])
        with pytest.raises(ProviderRateLimited) as info:
            await role_model(chat).structured(Shape, "s", "u")
        # The provider's own wording names the limit that was hit.
        assert "50" in str(info.value)

    async def test_rate_limits_are_caught_by_llm_error_handlers(self) -> None:
        """The property that actually matters: the nodes catch LLMError, so
        a rate limit must be one."""
        from agentic_research.llm.base import LLMError

        chat = FakeChatModel([RuntimeError(self.REAL_429)])
        with pytest.raises(LLMError):
            await role_model(chat).structured(Shape, "s", "u")

    @pytest.mark.parametrize(
        "message",
        [
            "Error code: 429 - too many requests",
            "You exceeded your current quota",
            "insufficient_quota: please check your plan",
            "Rate limit reached for requests",
        ],
    )
    async def test_quota_refusals_are_recognised(self, message: str) -> None:
        chat = FakeChatModel([RuntimeError(message)])
        with pytest.raises(ProviderRateLimited):
            await role_model(chat).structured(Shape, "s", "u")

    async def test_a_rate_limit_is_not_confused_with_unreachability(self) -> None:
        """A 429 body can mention 'connection'; reporting it as an
        unreachable model would send the user to fix the wrong thing."""
        chat = FakeChatModel(
            [RuntimeError("Error code: 429 - rate limit; connection pool saturated")]
        )
        with pytest.raises(ProviderRateLimited):
            await role_model(chat).structured(Shape, "s", "u")

    async def test_a_genuine_connection_failure_still_reports_as_unavailable(
        self,
    ) -> None:
        chat = FakeChatModel([httpx.ConnectError("refused")])
        with pytest.raises(ModelUnavailableError):
            await role_model(chat).structured(Shape, "s", "u")
