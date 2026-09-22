"""Model routing, structured-output repair, usage accounting and fallback."""

from __future__ import annotations

import httpx
import pytest
import respx
from pydantic import BaseModel

from agentic_research.config import ModelRole, ModelSpec, Provider, Settings
from agentic_research.llm.base import (
    BudgetExceededError,
    LLMCallRecord,
    ModelTimeoutError,
    ModelUnavailableError,
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
        tracker = UsageTracker(max_calls=10)
        tracker.record(
            LLMCallRecord(ModelRole.PLANNER, Provider.OPENAI, "gpt-6-sol", "S", 1.0, 1000, 500)
        )
        tracker.record(
            LLMCallRecord(ModelRole.CRITIC, Provider.OPENAI, "gpt-unknown-99", "S", 1.0, 100, 100)
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
        totals = tracker.totals()
        assert totals.known_cost_usd == 0.0
        assert totals.cost_is_complete
        assert totals.total_tokens == 5900
        assert totals.by_provider == {"ollama": 1}

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
