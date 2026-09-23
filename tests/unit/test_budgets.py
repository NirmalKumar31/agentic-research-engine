"""Cloud spend ceilings.

Round, query and source budgets bound how much work happens. These bound how
much *money* happens, which is a different failure: exceeding a round limit
wastes time, exceeding a spend limit costs something unrecoverable. All are
checked before dispatch.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from agentic_research.config import ModelRole, Provider, Settings
from agentic_research.llm.base import (
    BudgetExceededError,
    CloudBudgetExceededError,
    LLMCallRecord,
    UsageTracker,
)
from agentic_research.models import SearchQuery
from agentic_research.search import SearchService, TavilyProvider

TAVILY_URL = "https://api.tavily.com/search"


def budget(**overrides: object) -> object:
    return Settings(llm_mode="local", _env_file=None, **overrides).cloud_budget  # type: ignore[arg-type]


def cloud_call(tokens_in: int = 1000, tokens_out: int = 500) -> LLMCallRecord:
    return LLMCallRecord(
        ModelRole.SYNTHESIZER, Provider.OPENAI, "gpt-6-luna", "R", 1.0, tokens_in, tokens_out
    )


class TestCloudCallCeiling:
    async def test_cloud_calls_are_capped(self) -> None:
        tracker = UsageTracker(100, budget(max_cloud_calls=2))
        for _ in range(2):
            await tracker.reserve(Provider.OPENAI, role=ModelRole.VERIFIER, model="gpt-6-luna")
        with pytest.raises(CloudBudgetExceededError) as info:
            await tracker.reserve(Provider.OPENAI, role=ModelRole.VERIFIER, model="gpt-6-luna")
        assert info.value.dimension == "calls"

    async def test_local_calls_are_not_charged_to_the_cloud_budget(self) -> None:
        """Local inference has no vendor cost, so a spent cloud budget must
        not stop a local-mode run."""
        tracker = UsageTracker(100, budget(max_cloud_calls=1))
        await tracker.reserve(Provider.OPENAI, role=ModelRole.VERIFIER, model="gpt-6-luna")
        for _ in range(5):
            await tracker.reserve(Provider.OLLAMA, role=ModelRole.RESEARCHER, model="qwen3:4b")

    async def test_zero_means_unlimited_not_forbidden(self) -> None:
        """An explicitly zero budget would otherwise be indistinguishable
        from an unconfigured one."""
        tracker = UsageTracker(100, budget(max_cloud_calls=0, max_cloud_cost_usd=0.0))
        for _ in range(5):
            await tracker.reserve(Provider.OPENAI, role=ModelRole.VERIFIER, model="gpt-6-luna")


class TestSpendCeiling:
    async def test_cost_is_checked_before_dispatch_not_after(self) -> None:
        """The whole point: the call that would breach the ceiling never runs."""
        tracker = UsageTracker(100, budget(max_cloud_cost_usd=0.02))
        tracker.record(cloud_call(tokens_in=100_000, tokens_out=30_000))
        with pytest.raises(CloudBudgetExceededError) as info:
            await tracker.reserve(
                Provider.OPENAI,
                role=ModelRole.SYNTHESIZER,
                model="gpt-6-luna",
                estimated_input_tokens=50_000,
            )
        assert info.value.dimension == "cost_usd"

    async def test_worst_case_output_is_included_in_the_check(self) -> None:
        """Checking only committed spend would let one huge generation
        overshoot, since its cost is unknown until it has been billed."""
        # Nothing is committed yet, so a ceiling below this single call's
        # worst case (6k output tokens for the synthesiser) must still block.
        tracker = UsageTracker(100, budget(max_cloud_cost_usd=0.0029))
        with pytest.raises(CloudBudgetExceededError):
            await tracker.reserve(
                Provider.OPENAI,
                role=ModelRole.SYNTHESIZER,
                model="gpt-6-luna",
                estimated_input_tokens=1_000,
            )

    async def test_input_token_ceiling(self) -> None:
        tracker = UsageTracker(100, budget(max_cloud_input_tokens=10_000))
        with pytest.raises(CloudBudgetExceededError) as info:
            await tracker.reserve(
                Provider.OPENAI,
                role=ModelRole.VERIFIER,
                model="gpt-6-luna",
                estimated_input_tokens=20_000,
            )
        assert info.value.dimension == "input_tokens"

    async def test_output_token_ceiling(self) -> None:
        tracker = UsageTracker(100, budget(max_cloud_output_tokens=1_000))
        with pytest.raises(CloudBudgetExceededError) as info:
            await tracker.reserve(Provider.OPENAI, role=ModelRole.SYNTHESIZER, model="gpt-6-luna")
        assert info.value.dimension == "output_tokens"

    async def test_plain_call_ceiling_still_applies(self) -> None:
        tracker = UsageTracker(1, budget())
        await tracker.reserve(Provider.OLLAMA, role=ModelRole.RESEARCHER, model="q")
        with pytest.raises(BudgetExceededError):
            await tracker.reserve(Provider.OLLAMA, role=ModelRole.RESEARCHER, model="q")


class TestPerRoleOutputCaps:
    def test_roles_have_distinct_output_ceilings(self) -> None:
        caps = budget().max_output_tokens_per_call  # type: ignore[attr-defined]
        assert caps["verifier"] < caps["researcher"] < caps["synthesizer"]

    def test_cap_is_applied_to_the_built_client(self) -> None:
        """A runaway generation is billed in full before anything notices, so
        the ceiling has to reach the provider, not just our accounting."""
        from agentic_research.llm.router import ModelRouter

        settings = Settings(
            llm_mode="cloud",
            openai_api_key="sk-test",
            max_output_tokens_verifier=321,
            _env_file=None,
        )
        client = ModelRouter(settings).get(ModelRole.VERIFIER)._model
        # langchain-openai normalises max_completion_tokens onto max_tokens.
        assert getattr(client, "max_tokens", None) == 321


class TestSearchCreditCeiling:
    @respx.mock
    async def test_search_stops_at_the_credit_ceiling(self, settings: Settings) -> None:
        route = respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={"results": []}))
        settings.max_search_credits = 2
        service = SearchService(TavilyProvider("tvly-x"), settings)
        async with service:
            for i in range(4):
                await service.run_query(
                    SearchQuery(id=f"Q{i}", sub_question_id="SQ1", text="q", round_number=1)
                )
        assert route.call_count == 2, "third query must not be dispatched"
        assert service.stats.credits == 2

    @respx.mock
    async def test_exhausted_credits_return_an_error_not_an_exception(
        self, settings: Settings
    ) -> None:
        """A search worker that raised would take down its parallel round."""
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={"results": []}))
        settings.max_search_credits = 1
        service = SearchService(TavilyProvider("tvly-x"), settings)
        async with service:
            await service.run_query(
                SearchQuery(id="Q1", sub_question_id="SQ1", text="q", round_number=1)
            )
            response = await service.run_query(
                SearchQuery(id="Q2", sub_question_id="SQ1", text="q", round_number=1)
            )
        assert not response.ok
        assert "credit budget" in (response.error or "")


class TestLocalOutputCap:
    def test_local_models_get_a_generation_cap(self) -> None:
        """Measured: without num_predict, a 4B model asked for a research
        plan ran past 240s. With it, the call is bounded. The cap converts
        an unbounded hang into a bounded failure."""
        from agentic_research.llm.router import ModelRouter

        settings = Settings(
            llm_mode="local",
            ollama_model="qwen3:4b",
            max_output_tokens_planner=2_000,
            _env_file=None,
        )
        client = ModelRouter(settings).get(ModelRole.PLANNER)._model
        assert getattr(client, "num_predict", None) == 2_000

    def test_roles_get_their_own_local_cap(self) -> None:
        from agentic_research.llm.router import ModelRouter

        settings = Settings(
            llm_mode="local",
            max_output_tokens_verifier=400,
            max_output_tokens_synthesizer=6_000,
            _env_file=None,
        )
        router = ModelRouter(settings)
        assert router.get(ModelRole.VERIFIER)._model.num_predict == 400
        assert router.get(ModelRole.SYNTHESIZER)._model.num_predict == 6_000
