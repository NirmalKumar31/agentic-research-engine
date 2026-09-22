"""Tests that talk to real providers.

Excluded from default runs. Each one skips rather than fails when its
dependency is missing, so a contributor without credentials sees skips, not a
broken suite.
"""

from __future__ import annotations

import os

import httpx
import pytest

from agentic_research.config import ModelRole, Settings
from agentic_research.llm.router import ModelRouter
from agentic_research.models import SearchQuery
from agentic_research.schemas import AnalysisOut, ExtractionOut
from agentic_research.search.service import SearchService, build_provider

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")


def _ollama_running() -> bool:
    try:
        return httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


ollama = pytest.mark.skipif(not _ollama_running(), reason="Ollama is not reachable")
tavily = pytest.mark.skipif(
    not os.environ.get("TAVILY_API_KEY"), reason="TAVILY_API_KEY is not set"
)


@pytest.fixture
def local_settings() -> Settings:
    return Settings(
        llm_mode="local",
        ollama_model=OLLAMA_MODEL,
        ollama_base_url=OLLAMA_URL,
        llm_timeout_seconds=300,
        max_parallel_local_llm_calls=1,
        persist_runs=False,
        checkpoint_backend="memory",
        _env_file=None,
    )


@pytest.mark.ollama
@ollama
class TestLocalModel:
    async def test_preflight_accepts_an_installed_model(self, local_settings: Settings) -> None:
        assert await ModelRouter(local_settings).preflight() == []

    async def test_structured_output_works_on_a_small_local_model(
        self, local_settings: Settings
    ) -> None:
        """The whole local mode depends on this. Small models are the ones
        most likely to return prose where a JSON object was requested."""
        router = ModelRouter(local_settings)
        await router.preflight()
        out = await router.get(ModelRole.PLANNER).structured(
            AnalysisOut,
            "You analyse research questions.",
            "Question: Compare approaches for detecting fraud in imbalanced datasets.",
        )
        assert out.normalized_query.strip()
        assert out.output_format in {
            "comparison",
            "overview",
            "howto",
            "timeline",
            "decision_support",
        }

    async def test_usage_is_reported_for_local_calls(self, local_settings: Settings) -> None:
        router = ModelRouter(local_settings)
        await router.preflight()
        await router.get(ModelRole.PLANNER).structured(
            AnalysisOut, "You analyse questions.", "Question: what is a hash join?"
        )
        totals = router.tracker.totals()
        assert totals.calls == 1
        assert totals.input_tokens > 0
        # Local inference has no per-token vendor charge, but it is priced,
        # so cost is complete rather than unknown.
        assert totals.cost_is_complete
        assert totals.known_cost_usd == 0.0

    async def test_extraction_returns_quotes_present_in_the_source(
        self, local_settings: Settings
    ) -> None:
        """Quote fidelity is the property the evidence model rests on."""
        from agentic_research.evidence.store import verify_quote
        from agentic_research.graph.prompts import EXTRACTOR_SYSTEM, extractor_user

        text = (
            "Class imbalance is the central difficulty in fraud detection. "
            "Fraudulent transactions typically represent far less than one percent "
            "of all records. Cost-sensitive learning assigns a higher penalty to "
            "missed fraud cases, and precision-recall curves are more informative "
            "than ROC curves when the positive class is rare."
        )
        router = ModelRouter(local_settings)
        await router.preflight()
        out = await router.get(ModelRole.RESEARCHER).structured(
            ExtractionOut,
            EXTRACTOR_SYSTEM,
            extractor_user("SQ1: How is fraud detection evaluated?", "Fraud notes", text),
        )
        assert out.evidence, "expected at least one finding from an on-topic source"
        verified = [e for e in out.evidence if verify_quote(e.quote, text)]
        assert verified, (
            "no quote could be located in the source; extraction is paraphrasing "
            f"instead of quoting: {[e.quote for e in out.evidence]}"
        )


@pytest.mark.integration
@tavily
class TestLiveSearch:
    async def test_search_returns_normalised_results(self) -> None:
        settings = Settings(
            llm_mode="local",
            tavily_api_key=os.environ["TAVILY_API_KEY"],
            max_search_results=5,
            _env_file=None,
        )
        async with SearchService(build_provider(settings), settings) as service:
            response = await service.run_query(
                SearchQuery(
                    id="Q1",
                    sub_question_id="SQ1",
                    text="class imbalance fraud detection evaluation metrics",
                    round_number=1,
                )
            )

        assert response.ok, response.error
        assert response.results
        for result in response.results:
            assert result.url.startswith("http")
            assert result.title
            assert result.provider == "tavily"
            assert result.query_id == "Q1"
        assert service.stats.credits >= 1.0


@pytest.mark.integration
@tavily
class TestEndToEnd:
    async def test_full_run_produces_a_verified_report(self) -> None:
        """The acceptance test: a real question, real sources, real citations.

        Runs in local mode so it needs no OpenAI key; only search is paid.
        """
        from agentic_research.runner import run_research

        settings = Settings(
            llm_mode="local",
            ollama_model=OLLAMA_MODEL,
            ollama_base_url=OLLAMA_URL,
            tavily_api_key=os.environ["TAVILY_API_KEY"],
            max_research_rounds=1,
            max_search_queries=4,
            max_sources=6,
            max_sources_per_round=4,
            llm_timeout_seconds=300,
            max_parallel_local_llm_calls=1,
            persist_runs=False,
            checkpoint_backend="memory",
            _env_file=None,
        )
        if not _ollama_running():
            pytest.skip("Ollama is not reachable")

        result = await run_research(
            "What evaluation metrics are appropriate for fraud detection models "
            "trained on highly imbalanced data?",
            settings,
        )

        assert result.markdown.startswith("# ")
        assert result.metrics.unique_sources > 0
        assert result.metrics.evidence_items > 0

        # Every citation must resolve to a source actually retrieved. This is
        # the property the citation subsystem exists to guarantee.
        assert result.metrics.citation_validity_rate == 1.0

        source_ids = {s.id for s in result.state["sources"]}
        for item in result.state["evidence"]:
            assert item.source_id in source_ids
