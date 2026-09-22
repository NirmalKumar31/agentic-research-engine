"""Shared fixtures.

Default runs never touch a paid API. Anything that needs real credentials is
marked ``integration`` and deselected unless explicitly requested.
"""

from __future__ import annotations

import pytest

from agentic_research.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings that need no credentials."""
    return Settings(
        llm_mode="local",
        ollama_model="test-model:1b",
        tavily_api_key="tvly-test-key",
        max_research_rounds=2,
        max_search_queries=8,
        max_sources=10,
        max_sources_per_round=5,
        max_llm_calls=20,
        max_parallel_searches=3,
        max_parallel_fetches=4,
        persist_runs=False,
        checkpoint_backend="memory",
        _env_file=None,
    )


@pytest.fixture(autouse=True)
def _quiet_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "CRITICAL")
