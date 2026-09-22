"""Configuration loading, model-spec parsing and role resolution."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_research.config import (
    LLMMode,
    ModelRole,
    ModelSpec,
    Provider,
    Settings,
)


class TestModelSpec:
    def test_parses_provider_and_model(self) -> None:
        spec = ModelSpec.parse("openai:gpt-6-sol")
        assert spec.provider is Provider.OPENAI
        assert spec.model == "gpt-6-sol"

    def test_ollama_tag_colon_is_preserved(self) -> None:
        """Splitting on every colon would turn qwen3:4b into qwen3 and request
        a model that does not exist."""
        spec = ModelSpec.parse("ollama:qwen3:4b")
        assert spec.provider is Provider.OLLAMA
        assert spec.model == "qwen3:4b"

    def test_roundtrips_through_str(self) -> None:
        assert str(ModelSpec.parse("ollama:llama3.2:3b")) == "ollama:llama3.2:3b"

    @pytest.mark.parametrize("bad", ["gpt-6-sol", "anthropic:claude", "openai:", "  ", ":model"])
    def test_rejects_malformed_specs(self, bad: str) -> None:
        with pytest.raises(ValueError):
            ModelSpec.parse(bad)

    def test_error_names_the_supported_providers(self) -> None:
        with pytest.raises(ValueError, match="openai"):
            ModelSpec.parse("cohere:command")


class TestModeDefaults:
    def test_local_mode_uses_ollama_everywhere(self) -> None:
        settings = Settings(llm_mode="local", ollama_model="qwen3:4b", _env_file=None)
        assert all(spec.provider is Provider.OLLAMA for spec in settings.resolve_models().values())

    def test_cloud_mode_uses_the_cheap_model_for_the_researcher_role(self) -> None:
        """Extraction runs once per source and is the highest-volume role."""
        settings = Settings(llm_mode="cloud", openai_api_key="sk-x", _env_file=None)
        resolved = settings.resolve_models()
        assert resolved[ModelRole.RESEARCHER].model == settings.openai_fast_model
        assert resolved[ModelRole.SYNTHESIZER].model == settings.openai_model

    def test_hybrid_puts_only_the_researcher_local(self) -> None:
        settings = Settings(llm_mode="hybrid", openai_api_key="sk-x", _env_file=None)
        resolved = settings.resolve_models()
        assert resolved[ModelRole.RESEARCHER].provider is Provider.OLLAMA
        for role in (
            ModelRole.PLANNER,
            ModelRole.CRITIC,
            ModelRole.SYNTHESIZER,
            ModelRole.VERIFIER,
        ):
            assert resolved[role].provider is Provider.OPENAI

    def test_per_role_override_wins(self) -> None:
        settings = Settings(
            llm_mode="local",
            ollama_model="qwen3:4b",
            synthesizer_model="openai:gpt-6-astra",
            openai_api_key="sk-x",
            _env_file=None,
        )
        resolved = settings.resolve_models()
        assert resolved[ModelRole.SYNTHESIZER] == ModelSpec.parse("openai:gpt-6-astra")
        assert resolved[ModelRole.PLANNER].provider is Provider.OLLAMA


class TestValidation:
    def test_cloud_mode_without_a_key_fails_at_startup(self) -> None:
        """Failing here costs a second; failing lazily costs a partial run."""
        with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
            Settings(llm_mode="cloud", _env_file=None)

    def test_local_mode_needs_no_openai_key(self) -> None:
        Settings(llm_mode="local", _env_file=None)

    def test_an_openai_role_override_requires_a_key_even_in_local_mode(self) -> None:
        with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
            Settings(llm_mode="local", critic_model="openai:gpt-6-sol", _env_file=None)

    def test_per_round_source_cap_cannot_exceed_the_total(self) -> None:
        with pytest.raises(ValidationError, match="MAX_SOURCES_PER_ROUND"):
            Settings(llm_mode="local", max_sources=5, max_sources_per_round=10, _env_file=None)

    @pytest.mark.parametrize("value", ["deep", "BASIC ", "fastest"])
    def test_search_depth_is_validated(self, value: str) -> None:
        if value.strip().lower() in {"basic", "advanced"}:
            assert Settings(llm_mode="local", search_depth=value, _env_file=None)
        else:
            with pytest.raises(ValidationError):
                Settings(llm_mode="local", search_depth=value, _env_file=None)

    def test_log_level_is_normalised(self) -> None:
        assert Settings(llm_mode="local", log_level="debug", _env_file=None).log_level == "DEBUG"

    def test_bad_checkpoint_backend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings(llm_mode="local", checkpoint_backend="postgres", _env_file=None)

    def test_secrets_are_not_printed_in_repr(self) -> None:
        settings = Settings(llm_mode="local", tavily_api_key="tvly-supersecret", _env_file=None)
        assert "tvly-supersecret" not in repr(settings)
        assert settings.tavily_api_key is not None
        assert settings.tavily_api_key.get_secret_value() == "tvly-supersecret"


class TestBudget:
    def test_budget_is_derived_from_settings(self) -> None:
        settings = Settings(
            llm_mode="local", max_research_rounds=4, max_llm_calls=33, _env_file=None
        )
        budget = settings.budget
        assert budget.max_research_rounds == 4
        assert budget.max_llm_calls == 33

    def test_environment_variables_are_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MODE", "local")
        monkeypatch.setenv("MAX_RESEARCH_ROUNDS", "7")
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
        settings = Settings(_env_file=None)
        assert settings.llm_mode is LLMMode.LOCAL
        assert settings.max_research_rounds == 7
        assert settings.resolve_models()[ModelRole.PLANNER].model == "llama3.2:3b"
