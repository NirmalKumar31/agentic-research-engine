"""Typed application configuration.

All runtime knobs live here so that the graph, providers and CLI never read
``os.environ`` directly. Two things matter about the design:

* Model selection is expressed per *role*, not per call site. Application code
  asks for ``ModelRole.SYNTHESIZER`` and configuration decides whether that is
  a cloud or a local model.
* Budgets are part of configuration rather than scattered constants, because
  every one of them is a hard stop that the graph is required to honour.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMMode(StrEnum):
    """Where work runs by default."""

    CLOUD = "cloud"
    LOCAL = "local"
    HYBRID = "hybrid"


class Provider(StrEnum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class ModelRole(StrEnum):
    """Logical jobs a model can be asked to do.

    Roles are deliberately coarse. Finer roles would give more routing control
    but would make configuration tedious, and the cost difference between, say,
    query generation and evidence extraction is not worth another knob.
    """

    PLANNER = "planner"
    """Query analysis and research decomposition. Reasoning-heavy."""

    RESEARCHER = "researcher"
    """Query writing and per-source evidence extraction. High call volume,
    mechanical. The natural place to spend a local model instead of a cloud one."""

    CRITIC = "critic"
    """Coverage and gap analysis. Decides whether another round is worth it."""

    SYNTHESIZER = "synthesizer"
    """Final report generation. Quality here is what the user actually sees."""

    VERIFIER = "verifier"
    """Claim/evidence entailment checking during citation verification."""


class ModelSpec(BaseModel):
    """A resolved provider + model pair."""

    model_config = {"frozen": True}

    provider: Provider
    model: str

    @classmethod
    def parse(cls, spec: str) -> ModelSpec:
        """Parse a ``"<provider>:<model>"`` string.

        Split on the *first* colon only. Ollama model tags embed a colon
        (``qwen3:4b``), so a naive split would silently truncate the tag and
        request a model that does not exist.
        """
        raw = spec.strip()
        if ":" not in raw:
            raise ValueError(
                f"Model spec {spec!r} must be '<provider>:<model>', e.g. 'openai:gpt-6-sol'"
            )
        provider_part, model_part = raw.split(":", 1)
        provider_part = provider_part.strip().lower()
        model_part = model_part.strip()
        if not model_part:
            raise ValueError(f"Model spec {spec!r} is missing a model name")
        try:
            provider = Provider(provider_part)
        except ValueError:
            supported = ", ".join(p.value for p in Provider)
            raise ValueError(
                f"Unknown provider {provider_part!r} in {spec!r}. Supported: {supported}"
            ) from None
        return cls(provider=provider, model=model_part)

    def __str__(self) -> str:
        return f"{self.provider.value}:{self.model}"


class RunBudget(BaseModel):
    """Hard ceilings for a single research run.

    These exist because an agent with a loop and a wallet is a liability. Every
    limit is enforced before the work is dispatched, not after it is billed.
    """

    model_config = {"frozen": True}

    max_research_rounds: int = Field(ge=1, le=10)
    max_search_queries: int = Field(ge=1)
    max_sources: int = Field(ge=1)
    max_sources_per_round: int = Field(ge=1)
    max_llm_calls: int = Field(ge=1)
    max_parallel_searches: int = Field(ge=1, le=32)
    max_parallel_fetches: int = Field(ge=1, le=64)


class Settings(BaseSettings):
    """Application settings, loaded from environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Credentials -------------------------------------------------------
    openai_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    brave_api_key: SecretStr | None = None

    # --- Mode and models ---------------------------------------------------
    llm_mode: LLMMode = LLMMode.HYBRID
    openai_model: str = "gpt-6-sol"
    openai_fast_model: str = "gpt-6-luna"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"

    planner_model: str | None = None
    researcher_model: str | None = None
    critic_model: str | None = None
    synthesizer_model: str | None = None
    verifier_model: str | None = None

    allow_cloud_fallback: bool = False

    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=180.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    max_parallel_local_llm_calls: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Concurrent requests to a local model. Ollama serialises per model, "
        "so a high value adds queueing latency rather than throughput.",
    )
    local_num_ctx: int = Field(default=8192, ge=2048)

    # --- Search ------------------------------------------------------------
    search_provider: str = "tavily"
    search_depth: str = "basic"
    search_escalate_on_followup: bool = True
    max_search_results: int = Field(default=8, ge=1, le=20)
    search_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Budgets -----------------------------------------------------------
    max_research_rounds: int = Field(default=3, ge=1, le=10)
    max_search_queries: int = Field(default=24, ge=1)
    max_sources: int = Field(default=40, ge=1)
    max_sources_per_round: int = Field(default=12, ge=1)
    max_llm_calls: int = Field(default=60, ge=1)
    max_parallel_searches: int = Field(default=5, ge=1, le=32)
    max_parallel_fetches: int = Field(default=8, ge=1, le=64)

    # --- Retrieval ---------------------------------------------------------
    fetch_timeout_seconds: float = Field(default=15.0, gt=0)
    max_page_bytes: int = Field(default=2_000_000, ge=10_000)
    max_extract_chars: int = Field(default=12_000, ge=1_000)
    user_agent: str = (
        "AgenticResearchEngine/0.1 (+https://github.com/NirmalKumar31/agentic-research-engine)"
    )

    # --- Output and persistence -------------------------------------------
    output_dir: Path = Path("outputs")
    persist_runs: bool = True
    checkpoint_backend: str = "sqlite"
    checkpoint_path: Path = Path("checkpoints/research.sqlite")

    # --- Observability -----------------------------------------------------
    log_level: str = "INFO"
    log_format: str = "console"
    enable_langsmith: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "agentic-research-engine"

    @field_validator("search_depth")
    @classmethod
    def _check_depth(cls, v: str) -> str:
        allowed = {"basic", "advanced"}
        value = v.strip().lower()
        if value not in allowed:
            raise ValueError(f"search_depth must be one of {sorted(allowed)}, got {v!r}")
        return value

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        value = v.strip().upper()
        if value not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {v!r}")
        return value

    @field_validator("checkpoint_backend")
    @classmethod
    def _check_backend(cls, v: str) -> str:
        allowed = {"sqlite", "memory", "none"}
        value = v.strip().lower()
        if value not in allowed:
            raise ValueError(f"checkpoint_backend must be one of {sorted(allowed)}, got {v!r}")
        return value

    @model_validator(mode="after")
    def _check_mode_credentials(self) -> Settings:
        """Fail at startup rather than three minutes into a run."""
        needs_cloud = self.llm_mode in (LLMMode.CLOUD, LLMMode.HYBRID) or any(
            spec and spec.strip().lower().startswith("openai:")
            for spec in self._role_overrides().values()
        )
        if needs_cloud and not self.openai_api_key:
            raise ValueError(
                f"LLM_MODE={self.llm_mode.value} requires OPENAI_API_KEY. "
                "Set it in .env, or use LLM_MODE=local to run entirely on Ollama."
            )
        if self.max_sources_per_round > self.max_sources:
            raise ValueError("MAX_SOURCES_PER_ROUND cannot exceed MAX_SOURCES")
        return self

    def _role_overrides(self) -> dict[ModelRole, str | None]:
        return {
            ModelRole.PLANNER: self.planner_model,
            ModelRole.RESEARCHER: self.researcher_model,
            ModelRole.CRITIC: self.critic_model,
            ModelRole.SYNTHESIZER: self.synthesizer_model,
            ModelRole.VERIFIER: self.verifier_model,
        }

    def _default_spec(self, role: ModelRole) -> ModelSpec:
        """Role default for the current mode.

        The hybrid split follows one rule: a role goes local when its output is
        short, schema-constrained and produced many times per run; it stays in
        the cloud when a wrong answer changes the final report. Evidence
        extraction is the clearest local candidate — it runs once per source
        and its job is quotation, not judgement.
        """
        local = ModelSpec(provider=Provider.OLLAMA, model=self.ollama_model)
        cloud = ModelSpec(provider=Provider.OPENAI, model=self.openai_model)
        cloud_fast = ModelSpec(provider=Provider.OPENAI, model=self.openai_fast_model)

        if self.llm_mode is LLMMode.LOCAL:
            return local
        if self.llm_mode is LLMMode.CLOUD:
            return cloud_fast if role is ModelRole.RESEARCHER else cloud
        return local if role is ModelRole.RESEARCHER else cloud

    def resolve_models(self) -> dict[ModelRole, ModelSpec]:
        """Final role → model mapping, overrides applied."""
        resolved: dict[ModelRole, ModelSpec] = {}
        for role, override in self._role_overrides().items():
            resolved[role] = ModelSpec.parse(override) if override else self._default_spec(role)
        return resolved

    @property
    def budget(self) -> RunBudget:
        return RunBudget(
            max_research_rounds=self.max_research_rounds,
            max_search_queries=self.max_search_queries,
            max_sources=self.max_sources,
            max_sources_per_round=self.max_sources_per_round,
            max_llm_calls=self.max_llm_calls,
            max_parallel_searches=self.max_parallel_searches,
            max_parallel_fetches=self.max_parallel_fetches,
        )

    def uses_provider(self, provider: Provider) -> bool:
        return any(spec.provider is provider for spec in self.resolve_models().values())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so that importing modules do not each re-read ``.env``. Tests call
    ``get_settings.cache_clear()`` to pick up patched environments.
    """
    return Settings()
