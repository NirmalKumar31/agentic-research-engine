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

import re
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_research import __version__ as _VERSION

# One source of truth for the pinned verifier. Duplicating the revision
# here and in the NLI adapter let them drift, and the adapter's copy was
# the unpinned string "main".
from agentic_research.citations.nli_pin import (
    NLI_DEFAULT_MODEL_ID,
    NLI_DEFAULT_REVISION,
)


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


class CloudBudget(BaseModel):
    """Hard ceilings on money and paid capacity for a single run.

    Separate from :class:`RunBudget` because the units are different and the
    consequences are different: exceeding a round limit wastes time, while
    exceeding a spend limit costs money that cannot be reclaimed. These apply
    only to cloud providers; local inference is unmetered here.

    Checked before dispatch using the worst case a call could cost, so a
    configured ceiling is never silently exceeded.
    """

    model_config = {"frozen": True}

    max_cloud_calls: int = Field(ge=0)
    max_cloud_input_tokens: int = Field(ge=0)
    max_cloud_output_tokens: int = Field(ge=0)
    max_cloud_cost_usd: float = Field(ge=0.0)
    max_search_credits: float = Field(ge=0.0)
    max_output_tokens_per_call: dict[str, int] = Field(default_factory=dict)

    def output_cap_for(self, role: str) -> int | None:
        return self.max_output_tokens_per_call.get(role)


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
    max_run_seconds: float = Field(default=3600.0, gt=0)
    """Wall-clock ceiling for one research run.

    Separate from the per-call deadline because many individually
    bounded calls still compose into an unbounded run: fifty calls at
    three minutes each is two and a half hours, and nothing below this
    level would notice. Counted in elapsed time, not in calls times
    nominal timeout, so a slow provider cannot quietly extend it."""


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
    llm_mode: LLMMode = LLMMode.LOCAL
    """Local by default, matching .env.example and the quickstart.

    The default was HYBRID while both of those said local, so a fresh
    checkout failed to start with a message about a missing API key for
    a mode the documentation never told anyone they were in."""
    openai_model: str = "gpt-6-sol"
    openai_fast_model: str = "gpt-6-luna"
    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str = "https://api.openai.com/v1"
    """Overridable for a compatible gateway. Used by the model-retrieval
    preflight as well as by the client."""
    ollama_model: str = "qwen3:4b"

    planner_model: str | None = None
    researcher_model: str | None = None
    critic_model: str | None = None
    synthesizer_model: str | None = None

    allow_cloud_fallback: bool = False

    # --- semantic verification ------------------------------------------
    # Claim support is decided by a dedicated NLI classifier, not by the
    # generative models above. Three attempts at using a 4B instruction
    # model for this failed in three different ways; docs/LIMITATIONS.md
    # records what each measured.
    #
    # All three of model, revision and threshold are pinned and recorded
    # in every judgment. Changing any one changes which claims publish,
    # so a verdict that does not say which it used cannot be audited.
    nli_model_id: str = NLI_DEFAULT_MODEL_ID
    nli_model_revision: str = NLI_DEFAULT_REVISION
    nli_support_threshold: float = Field(default=0.98, ge=0.0, le=1.0)
    """Entailment probability a cited quote must reach. Calibrated, not
    guessed -- see examples/verifier-calibration/nli-calibration.json."""

    nli_mode: Literal["local", "remote"] = "local"
    """``local`` loads the checkpoint in-process, which needs roughly
    1.4GB of RSS. ``remote`` calls a hosted inference endpoint instead,
    for hosts too small to hold the model."""
    nli_endpoint: str | None = None
    nli_api_key: SecretStr | None = None
    nli_dialect: Literal["contract", "hf"] = "contract"
    """Wire format the remote speaks.

    ``contract`` is this project's own scoring service, which echoes the
    model, revision, per-pair id and truncation flag, so each response
    is checked against the request that produced it.

    ``hf`` is a managed Hugging Face Inference Endpoint on the stock
    text-classification handler, which echoes none of those. Choosing it
    moves the revision check to preflight, against the Hugging Face
    control plane, and moves truncation detection into this process."""
    nli_timeout_seconds: float = Field(default=30.0, gt=0)
    nli_scale_up_timeout_seconds: float = Field(default=0.0, ge=0)
    """Bounded wait for a scaled-to-zero endpoint to wake. 0 disables
    warm-up waiting entirely, which is the correct setting until an
    always-warm deployment has passed acceptance."""

    # --- durable public-demo quota ---------------------------------------
    demo_quota_url: str | None = None
    """Redis/Key-Value URL backing the global daily run cap.

    The in-process counter bounds a process, not a day: a free instance
    sleeps and wakes with the count at zero. Harmless for replay, where
    a run costs nothing; not harmless for live research, which spends at
    three providers per run."""

    demo_quota_required: bool = True
    """Refuse live research when durable counting is unavailable.

    An unbounded fallback is precisely the failure the counter exists to
    prevent, so the default is to fail closed. Replay never consults
    it."""

    demo_quota_namespace: str = ""
    """Separates one deployment's daily counter from another's.

    Two deployments pointed at one Key Value store share a key, so a
    staging run would spend production's allowance for the day. The
    namespace is inserted between the fixed prefix and the UTC date,
    and the *default is empty*: an unset namespace produces exactly the
    key production already uses, so adopting this setting does not
    silently reset a live counter mid-day.

    A colon is the key separator, so a namespace may not contain one --
    otherwise a staging value could be written to address production's
    key. The validator enforces that rather than trusting the operator,
    because the whole point of the setting is isolation."""

    trusted_proxy_hops: int = Field(default=0, ge=0, le=4)
    """How many trailing X-Forwarded-For entries were added by proxies we
    control. 0 means the header is not trusted at all: a client can send
    any value, so only entries appended by our own boundary carry
    meaning."""

    # --- cloud spend ceilings -------------------------------------------
    # Zero disables a dimension rather than meaning "no spend allowed";
    # an explicitly zero-budget run would be indistinguishable from an
    # unconfigured one, so `0` is read as unlimited and documented as such.
    # Ceiling on HTTP requests to the model provider. Distinct from
    # MAX_LLM_CALLS, which bounds logical model calls: one logical call can
    # emit several requests (repair, compatibility retry, transport retry)
    # and providers rate-limit on requests, not on our abstraction.
    # Search is not counted here -- it is metered by the search provider
    # in credits and bounded by MAX_SEARCH_CREDITS.
    max_provider_requests: int = Field(default=120, ge=0)
    # Cloud request ceiling. Named max_cloud_calls for continuity, but it is
    # now counted in provider requests rather than logical calls.
    max_cloud_calls: int = Field(default=40, ge=0)
    max_cloud_input_tokens: int = Field(default=400_000, ge=0)
    max_cloud_output_tokens: int = Field(default=60_000, ge=0)
    max_cloud_cost_usd: float = Field(default=0.50, ge=0.0)
    """Estimated application spend limit for one run. Not a billing cap.

    The engine reserves against this before every provider request, but
    the input side of the reservation is an estimate: token counts are
    approximated from character length, which undershoots on code,
    punctuation-dense text, non-Latin scripts and structured-output
    schemas. The reservation also covers the messages this engine
    assembles, not whatever the provider adds around them.

    So it bounds what the engine believes it is about to spend, which is
    enough to stop a runaway loop and not enough to be called a
    guarantee. **The financial backstop is the hard limit set on the
    provider account or project**, which is enforced by the provider
    against real usage. Set one."""
    max_search_credits: float = Field(default=50.0, ge=0.0)

    # Per-role output ceilings. Synthesis legitimately needs room; a query
    # writer emitting 4k tokens is a malfunction, not a long answer.
    max_output_tokens_planner: int = Field(default=2_000, ge=64)
    max_output_tokens_researcher: int = Field(
        default=3_000,
        ge=64,
        description=(
            "Extraction returns up to six findings, each with a verbatim quote. "
            "1500 was measured to truncate real responses mid-JSON, which the "
            "structured-output layer then reports as a parse failure."
        ),
    )
    max_output_tokens_critic: int = Field(default=1_500, ge=64)
    max_output_tokens_synthesizer: int = Field(default=6_000, ge=64)

    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=180.0, gt=0)
    run_timeout_seconds: float = Field(default=3600.0, gt=0)
    """Wall-clock ceiling for one whole research run. See RunBudget."""
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description=(
            "Ollama only. OpenAI clients are built with max_retries=0 so no "
            "provider request escapes the router's accounting."
        ),
    )
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
    max_pdf_bytes: int = Field(
        default=15_000_000, ge=10_000, description="PDFs are legitimately larger than pages"
    )
    max_pdf_pages: int = Field(default=60, ge=1, le=2_000)
    max_extract_chars: int = Field(default=12_000, ge=1_000)
    user_agent: str = (
        f"AgenticResearchEngine/{_VERSION}"
        " (+https://github.com/NirmalKumar31/agentic-research-engine)"
    )

    # --- Output and persistence -------------------------------------------
    output_dir: Path = Path("outputs")
    persist_runs: bool = True
    checkpoint_backend: str = "sqlite"
    checkpoint_path: Path = Path("checkpoints/research.sqlite")

    # --- Hosted demo -------------------------------------------------------
    # Server-controlled. When true the API clamps every run to the fixed
    # demo ceilings regardless of what the client asks for.
    demo_mode: bool = False
    live_research_enabled: bool = Field(
        default=False,
        description=(
            "Whether an anonymous HTTP request may start a paid research run. "
            "Off by default: turning it on requires provider credentials, a "
            "pinned remote verifier and a reachable shared quota store, and a "
            "deployment missing any of them is refused at startup rather than "
            "booting half-configured. When on, the daily admission count is "
            "held in that shared store, so it is one count across replicas and "
            "survives a web cold start. The public site serves recorded runs "
            "instead, and the CLI is unaffected."
        ),
    )
    cors_origins: str = Field(
        default="", description="Comma-separated allowed origins; empty disables CORS"
    )
    demo_max_runtime_seconds: float = Field(
        default=240.0,
        gt=0,
        description=(
            "Wall-clock ceiling for a hosted run. 240s suits a cloud model; a "
            "local 4B model needs far longer and is not viable for a public demo."
        ),
    )
    demo_runs_per_hour: int = Field(default=3, ge=1)
    demo_max_concurrent_runs: int = Field(default=2, ge=1)
    demo_provider_requests_per_day: int = Field(
        default=50,
        ge=1,
        description=(
            "Account-level provider request quota. The demo's daily run cap is "
            "derived from this, because a run costs ~22 requests and picking the "
            "two numbers independently ends in a 429 mid-run."
        ),
    )

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

    @field_validator("demo_quota_namespace")
    @classmethod
    def _check_quota_namespace(cls, v: str) -> str:
        """Reject anything that could address another deployment's key.

        Empty means "no namespace" and is the production default. A
        non-empty value is restricted to lowercase alphanumerics and
        hyphens: no colon (the key separator), no whitespace, nothing
        that varies with locale or case folding. Length is bounded so
        the key stays a key rather than a payload.
        """
        value = v.strip()
        if not value:
            return ""
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", value):
            raise ValueError(
                "demo_quota_namespace must be 1-32 characters of lowercase "
                f"letters, digits or hyphens and start with a letter or digit, got {v!r}"
            )
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
    def cloud_budget(self) -> CloudBudget:
        return CloudBudget(
            max_cloud_calls=self.max_cloud_calls,
            max_cloud_input_tokens=self.max_cloud_input_tokens,
            max_cloud_output_tokens=self.max_cloud_output_tokens,
            max_cloud_cost_usd=self.max_cloud_cost_usd,
            max_search_credits=self.max_search_credits,
            max_output_tokens_per_call={
                ModelRole.PLANNER.value: self.max_output_tokens_planner,
                ModelRole.RESEARCHER.value: self.max_output_tokens_researcher,
                ModelRole.CRITIC.value: self.max_output_tokens_critic,
                ModelRole.SYNTHESIZER.value: self.max_output_tokens_synthesizer,
            },
        )

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
            max_run_seconds=self.run_timeout_seconds,
        )

    @model_validator(mode="after")
    def _live_research_needs_a_complete_path(self) -> Settings:
        """A hosted live deployment must not boot healthy half-configured.

        Live research spends money at three providers and publishes
        nothing without a verifier. Booting with local NLI on a 512MB
        web host, or with no endpoint at all, produces exactly the
        failure this check exists to stop: a paid run whose every claim
        is withheld because verification was never possible.
        """
        if not self.live_research_enabled:
            return self

        problems: list[str] = []
        if self.nli_mode != "remote":
            problems.append("NLI_MODE must be 'remote' (the web image carries no torch)")
        elif not self.nli_endpoint:
            problems.append("NLI_ENDPOINT is required when NLI_MODE=remote")
        if self.nli_mode == "remote" and not self.nli_api_key:
            problems.append("NLI_API_KEY is required for a private endpoint")
        if len(self.nli_model_revision or "") != 40:
            problems.append("NLI_MODEL_REVISION must be a full 40-character commit")
        if self.llm_mode is LLMMode.CLOUD and not self.openai_api_key:
            problems.append("OPENAI_API_KEY is required for LLM_MODE=cloud")
        if not self.tavily_api_key and not self.brave_api_key:
            problems.append("a search provider key is required")
        if self.demo_quota_required and not self.demo_quota_url:
            problems.append(
                "DEMO_QUOTA_URL is required, or set DEMO_QUOTA_REQUIRED=false "
                "to accept a process-local cap"
            )
        if problems:
            raise ValueError(
                "LIVE_RESEARCH_ENABLED=true but the live path is incomplete: " + "; ".join(problems)
            )
        return self

    def uses_provider(self, provider: Provider) -> bool:
        return any(spec.provider is provider for spec in self.resolve_models().values())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so that importing modules do not each re-read ``.env``. Tests call
    ``get_settings.cache_clear()`` to pick up patched environments.
    """
    return Settings()
