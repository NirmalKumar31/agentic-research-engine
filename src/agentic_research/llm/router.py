"""Role-based model routing.

Application code never constructs a provider. It asks for a *role* and the
router decides what backs it::

    report = await router.get(ModelRole.SYNTHESIZER).structured(ReportOut, ...)

That indirection is what makes cloud/local/hybrid a configuration change rather
than a code change, and it gives the engine exactly one place to meter calls,
retry malformed output and handle a provider being down.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar, cast

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agentic_research.config import LLMMode, ModelRole, ModelSpec, Provider, Settings
from agentic_research.llm.base import (
    LLMCallRecord,
    LLMError,
    ModelTimeoutError,
    ModelUnavailableError,
    ProviderRateLimited,
    ProviderRejectedRequest,
    StructuredOutputError,
    UsageTracker,
)
from agentic_research.observability import get_logger

log = get_logger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _is_rate_limit(exc: Exception) -> bool:
    """Detect a provider quota or rate refusal.

    Matched on the message rather than the SDK's exception class so the
    router does not need to import provider-specific types, and so it keeps
    working if a provider is added or an SDK renames its errors.
    """
    text = str(exc).lower()
    return (
        "error code: 429" in text
        or "rate limit" in text
        or "rate_limit" in text
        or "quota" in text
        or "insufficient_quota" in text
    )


def _is_temperature_rejection(exc: Exception) -> bool:
    text = str(exc).lower()
    return "temperature" in text and ("does not support" in text or "unsupported value" in text)


_REPAIR_TEMPLATE = (
    "Your previous response could not be parsed into the required schema.\n"
    "Error: {error}\n"
    "Return only a JSON object matching the schema exactly. Do not add commentary, "
    "markdown fences, or fields that are not in the schema."
)


class RoleModel:
    """A model bound to one role, with metering and output repair.

    Not a provider abstraction of its own: LangChain's ``BaseChatModel`` already
    is one, and reimplementing it would mean re-solving retries, streaming and
    tool schemas for no benefit. What this adds is the engine's concerns —
    budget, usage accounting, structured-output repair and fallback.
    """

    def __init__(
        self,
        role: ModelRole,
        spec: ModelSpec,
        model: BaseChatModel,
        tracker: UsageTracker,
        *,
        fell_back: bool = False,
        repair_attempts: int = 1,
        gate: asyncio.Semaphore | None = None,
        rebuild_without_temperature: Callable[[], BaseChatModel] | None = None,
    ) -> None:
        self.role = role
        self.spec = spec
        self._model = model
        self._tracker = tracker
        self._fell_back = fell_back
        self._repair_attempts = repair_attempts
        self._gate = gate
        self._rebuild_without_temperature = rebuild_without_temperature

    async def structured(
        self,
        schema: type[SchemaT],
        system: str,
        user: str,
        *,
        repair: bool = True,
    ) -> SchemaT:
        """Invoke the model and return a validated instance of ``schema``.

        Uses ``include_raw=True`` so a schema violation comes back as data
        rather than an exception. That matters for two reasons: token usage is
        still on the raw message even when parsing failed (so a failed call is
        still billed and still counted), and it lets us feed the validation
        error back to the model instead of losing the attempt.
        """
        # Rough token estimate for the pre-dispatch spend check. Deliberately
        # a cheap approximation (~4 chars/token): exact counting would need a
        # tokeniser per model, and the check only needs to be conservative,
        # not precise. The ceiling is enforced again by the recorded actuals.
        estimated_input = (len(system) + len(user)) // 4
        await self._tracker.reserve(
            self.spec.provider,
            role=self.role,
            model=self.spec.model,
            estimated_input_tokens=estimated_input,
        )
        if self._gate is not None:
            async with self._gate:
                return await self._invoke_handling_quirks(schema, system, user, repair=repair)
        return await self._invoke_handling_quirks(schema, system, user, repair=repair)

    async def _invoke_handling_quirks(
        self, schema: type[SchemaT], system: str, user: str, *, repair: bool
    ) -> SchemaT:
        """Absorb provider parameter quirks rather than making callers know them.

        Some OpenAI models accept only the default temperature and reject any
        explicit value with a 400. Which models those are is not discoverable
        without asking, so the router tries, and on that specific rejection
        rebuilds the client without the parameter and retries once. The
        rejected request is a 400, so it bills nothing.
        """
        try:
            return await self._invoke(schema, system, user, repair=repair)
        except LLMError as exc:
            if self._rebuild_without_temperature is None or not _is_temperature_rejection(exc):
                raise
            log.info(
                "temperature_unsupported_retrying",
                model=str(self.spec),
                detail="model accepts only its default temperature",
            )
            self._model = self._rebuild_without_temperature()
            return await self._invoke(schema, system, user, repair=repair)

    async def _invoke(
        self, schema: type[SchemaT], system: str, user: str, *, repair: bool
    ) -> SchemaT:
        runnable = self._model.with_structured_output(
            schema, method="json_schema", include_raw=True
        )
        messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=user)]

        started = time.perf_counter()
        input_tokens = output_tokens = 0
        attempts = 0
        last_error = ""
        max_attempts = 1 + (self._repair_attempts if repair else 0)

        while attempts < max_attempts:
            attempts += 1
            try:
                # include_raw=True always yields the {raw, parsed, parsing_error}
                # envelope, but the return type is declared as the union of both
                # modes, so the narrowing has to be stated here.
                result = cast("dict[str, Any]", await runnable.ainvoke(messages))
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._finish(
                    schema.__name__,
                    started,
                    input_tokens,
                    output_tokens,
                    attempts,
                    ok=False,
                    error=last_error,
                )
                raise self._as_domain_error(exc) from exc

            raw = result.get("raw")
            usage = getattr(raw, "usage_metadata", None) or {}
            input_tokens += int(usage.get("input_tokens", 0) or 0)
            output_tokens += int(usage.get("output_tokens", 0) or 0)

            parsed = result.get("parsed")
            parse_error = result.get("parsing_error")
            if parsed is not None and parse_error is None:
                self._finish(
                    schema.__name__, started, input_tokens, output_tokens, attempts, ok=True
                )
                return cast("SchemaT", parsed)

            last_error = str(parse_error) if parse_error else "model returned no parsable object"
            log.warning(
                "structured_output_repair",
                role=self.role.value,
                model=str(self.spec),
                schema=schema.__name__,
                attempt=attempts,
                error=last_error[:200],
            )
            if attempts < max_attempts:
                if raw is not None:
                    messages.append(raw)
                messages.append(HumanMessage(content=_REPAIR_TEMPLATE.format(error=last_error)))

        self._finish(
            schema.__name__,
            started,
            input_tokens,
            output_tokens,
            attempts,
            ok=False,
            error=last_error,
        )
        raise StructuredOutputError(self.spec, schema.__name__, attempts, last_error)

    def _finish(
        self,
        schema_name: str,
        started: float,
        input_tokens: int,
        output_tokens: int,
        attempts: int,
        *,
        ok: bool,
        error: str | None = None,
    ) -> None:
        self._tracker.record(
            LLMCallRecord(
                role=self.role,
                provider=self.spec.provider,
                model=self.spec.model,
                schema=schema_name,
                latency_s=round(time.perf_counter() - started, 3),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                attempts=attempts,
                ok=ok,
                error=error,
                fell_back=self._fell_back,
            )
        )

    def _as_domain_error(self, exc: Exception) -> Exception:
        """Turn a transport-level failure into an actionable domain error.

        Everything a caller might reasonably catch must end up as an
        ``LLMError``. A raw ``httpx.ReadTimeout`` leaking out of here escaped a
        parallel worker's `except LLMError` and took down the whole super-step,
        which is precisely the failure the worker design exists to prevent.
        """
        if isinstance(exc, httpx.TimeoutException):
            hint = (
                "Raise LLM_TIMEOUT_SECONDS, lower MAX_PARALLEL_LOCAL_LLM_CALLS, "
                "or use a smaller local model"
                if self.spec.provider is Provider.OLLAMA
                else "Raise LLM_TIMEOUT_SECONDS"
            )
            return ModelTimeoutError(f"{self.spec} timed out. {hint}")
        text = str(exc).lower()
        # Checked before the connection heuristics: a 429 body can mention
        # "connection" and would otherwise be misreported as unreachable.
        if _is_rate_limit(exc):
            return ProviderRateLimited(
                f"{self.spec} rate limited or out of quota: {str(exc)[:300]}"
            )
        if isinstance(exc, httpx.ConnectError) or "connection" in text or "refused" in text:
            hint = (
                "Start the local server with `ollama serve`"
                if self.spec.provider is Provider.OLLAMA
                else "Check network access and OPENAI_API_KEY"
            )
            return ModelUnavailableError(self.spec, "connection failed", hint)
        if isinstance(exc, httpx.HTTPError):
            return ModelUnavailableError(self.spec, f"transport error: {exc}")
        if "invalid_request_error" in text or "unsupported value" in text:
            return ProviderRejectedRequest(f"{self.spec} rejected the request: {exc}")
        return exc


class ModelRouter:
    """Resolves roles to models and owns provider health.

    Model instances are cached per spec: each ``ChatOpenAI`` holds an HTTP
    client, and rebuilding one per node call would discard connection pooling
    for no reason.
    """

    def __init__(self, settings: Settings, tracker: UsageTracker | None = None) -> None:
        self.settings = settings
        self.tracker = tracker or UsageTracker(settings.max_llm_calls, settings.cloud_budget)
        self._assignments = settings.resolve_models()
        self._clients: dict[str, BaseChatModel] = {}
        self._fallbacks: dict[ModelRole, ModelSpec] = {}
        self._preflighted = False
        # Created lazily so it binds to the running loop rather than whichever
        # loop happened to construct the router.
        self._local_gate: asyncio.Semaphore | None = None
        # Models discovered at runtime to reject an explicit temperature.
        # Remembered so the retry happens once per model, not once per call.
        self._no_temperature: set[str] = set()

    # -- introspection ----------------------------------------------------

    def assignments(self) -> dict[ModelRole, ModelSpec]:
        """Effective role → model map, including any fallbacks applied."""
        return {role: self._fallbacks.get(role, spec) for role, spec in self._assignments.items()}

    def describe(self) -> dict[str, str]:
        return {role.value: str(spec) for role, spec in self.assignments().items()}

    # -- health -----------------------------------------------------------

    async def preflight(self) -> list[str]:
        """Check every configured provider before the graph starts.

        Failing here costs a second; failing lazily costs the user a partial
        run and, in hybrid mode, real money already spent on earlier nodes.
        Returns human-readable warnings for degradations that were tolerated.
        """
        warnings: list[str] = []
        ollama_roles = [r for r, s in self._assignments.items() if s.provider is Provider.OLLAMA]
        if ollama_roles:
            available, detail = await self._ollama_status()
            for role in ollama_roles:
                spec = self._assignments[role]
                installed = available is not None and self._model_installed(spec.model, available)
                if installed:
                    continue
                reason = detail if available is None else f"model {spec.model!r} is not pulled"
                hint = (
                    "Start it with `ollama serve`"
                    if available is None
                    else f"Install it with `ollama pull {spec.model}`"
                )
                if self.settings.allow_cloud_fallback and self.settings.openai_api_key:
                    fallback = self._cloud_equivalent(role)
                    self._fallbacks[role] = fallback
                    warnings.append(
                        f"{role.value}: {reason}; falling back to {fallback} "
                        "(ALLOW_CLOUD_FALLBACK=true)"
                    )
                    log.warning(
                        "model_fallback", role=role.value, from_=str(spec), to=str(fallback)
                    )
                else:
                    raise ModelUnavailableError(spec, reason, hint)
        self._preflighted = True
        return warnings

    async def _ollama_status(self) -> tuple[list[str] | None, str]:
        """Return installed model names, or ``None`` with a reason."""
        url = self.settings.ollama_base_url.rstrip("/") + "/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            return None, f"cannot reach Ollama at {self.settings.ollama_base_url} ({exc})"
        except ValueError:
            return None, "Ollama returned a non-JSON response"
        names = [m.get("name", "") for m in payload.get("models", []) if isinstance(m, dict)]
        return names, ""

    @staticmethod
    def _model_installed(wanted: str, available: list[str]) -> bool:
        """Ollama treats a bare name as implicitly ``:latest``."""
        return wanted in available or f"{wanted}:latest" in available

    def _cloud_equivalent(self, role: ModelRole) -> ModelSpec:
        model = (
            self.settings.openai_fast_model
            if role is ModelRole.RESEARCHER
            else self.settings.openai_model
        )
        return ModelSpec(provider=Provider.OPENAI, model=model)

    # -- resolution -------------------------------------------------------

    def get(self, role: ModelRole) -> RoleModel:
        spec = self._fallbacks.get(role) or self._assignments[role]
        output_cap = self.settings.cloud_budget.output_cap_for(role.value)
        return RoleModel(
            role=role,
            spec=spec,
            model=self._client_for(spec, output_cap),
            tracker=self.tracker,
            fell_back=role in self._fallbacks,
            repair_attempts=1,
            gate=self._gate_for(spec),
            rebuild_without_temperature=(
                partial(self._without_temperature, spec, output_cap)
                if spec.provider is Provider.OPENAI
                else None
            ),
        )

    def _without_temperature(self, spec: ModelSpec, output_cap: int | None) -> BaseChatModel:
        """Rebuild a client for a model that rejects an explicit temperature."""
        self._no_temperature.add(spec.model)
        key = f"{spec}#{output_cap}"
        client = self._build(spec, output_cap, with_temperature=False)
        self._clients[key] = client
        return client

    def _gate_for(self, spec: ModelSpec) -> asyncio.Semaphore | None:
        """Concurrency limit for local models.

        Ollama serves a model largely serially: concurrent requests queue
        rather than overlap, so fanning eight extraction calls at one local
        model buys no speedup and pushes the later ones past their read
        timeout. Cloud providers handle concurrency server-side and are gated
        by the stage semaphores instead.
        """
        if spec.provider is not Provider.OLLAMA:
            return None
        if self._local_gate is None:
            self._local_gate = asyncio.Semaphore(self.settings.max_parallel_local_llm_calls)
        return self._local_gate

    def _client_for(self, spec: ModelSpec, output_cap: int | None) -> BaseChatModel:
        # Keyed by output cap as well as spec: the cap is baked into the
        # client, so two roles sharing a model still need separate instances.
        key = f"{spec}#{output_cap}"
        cached = self._clients.get(key)
        if cached is not None:
            return cached
        client = self._build(spec, output_cap)
        self._clients[key] = client
        return client

    def _build(
        self,
        spec: ModelSpec,
        output_cap: int | None = None,
        *,
        with_temperature: bool = True,
    ) -> BaseChatModel:
        settings = self.settings
        if spec.provider is Provider.OPENAI:
            from langchain_openai import ChatOpenAI

            if not settings.openai_api_key:
                raise ModelUnavailableError(
                    spec, "OPENAI_API_KEY is not set", "Add it to .env or use LLM_MODE=local"
                )
            extra: dict[str, Any] = {}
            if with_temperature and spec.model not in self._no_temperature:
                extra["temperature"] = settings.llm_temperature
            return ChatOpenAI(
                model=spec.model,
                api_key=settings.openai_api_key,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
                **extra,
                # Hard per-call output ceiling. A runaway generation is
                # otherwise billed in full before anything notices.
                max_completion_tokens=output_cap,
            )

        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=spec.model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            num_ctx=settings.local_num_ctx,
            num_predict=output_cap or settings.local_num_ctx,
            # Chain-of-thought text before the JSON object is a common cause of
            # structured-output failures on small local models, and the thinking
            # tokens are pure latency for the mechanical roles that run locally.
            reasoning=False,
            client_kwargs={"timeout": settings.llm_timeout_seconds},
        )


def build_router(settings: Settings) -> ModelRouter:
    router = ModelRouter(settings)
    log.info("model_routing_resolved", mode=settings.llm_mode.value, assignments=router.describe())
    return router


def mode_summary(settings: Settings) -> str:
    router = ModelRouter(settings)
    parts = [f"{role}={spec}" for role, spec in sorted(router.describe().items())]
    prefix = {
        LLMMode.CLOUD: "cloud",
        LLMMode.LOCAL: "local",
        LLMMode.HYBRID: "hybrid",
    }[settings.llm_mode]
    return f"{prefix}: " + ", ".join(parts)
