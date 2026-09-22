"""Role-based model routing.

Application code never constructs a provider. It asks for a *role* and the
router decides what backs it::

    report = await router.get(ModelRole.SYNTHESIZER).structured(ReportOut, ...)

That indirection is what makes cloud/local/hybrid a configuration change rather
than a code change, and it gives the engine exactly one place to meter calls,
retry malformed output and handle a provider being down.
"""

from __future__ import annotations

import time
from typing import Any, TypeVar, cast

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agentic_research.config import LLMMode, ModelRole, ModelSpec, Provider, Settings
from agentic_research.llm.base import (
    LLMCallRecord,
    ModelUnavailableError,
    StructuredOutputError,
    UsageTracker,
)
from agentic_research.observability import get_logger

log = get_logger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

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
    ) -> None:
        self.role = role
        self.spec = spec
        self._model = model
        self._tracker = tracker
        self._fell_back = fell_back
        self._repair_attempts = repair_attempts

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
        await self._tracker.reserve()
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
        """Turn a transport-level failure into something actionable."""
        text = str(exc).lower()
        if isinstance(exc, httpx.ConnectError) or "connection" in text or "refused" in text:
            hint = (
                "Start the local server with `ollama serve`"
                if self.spec.provider is Provider.OLLAMA
                else "Check network access and OPENAI_API_KEY"
            )
            return ModelUnavailableError(self.spec, "connection failed", hint)
        return exc


class ModelRouter:
    """Resolves roles to models and owns provider health.

    Model instances are cached per spec: each ``ChatOpenAI`` holds an HTTP
    client, and rebuilding one per node call would discard connection pooling
    for no reason.
    """

    def __init__(self, settings: Settings, tracker: UsageTracker | None = None) -> None:
        self.settings = settings
        self.tracker = tracker or UsageTracker(settings.max_llm_calls)
        self._assignments = settings.resolve_models()
        self._clients: dict[str, BaseChatModel] = {}
        self._fallbacks: dict[ModelRole, ModelSpec] = {}
        self._preflighted = False

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
        return RoleModel(
            role=role,
            spec=spec,
            model=self._client_for(spec),
            tracker=self.tracker,
            fell_back=role in self._fallbacks,
            repair_attempts=1,
        )

    def _client_for(self, spec: ModelSpec) -> BaseChatModel:
        cached = self._clients.get(str(spec))
        if cached is not None:
            return cached
        client = self._build(spec)
        self._clients[str(spec)] = client
        return client

    def _build(self, spec: ModelSpec) -> BaseChatModel:
        settings = self.settings
        if spec.provider is Provider.OPENAI:
            from langchain_openai import ChatOpenAI

            if not settings.openai_api_key:
                raise ModelUnavailableError(
                    spec, "OPENAI_API_KEY is not set", "Add it to .env or use LLM_MODE=local"
                )
            return ChatOpenAI(
                model=spec.model,
                api_key=settings.openai_api_key,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )

        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=spec.model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            num_ctx=settings.local_num_ctx,
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
