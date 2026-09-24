"""The live deployment: its ceilings, and how it fails.

render-live.yaml turns on an endpoint that spends money, so the values in
it are asserted rather than trusted to review. A typo that widens a
ceiling is invisible in a diff and expensive in production.

The failure behaviour matters as much as the limits. A visitor who arrives
when the quota is gone should be told that, and should still be able to
explore the recorded runs -- the live path breaking must not take the
replay path with it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from agentic_research.config import Settings
from agentic_research.llm.base import ProviderRateLimited
from agentic_research.web import recordings
from agentic_research.web.api import create_app
from agentic_research.web.recordings import RECORDING_SCHEMA_VERSION

REPO = Path(__file__).resolve().parents[2]


def _blueprint(name: str) -> dict[str, str]:
    doc = yaml.safe_load((REPO / name).read_text(encoding="utf-8"))
    service = doc["services"][0]
    return {e["key"]: e.get("value", "<prompted>") for e in service["envVars"]}


class TestTheLiveBlueprint:
    """Values, not prose. Every one of these bounds real spend."""

    @pytest.fixture
    def env(self) -> dict[str, str]:
        return _blueprint("deploy/render-live.yaml")

    def test_secrets_are_prompted_never_committed(self, env: dict[str, str]) -> None:
        assert env["OPENAI_API_KEY"] == "<prompted>"
        assert env["TAVILY_API_KEY"] == "<prompted>"
        raw = (REPO / "deploy/render-live.yaml").read_text(encoding="utf-8")
        for marker in ("sk-", "tvly-"):
            assert marker not in raw, f"{marker} literal in a committed blueprint"

    def test_live_research_is_on_and_cloud_backed(self, env: dict[str, str]) -> None:
        """These two belong together: there is no Ollama on Render, so
        local plus live would fail preflight on the first request."""
        assert env["LIVE_RESEARCH_ENABLED"] == "true"
        assert env["LLM_MODE"] == "cloud"
        assert env["DEMO_MODE"] == "true", "client values must stay clamped"

    def test_only_the_cheap_model_is_reachable(self, env: dict[str, str]) -> None:
        assert env["OPENAI_MODEL"] == "gpt-6-luna"
        assert env["OPENAI_FAST_MODEL"] == "gpt-6-luna"
        # Checked against the configured values, not the file text: a
        # comment saying "Sol and Astra are never used" would otherwise
        # fail its own assertion.
        configured = {
            v.lower() for k, v in env.items() if k.startswith("OPENAI_") and v != "<prompted>"
        }
        assert not any("sol" in v or "astra" in v for v in configured), configured
        # A missing model must fail rather than escalate to a pricier one.
        assert env["ALLOW_CLOUD_FALLBACK"] == "false"

    def test_per_run_ceilings_match_the_agreed_values(self, env: dict[str, str]) -> None:
        assert env["MAX_CLOUD_CALLS"] == "20"
        assert env["MAX_CLOUD_COST_USD"] == "0.05"
        assert env["MAX_CLOUD_INPUT_TOKENS"] == "120000"
        assert env["MAX_CLOUD_OUTPUT_TOKENS"] == "20000"
        assert env["MAX_SEARCH_CREDITS"] == "8"
        assert env["MAX_PROVIDER_REQUESTS"] == "30"

    def test_the_run_stays_small(self, env: dict[str, str]) -> None:
        assert env["MAX_RESEARCH_ROUNDS"] == "1"
        assert env["MAX_SOURCES"] == "6"
        assert env["MAX_SOURCES_PER_ROUND"] == "6"
        assert env["MAX_SEARCH_QUERIES"] == "6"
        assert env["MAX_LLM_CALLS"] == "20"

    def test_traffic_is_shaped_for_one_small_instance(self, env: dict[str, str]) -> None:
        assert env["DEMO_MAX_CONCURRENT_RUNS"] == "1"
        assert env["DEMO_RUNS_PER_HOUR"] == "2"
        assert env["DEMO_PROVIDER_REQUESTS_PER_DAY"] == "50"
        assert env["DEMO_MAX_RUNTIME_SECONDS"] == "240"

    def test_nothing_is_persisted(self, env: dict[str, str]) -> None:
        assert env["PERSIST_RUNS"] == "false"
        assert env["CHECKPOINT_BACKEND"] == "memory"

    def test_the_daily_cap_derives_from_the_quota(self, env: dict[str, str]) -> None:
        """50 provider requests a day at 20 per run affords two runs. It is
        derived rather than written down, so raising the tier raises the
        cap without another edit."""
        from agentic_research.web.limits import runs_affordable

        affordable = runs_affordable(
            int(env["DEMO_PROVIDER_REQUESTS_PER_DAY"]), int(env["MAX_CLOUD_CALLS"])
        )
        assert affordable == 2

    def test_every_key_maps_to_a_real_setting(self, env: dict[str, str]) -> None:
        """A misspelled variable is silently ignored by pydantic-settings,
        so the ceiling it was meant to set simply would not exist."""
        known = {f.upper() for f in Settings.model_fields}
        unknown = [k for k in env if k not in known]
        assert unknown == [], f"these set nothing: {unknown}"


class TestTheReplayBlueprintStaysSafe:
    """Adding a live deployment must not weaken the safe one."""

    def test_it_declares_no_credentials_at_all(self) -> None:
        env = _blueprint("render.yaml")
        assert "OPENAI_API_KEY" not in env
        assert "TAVILY_API_KEY" not in env

    def test_live_research_is_off(self) -> None:
        assert _blueprint("render.yaml")["LIVE_RESEARCH_ENABLED"] == "false"


RECORDING = {
    "recording_schema_version": RECORDING_SCHEMA_VERSION,
    "meta": {"id": "demo", "label": "Demo", "question": "q", "order": 1},
    "result": {
        "run_id": "r",
        "plan": None,
        "report": None,
        "evidence": [],
        "sources": [],
        "verification": None,
        "metrics": {"duration_s": 1.0},
        "markdown": "",
    },
}


@pytest.fixture(autouse=True)
def _recordings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "rec"
    directory.mkdir()
    (directory / "demo.json").write_text(json.dumps(RECORDING), encoding="utf-8")
    monkeypatch.setattr(recordings, "RECORDINGS_DIR", directory)
    recordings._index.cache_clear()
    yield
    recordings._index.cache_clear()


def _live_settings(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_mode": "local",
        "demo_mode": True,
        "live_research_enabled": True,
        "tavily_api_key": "tvly-test-key",
        "_env_file": None,
    }
    base.update(over)
    return Settings(**base)


class TestLiveFailureDoesNotBreakTheRecordedDemos:
    """The whole point of keeping replay alongside live."""

    def _client(self, monkeypatch: pytest.MonkeyPatch, exc: Exception) -> TestClient:
        import agentic_research.web.api as api_module

        async def failing(query: str, settings: Any, run_id: str = "") -> Any:
            raise exc
            yield {}  # pragma: no cover - makes this an async generator

        monkeypatch.setattr(api_module, "stream_research", failing)
        return TestClient(create_app(_live_settings()))

    def test_a_provider_429_is_reported_as_capacity_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'Please try again' would be wrong advice: the retry fails the
        same way and spends another provider request doing it."""
        with (
            self._client(monkeypatch, ProviderRateLimited("quota exhausted")) as client,
            client.stream("POST", "/api/research", json={"query": "a real question"}) as r,
        ):
            body = "".join(r.iter_text())
        assert "capacity_reached" in body
        assert "quota" in body.lower()
        assert "try again" not in body.lower()

    def test_the_provider_message_is_never_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provider errors name models, limits and sometimes keys."""
        leak = "Rate limit reached for gpt-6-luna org-abc123 key sk-secret"
        with (
            self._client(monkeypatch, ProviderRateLimited(leak)) as client,
            client.stream("POST", "/api/research", json={"query": "a real question"}) as r,
        ):
            body = "".join(r.iter_text())
        assert "org-abc123" not in body
        assert "sk-secret" not in body
        assert "gpt-6-luna" not in body

    def test_recorded_runs_still_serve_after_a_live_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, ProviderRateLimited("gone")) as client:
            with client.stream("POST", "/api/research", json={"query": "a real question"}) as r:
                "".join(r.iter_text())
            assert client.get("/api/examples").json()["examples"], "replay broke with live"
            assert client.get("/api/examples/demo").json()["recorded"] is True
            assert client.get("/api/health").json()["status"] == "ok"

    def test_a_generic_failure_does_not_claim_capacity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with (
            self._client(monkeypatch, RuntimeError("boom")) as client,
            client.stream("POST", "/api/research", json={"query": "a real question"}) as r,
        ):
            body = "".join(r.iter_text())
        assert "capacity_reached" not in body
        assert "boom" not in body


class TestReplayNeedsNoProvider:
    def test_examples_work_with_no_credentials_configured(self) -> None:
        """If OpenAI and Tavily both vanished, the site still works."""
        settings = Settings(
            llm_mode="local", demo_mode=True, live_research_enabled=False, _env_file=None
        )
        with TestClient(create_app(settings)) as client:
            assert client.get("/api/examples").json()["examples"]
            assert client.get("/api/examples/demo").status_code == 200
            assert client.get("/api/config").json()["service_mode"] == "replay"


class TestClientCannotWidenTheLiveRun:
    def test_query_length_is_capped_at_300(self) -> None:
        from agentic_research.web.limits import DemoLimits, validate_query

        limits = DemoLimits()
        assert limits.max_query_chars == 300
        with pytest.raises(ValueError, match="too long"):
            validate_query("x" * 301, limits)


class TestDemoLimitsCannotBeWidened:
    """A public live endpoint spends money, so the ceilings have to hold
    against both a hostile client and a misconfigured environment.

    Client input may narrow a run. Nothing may widen one.
    """

    @staticmethod
    def _generous() -> Settings:
        """An environment configured far above the demo ceilings."""
        return Settings(
            llm_mode="local",
            demo_mode=True,
            live_research_enabled=True,
            tavily_api_key="tvly-test-key",
            max_research_rounds=9,
            max_sources=99,
            max_sources_per_round=99,
            max_search_queries=99,
            max_llm_calls=999,
            # Kept at the blueprint value: a larger per-run request ceiling
            # means the daily quota affords zero runs, which is a separate
            # (and correct) refusal tested below.
            max_cloud_calls=20,
            max_cloud_cost_usd=100.0,
            max_search_credits=999.0,
            _env_file=None,
        )

    def test_environment_values_are_clamped_to_the_demo_ceilings(self) -> None:
        from agentic_research.web.limits import apply_demo_limits, limits_from_settings

        settings = self._generous()
        limits = limits_from_settings(settings)
        clamped = apply_demo_limits(settings, limits)

        assert clamped.max_research_rounds <= limits.max_rounds
        assert clamped.max_sources <= limits.max_sources
        assert clamped.max_sources_per_round <= limits.max_sources_per_round
        assert clamped.max_search_queries <= limits.max_search_queries
        assert clamped.max_llm_calls <= limits.max_llm_calls
        assert clamped.max_cloud_cost_usd <= limits.max_cloud_cost_usd
        assert clamped.max_search_credits <= limits.max_search_credits

    def test_clamping_never_raises_a_value_that_was_already_lower(self) -> None:
        """A locally tighter setting must survive: the clamp is a ceiling,
        not an assignment."""
        from agentic_research.web.limits import apply_demo_limits, limits_from_settings

        settings = Settings(
            llm_mode="local",
            demo_mode=True,
            live_research_enabled=True,
            max_sources=2,
            max_sources_per_round=2,
            max_llm_calls=5,
            max_cloud_cost_usd=0.01,
            _env_file=None,
        )
        clamped = apply_demo_limits(settings, limits_from_settings(settings))
        assert clamped.max_sources == 2
        assert clamped.max_llm_calls == 5
        assert clamped.max_cloud_cost_usd == 0.01

    def test_cloud_fallback_and_persistence_are_forced_off(self) -> None:
        from agentic_research.web.limits import apply_demo_limits, limits_from_settings

        settings = self._generous().model_copy(
            update={"allow_cloud_fallback": True, "persist_runs": True}
        )
        clamped = apply_demo_limits(settings, limits_from_settings(settings))
        assert clamped.allow_cloud_fallback is False
        assert clamped.persist_runs is False
        assert clamped.checkpoint_backend == "memory"

    @pytest.mark.parametrize(
        "payload",
        [
            {"query": "a genuine research question", "max_rounds": 5},
            {"query": "a genuine research question", "max_sources": 40},
            {"query": "a genuine research question", "max_rounds": 5, "max_sources": 40},
        ],
    )
    def test_a_client_cannot_widen_the_run_it_requests(
        self, monkeypatch: pytest.MonkeyPatch, payload: dict
    ) -> None:
        """The request model caps these, and the server clamps again after."""
        import agentic_research.web.api as api_module
        from agentic_research.web.limits import DemoLimits

        seen: list[Settings] = []

        async def capture(query: str, settings: Settings, run_id: str = "") -> Any:
            seen.append(settings)
            if False:  # pragma: no cover - makes this an async generator
                yield {}

        monkeypatch.setattr(api_module, "stream_research", capture)
        with TestClient(create_app(self._generous())) as client:
            client.post("/api/research", json=payload)

        assert seen, "the run never started"
        used = seen[0]
        limits = DemoLimits()
        assert used.max_research_rounds <= limits.max_rounds
        assert used.max_sources <= limits.max_sources

    def test_a_client_cannot_reach_the_provider_request_ceiling(self) -> None:
        """Nothing in the request schema names these, and that is the
        point: the fields a client may send are an allowlist."""
        from agentic_research.web.api import ResearchRequest

        assert set(ResearchRequest.model_fields) == {"query", "max_rounds", "max_sources"}

    def test_a_run_the_quota_cannot_pay_for_is_refused_up_front(self) -> None:
        """If one run may emit more provider requests than the daily quota
        allows, the honest answer is to refuse before starting rather than
        to 429 halfway through.

        Reached here by shrinking the quota rather than by inflating
        ``max_cloud_calls``: that field is now clamped to the demo ceiling,
        so a configured 999 genuinely becomes 20 and the quota really can
        afford a run. The refusal must still fire when the quota itself is
        too small to cover one clamped run.
        """
        from agentic_research.web.limits import limits_from_settings

        settings = self._generous().model_copy(update={"demo_provider_requests_per_day": 10})
        assert limits_from_settings(settings).global_runs_per_day == 0

        with TestClient(create_app(settings)) as client:
            response = client.post("/api/research", json={"query": "a genuine research question"})
        assert response.status_code == 429
        assert "cannot cover" in response.json()["error"]

    def test_query_length_is_bounded_on_both_sides(self) -> None:
        from agentic_research.web.limits import DemoLimits, validate_query

        limits = DemoLimits()
        assert limits.max_query_chars == 300
        with pytest.raises(ValueError, match="too long"):
            validate_query("x" * (limits.max_query_chars + 1), limits)
        with pytest.raises(ValueError, match="fuller question"):
            validate_query("hi", limits)


class TestEveryPaidDimensionIsClamped:
    """The ceilings that cost money, asserted individually.

    An earlier version clamped rounds, sources, queries, logical calls and
    cost, but left the provider-request and cloud-token budgets at their
    local-development defaults. The engine refuses a run whose reservation
    breaches *any* ceiling, so an unclamped request or token budget is a
    way to spend past the intended envelope while the dollar figure still
    reads correctly.
    """

    @staticmethod
    def _hostile_environment() -> Settings:
        """Every paid dimension configured absurdly high."""
        return Settings(
            llm_mode="local",
            demo_mode=True,
            live_research_enabled=True,
            tavily_api_key="tvly-test-key",
            max_cloud_calls=999_999,
            max_cloud_input_tokens=99_999_999,
            max_cloud_output_tokens=99_999_999,
            max_cloud_cost_usd=1_000.0,
            max_search_credits=9_999.0,
            max_provider_requests=100_000,
            max_research_rounds=9,
            max_sources=99,
            max_sources_per_round=99,
            max_search_queries=99,
            max_llm_calls=999,
            _env_file=None,
        )

    def test_the_paid_dimensions_are_all_clamped(self) -> None:
        from agentic_research.web.limits import apply_demo_limits, limits_from_settings

        settings = self._hostile_environment()
        limits = limits_from_settings(settings)
        clamped = apply_demo_limits(settings, limits)

        assert clamped.max_cloud_calls == 20
        assert clamped.max_cloud_input_tokens == 120_000
        assert clamped.max_cloud_output_tokens == 20_000
        assert clamped.max_cloud_cost_usd == 0.05
        assert clamped.max_search_credits == 8.0
        assert clamped.max_provider_requests == 30

    def test_the_operator_tunable_ceilings_cannot_be_raised(self) -> None:
        """Runtime, concurrency and per-IP rate are read from configuration,
        so without a clamp the dataclass ceiling is decorative."""
        from agentic_research.web.limits import DemoLimits, limits_from_settings

        settings = self._hostile_environment().model_copy(
            update={
                "demo_max_runtime_seconds": 99_999.0,
                "demo_runs_per_hour": 1_000,
                "demo_max_concurrent_runs": 500,
                "demo_provider_requests_per_day": 100_000,
            }
        )
        limits = limits_from_settings(settings)
        ceiling = DemoLimits()

        assert limits.max_runtime_seconds == ceiling.max_runtime_seconds
        assert limits.runs_per_ip_per_hour == ceiling.runs_per_ip_per_hour
        assert limits.max_concurrent_runs == ceiling.max_concurrent_runs
        # demo_provider_requests_per_day is intentionally absent: it states
        # what the provider account allows, not what this demo may spend.

    def test_a_lower_operator_value_still_wins(self) -> None:
        """The clamp is a ceiling, not an assignment."""
        from agentic_research.web.limits import apply_demo_limits, limits_from_settings

        settings = Settings(
            llm_mode="local",
            demo_mode=True,
            live_research_enabled=True,
            tavily_api_key="tvly-test-key",
            max_cloud_calls=5,
            max_cloud_input_tokens=1_000,
            max_cloud_output_tokens=500,
            max_cloud_cost_usd=0.01,
            max_search_credits=2.0,
            max_provider_requests=10,
            demo_max_runtime_seconds=60.0,
            demo_runs_per_hour=1,
            demo_max_concurrent_runs=1,
            _env_file=None,
        )
        limits = limits_from_settings(settings)
        clamped = apply_demo_limits(settings, limits)

        assert clamped.max_cloud_calls == 5
        assert clamped.max_cloud_input_tokens == 1_000
        assert clamped.max_cloud_output_tokens == 500
        assert clamped.max_cloud_cost_usd == 0.01
        assert clamped.max_search_credits == 2.0
        assert clamped.max_provider_requests == 10
        assert limits.max_runtime_seconds == 60.0
        assert limits.runs_per_ip_per_hour == 1
        assert limits.max_concurrent_runs == 1

    def test_zero_is_read_as_unlimited_and_does_not_fail_open(self) -> None:
        """The engine reads 0 as "no ceiling on this dimension".

        A plain ``min`` would therefore turn ``MAX_CLOUD_CALLS=0`` into a
        clamped 0 while actually disabling the limit. The demo maximum has
        to win instead.
        """
        from agentic_research.web.limits import apply_demo_limits, limits_from_settings

        settings = Settings(
            llm_mode="local",
            demo_mode=True,
            live_research_enabled=True,
            tavily_api_key="tvly-test-key",
            max_cloud_calls=0,
            max_cloud_input_tokens=0,
            max_cloud_output_tokens=0,
            max_cloud_cost_usd=0.0,
            max_search_credits=0.0,
            max_provider_requests=0,
            _env_file=None,
        )
        clamped = apply_demo_limits(settings, limits_from_settings(settings))

        assert clamped.max_cloud_calls == 20
        assert clamped.max_cloud_input_tokens == 120_000
        assert clamped.max_cloud_output_tokens == 20_000
        assert clamped.max_cloud_cost_usd == 0.05
        assert clamped.max_search_credits == 8.0
        assert clamped.max_provider_requests == 30

    def test_the_request_ceiling_covers_one_intended_demo_run(self) -> None:
        """Sized deliberately, not guessed: the clamped cloud-request and
        search-credit ceilings must fit inside the total request ceiling,
        or a compliant run would be refused by its own budget."""
        from agentic_research.web.limits import DemoLimits

        ceiling = DemoLimits()
        assert ceiling.max_cloud_calls + ceiling.max_search_credits <= (
            ceiling.max_provider_requests
        )
