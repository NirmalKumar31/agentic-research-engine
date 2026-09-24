"""Pre-flight for the live deployment, with no provider and no credentials.

Everything the live Blueprint turns on, asserted before it is turned on.
The first real live run surfaced four separate defects that a deploy-then-
look loop would have surfaced again, so the matrix below covers the whole
request surface: configuration, every clamped ceiling, and each HTTP path
including the failure ones.

A fake research stream stands in for the engine, so nothing here needs a
key, reaches a network, or costs anything.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from agentic_research.config import Settings
from agentic_research.llm.base import ProviderRateLimited
from agentic_research.web import api as api_module
from agentic_research.web import recordings
from agentic_research.web.api import create_app
from agentic_research.web.recordings import RECORDING_SCHEMA_VERSION

REPO = Path(__file__).resolve().parents[2]
LIVE_BLUEPRINT = "deploy/render-live.yaml"


def blueprint_env(name: str = LIVE_BLUEPRINT) -> dict[str, str]:
    doc = yaml.safe_load((REPO / name).read_text(encoding="utf-8"))
    return {e["key"]: e.get("value", "<prompted>") for e in doc["services"][0]["envVars"]}


def live_settings(**over: Any) -> Settings:
    """The live posture, but on a local model so nothing can be billed."""
    base: dict[str, Any] = {
        "llm_mode": "local",
        "demo_mode": True,
        "live_research_enabled": True,
        "tavily_api_key": "tvly-test-key",
        "_env_file": None,
    }
    base.update(over)
    return Settings(**base)


RECORDING = {
    "recording_schema_version": RECORDING_SCHEMA_VERSION,
    "meta": {"id": "demo", "label": "Demo", "question": "q", "order": 1},
    "result": {
        "run_id": "r",
        "plan": {"sub_questions": [{"id": "SQ1", "text": "A question?", "is_followup": False}]},
        "report": None,
        "evidence": [],
        "sources": [],
        "verification": None,
        "metrics": {"duration_s": 1.0},
        "markdown": "",
    },
    "trace": [{"event": "plan_generated", "count": 1}],
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


def fake_stream(*events: dict[str, Any], raises: Exception | None = None):
    """Stand in for stream_research without touching a provider."""

    async def _stream(query: str, settings: Any, run_id: str = "") -> AsyncIterator[dict[str, Any]]:
        for event in events:
            yield event
        if raises is not None:
            raise raises

    return _stream


# --- A. startup and configuration ------------------------------------------


class TestStartupConfiguration:
    def test_the_blueprint_turns_live_research_on_in_cloud_mode(self) -> None:
        env = blueprint_env()
        assert env["LIVE_RESEARCH_ENABLED"] == "true"
        assert env["LLM_MODE"] == "cloud"
        assert env["DEMO_MODE"] == "true"

    def test_only_luna_is_reachable_and_fallback_is_off(self) -> None:
        env = blueprint_env()
        assert env["OPENAI_MODEL"] == "gpt-6-luna"
        assert env["OPENAI_FAST_MODEL"] == "gpt-6-luna"
        assert env["ALLOW_CLOUD_FALLBACK"] == "false"
        configured = {v.lower() for k, v in env.items() if k.startswith("OPENAI_")}
        assert not any("sol" in v or "astra" in v for v in configured)

    def test_one_run_at_a_time(self) -> None:
        assert blueprint_env()["DEMO_MAX_CONCURRENT_RUNS"] == "1"

    def test_secrets_are_prompted_never_committed(self) -> None:
        env = blueprint_env()
        assert env["OPENAI_API_KEY"] == "<prompted>"
        assert env["TAVILY_API_KEY"] == "<prompted>"

    def test_the_service_reports_itself_as_live(self) -> None:
        with TestClient(create_app(live_settings())) as client:
            body = client.get("/api/config").json()
        assert body["service_mode"] == "live"
        assert body["live_research_enabled"] is True

    def test_recordings_still_load_in_the_live_service(self) -> None:
        with TestClient(create_app(live_settings())) as client:
            assert client.get("/api/examples").json()["examples"]

    def test_the_frontend_is_found_from_an_installed_layout(self) -> None:
        """The defect that made the first deploy serve JSON at /."""
        from agentic_research.web.api import _find_frontend_dist

        assert isinstance(_find_frontend_dist(), Path)


# --- B. limits --------------------------------------------------------------


class TestEveryCeilingIsServerEnforced:
    @pytest.fixture
    def clamped(self) -> Settings:
        from agentic_research.web.limits import apply_demo_limits, limits_from_settings

        # An environment that asks for far more than the demo allows.
        greedy = live_settings(
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
            demo_max_runtime_seconds=99_999.0,
            demo_runs_per_hour=1_000,
            demo_max_concurrent_runs=500,
        )
        return apply_demo_limits(greedy, limits_from_settings(greedy))

    def test_paid_dimensions(self, clamped: Settings) -> None:
        assert clamped.max_provider_requests == 30
        assert clamped.max_cloud_calls == 20
        assert clamped.max_cloud_input_tokens == 120_000
        assert clamped.max_cloud_output_tokens == 20_000
        assert clamped.max_cloud_cost_usd == 0.05
        assert clamped.max_search_credits == 8.0

    def test_run_shape(self, clamped: Settings) -> None:
        assert clamped.max_research_rounds == 1
        assert clamped.max_sources <= 6
        assert clamped.max_search_queries <= 6
        assert clamped.max_llm_calls <= 20

    def test_traffic_shaping(self) -> None:
        from agentic_research.web.limits import DemoLimits, limits_from_settings

        limits = limits_from_settings(
            live_settings(
                demo_max_runtime_seconds=99_999.0,
                demo_runs_per_hour=1_000,
                demo_max_concurrent_runs=500,
            )
        )
        ceiling = DemoLimits()
        assert limits.max_runtime_seconds == 240.0 == ceiling.max_runtime_seconds
        assert limits.runs_per_ip_per_hour == ceiling.runs_per_ip_per_hour
        assert limits.max_concurrent_runs == ceiling.max_concurrent_runs

    def test_a_hosted_run_never_falls_back_to_a_paid_provider(self, clamped: Settings) -> None:
        assert clamped.allow_cloud_fallback is False

    def test_nothing_is_persisted(self, clamped: Settings) -> None:
        assert clamped.persist_runs is False
        assert clamped.checkpoint_backend == "memory"


# --- C. HTTP surface --------------------------------------------------------


class TestTheHttpSurface:
    @pytest.fixture
    def client(self) -> Any:
        with TestClient(create_app(live_settings())) as c:
            yield c

    def test_health(self, client: Any) -> None:
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert "capacity" in body

    def test_config(self, client: Any) -> None:
        assert client.get("/api/config").status_code == 200

    def test_examples_collection_and_item(self, client: Any) -> None:
        assert client.get("/api/examples").status_code == 200
        assert client.get("/api/examples/demo").status_code == 200

    def test_an_unknown_example_is_a_json_404(self, client: Any) -> None:
        response = client.get("/api/examples/nope")
        assert response.status_code == 404
        assert response.json()["error"]

    def test_replay_streams(self, client: Any) -> None:
        with client.stream("GET", "/api/examples/demo/stream") as r:
            body = "".join(r.iter_text())
        assert "event: result" in body and "event: done" in body

    def test_an_unknown_api_path_is_json_not_the_app_shell(self, client: Any) -> None:
        response = client.get("/api/does-not-exist")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")

    def test_a_live_run_streams_to_completion(
        self, monkeypatch: pytest.MonkeyPatch, client: Any
    ) -> None:
        monkeypatch.setattr(
            api_module,
            "stream_research",
            fake_stream({"event": "planning"}, {"event": "completed", "stop_reason": "done"}),
        )
        with client.stream("POST", "/api/research", json={"query": "a real question"}) as r:
            body = "".join(r.iter_text())
        assert "event: progress" in body
        assert "event: done" in body

    @pytest.mark.parametrize("query", ["", "hi"])
    def test_an_invalid_query_is_refused(self, client: Any, query: str) -> None:
        assert client.post("/api/research", json={"query": query}).status_code in (400, 422)

    def test_an_oversized_query_is_refused(self, client: Any) -> None:
        assert client.post("/api/research", json={"query": "x" * 5_000}).status_code in (400, 422)

    def test_a_provider_429_is_reported_as_capacity(
        self, monkeypatch: pytest.MonkeyPatch, client: Any
    ) -> None:
        monkeypatch.setattr(
            api_module, "stream_research", fake_stream(raises=ProviderRateLimited("gone"))
        )
        with client.stream("POST", "/api/research", json={"query": "a real question"}) as r:
            body = "".join(r.iter_text())
        assert "capacity_reached" in body

    def test_a_generic_provider_failure_leaks_nothing(
        self, monkeypatch: pytest.MonkeyPatch, client: Any
    ) -> None:
        monkeypatch.setattr(
            api_module,
            "stream_research",
            fake_stream(raises=RuntimeError("https://internal/secret")),
        )
        with client.stream("POST", "/api/research", json={"query": "a real question"}) as r:
            body = "".join(r.iter_text())
        assert "internal" not in body
        assert "The research run failed" in body

    def test_capacity_is_refused_with_429_when_the_quota_cannot_pay(self) -> None:
        settings = live_settings(demo_provider_requests_per_day=10)
        with TestClient(create_app(settings)) as client:
            response = client.post("/api/research", json={"query": "a real question"})
        assert response.status_code == 429

    def test_a_timeout_stops_the_run_and_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio

        async def slow(query: str, settings: Any, run_id: str = "") -> AsyncIterator[dict]:
            await asyncio.sleep(5)
            yield {"event": "planning"}

        monkeypatch.setattr(api_module, "stream_research", slow)
        with (
            TestClient(create_app(live_settings(demo_max_runtime_seconds=0.05))) as client,
            client.stream("POST", "/api/research", json={"query": "a real question"}) as r,
        ):
            body = "".join(r.iter_text())
        assert '"timeout": true' in body

    def test_replay_still_works_after_a_failed_live_run(
        self, monkeypatch: pytest.MonkeyPatch, client: Any
    ) -> None:
        """A broken live path must not take the recorded demos with it."""
        monkeypatch.setattr(api_module, "stream_research", fake_stream(raises=RuntimeError("boom")))
        with client.stream("POST", "/api/research", json={"query": "a real question"}) as r:
            "".join(r.iter_text())
        assert client.get("/api/examples/demo").status_code == 200

    def test_the_slot_is_free_again_after_a_failed_run(
        self, monkeypatch: pytest.MonkeyPatch, client: Any
    ) -> None:
        monkeypatch.setattr(api_module, "stream_research", fake_stream(raises=RuntimeError("boom")))
        with client.stream("POST", "/api/research", json={"query": "a real question"}) as r:
            "".join(r.iter_text())
        assert client.get("/api/health").json()["capacity"]["active_runs"] == 0
