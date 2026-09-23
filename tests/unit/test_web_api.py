"""Hosted demo API: limits, secret non-exposure, SSE and disconnect handling.

A public URL backed by a real API key is a spend endpoint for anyone who
finds it, so most of what matters here is what the server refuses to do.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentic_research.config import Settings
from agentic_research.metrics import RunMetrics
from agentic_research.models import (
    Claim,
    ClaimKind,
    DiscoveryRef,
    EvidenceItem,
    QuoteMatch,
    ResearchReport,
    SourceDocument,
)
from agentic_research.runner import RunResult
from agentic_research.web.api import create_app
from agentic_research.web.limits import DemoLimits


def demo_settings(**overrides: Any) -> Settings:
    base = {
        "llm_mode": "local",
        "demo_mode": True,
        "tavily_api_key": "tvly-secret-value-for-testing-only",
        "max_research_rounds": 9,
        "max_sources": 99,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def sample_result(run_id: str = "r1") -> RunResult:
    source = SourceDocument(
        id="S1",
        url="https://x.org/a",
        canonical_url="https://x.org/a",
        title="Paper",
        domain="x.org",
        text="full body text that must not be exposed",
        content_hash="h",
        discovered_by=[DiscoveryRef(query_id="Q1", sub_question_id="SQ1")],
    )
    item = EvidenceItem(
        id="S1-e1",
        source_id="S1",
        sub_question_id="SQ1",
        claim="c",
        quote="a verbatim quote",
        quote_match=QuoteMatch.EXACT_NORMALIZED,
        relevance=0.9,
        discovery=DiscoveryRef(query_id="Q1", sub_question_id="SQ1"),
    )
    report = ResearchReport(
        title="T",
        summary_claims=[Claim(text="Summary point", evidence_ids=["S1-e1"], citation_ids=["S1"])],
        key_findings=[
            Claim(
                text="Finding",
                evidence_ids=["S1-e1"],
                citation_ids=["S1"],
                kind=ClaimKind.SYNTHESIS,
            )
        ],
    )
    return RunResult(
        run_id=run_id,
        markdown="# T\n\nBody.\n",
        metrics=RunMetrics(run_id=run_id, query="q", mode="local"),
        state={
            "report": report,
            "evidence": [item],
            "sources": [source],
            "verification": {"issues": []},
            "plan": None,
        },
    )


@pytest.fixture
def patched_stream(monkeypatch: pytest.MonkeyPatch):
    def install(events: list[dict[str, Any]] | None = None) -> None:
        async def fake_stream(query, settings, **kwargs):
            for event in events or [
                {"event": "plan_generated", "count": 2, "questions": ["a", "b"]},
                {
                    "event": "coverage_evaluated",
                    "ratio": 1.0,
                    "sufficient": True,
                    "covered": 2,
                    "weak": 0,
                    "missing": 0,
                    "round": 1,
                },
            ]:
                yield event
            yield {"event": "result", "result": sample_result()}

        monkeypatch.setattr("agentic_research.web.api.stream_research", fake_stream)

    return install


def read_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.strip().split("\n\n"):
        name = payload = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
        if name is not None:
            events.append((name, payload or {}))
    return events


class TestHealthAndConfig:
    def test_health_is_cheap_and_credential_free(self) -> None:
        with TestClient(create_app(demo_settings())) as client:
            response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["demo_mode"] is True
        assert "tvly" not in response.text

    def test_config_exposes_limits_but_no_secrets(self) -> None:
        with TestClient(create_app(demo_settings())) as client:
            response = client.get("/api/config")
        body = response.json()
        assert body["max_rounds"] == DemoLimits().max_rounds
        for forbidden in ("api_key", "tavily", "openai_api_key", "tvly"):
            assert forbidden not in response.text.lower()

    def test_interactive_docs_are_disabled_in_demo_mode(self) -> None:
        """The Swagger UI invites poking at an endpoint that spends money.

        When the built frontend is present the SPA catch-all answers /docs
        with the app shell, so the assertion is on the absence of the docs
        UI rather than on a 404.
        """
        with TestClient(create_app(demo_settings())) as client:
            response = client.get("/docs")
        assert "swagger" not in response.text.lower()
        assert "redoc" not in response.text.lower()


class TestInputValidation:
    @pytest.mark.parametrize("query", ["", "   ", "hi", "too short"])
    def test_short_or_empty_queries_are_rejected(self, query: str) -> None:
        with TestClient(create_app(demo_settings())) as client:
            response = client.post("/api/research", json={"query": query})
        assert response.status_code in (400, 422)

    def test_oversized_query_is_rejected_before_any_work(self) -> None:
        with TestClient(create_app(demo_settings())) as client:
            response = client.post("/api/research", json={"query": "x " * 5_000})
        assert response.status_code in (400, 422)

    def test_absurd_payload_is_rejected(self) -> None:
        with TestClient(create_app(demo_settings())) as client:
            response = client.post("/api/research", json={"query": "a" * 10_000_000})
        assert response.status_code in (400, 413, 422)


class TestClientCannotWidenLimits:
    def test_demo_clamps_settings_regardless_of_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of DEMO_MODE: a client value may narrow a run,
        never widen one."""
        seen: dict[str, Any] = {}

        async def capture(query, settings, **kwargs):
            seen["rounds"] = settings.max_research_rounds
            seen["sources"] = settings.max_sources
            seen["fallback"] = settings.allow_cloud_fallback
            seen["persist"] = settings.persist_runs
            yield {"event": "result", "result": sample_result()}

        monkeypatch.setattr("agentic_research.web.api.stream_research", capture)
        with TestClient(create_app(demo_settings())) as client:
            client.post(
                "/api/research",
                json={
                    "query": "a genuine research question here",
                    "max_rounds": 5,
                    "max_sources": 40,
                },
            ).read()

        assert seen["rounds"] == DemoLimits().max_rounds
        assert seen["sources"] <= DemoLimits().max_sources
        assert seen["fallback"] is False
        assert seen["persist"] is False

    def test_a_client_may_still_narrow_a_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        async def capture(query, settings, **kwargs):
            seen["sources"] = settings.max_sources
            yield {"event": "result", "result": sample_result()}

        monkeypatch.setattr("agentic_research.web.api.stream_research", capture)
        with TestClient(create_app(demo_settings())) as client:
            client.post(
                "/api/research",
                json={"query": "a genuine research question here", "max_sources": 2},
            ).read()
        assert seen["sources"] == 2


class TestRateLimiting:
    def test_per_client_hourly_cap_returns_429_with_retry_after(self, patched_stream) -> None:
        patched_stream()
        app = create_app(demo_settings())
        app.state.research.limits = DemoLimits(runs_per_ip_per_hour=1)
        from agentic_research.web.limits import RateLimiter

        app.state.research.limiter = RateLimiter(app.state.research.limits)

        with TestClient(app) as client:
            first = client.post("/api/research", json={"query": "a real question here"})
            first.read()
            second = client.post("/api/research", json={"query": "another real question"})

        assert first.status_code == 200
        assert second.status_code == 429
        assert second.json()["capacity_reached"] is True
        assert "Retry-After" in second.headers

    def test_capacity_message_is_honest_not_a_generic_error(self, patched_stream) -> None:
        patched_stream()
        app = create_app(demo_settings())
        app.state.research.limits = DemoLimits(global_runs_per_day=0)
        from agentic_research.web.limits import RateLimiter

        app.state.research.limiter = RateLimiter(app.state.research.limits)
        with TestClient(app) as client:
            response = client.post("/api/research", json={"query": "a real question here"})
        assert response.status_code == 429
        assert "daily budget" in response.json()["error"]


class TestStreaming:
    def test_graph_events_reach_the_browser(self, patched_stream) -> None:
        patched_stream()
        with TestClient(create_app(demo_settings())) as client:
            body = client.post("/api/research", json={"query": "a genuine research question"}).text

        events = read_sse(body)
        names = [name for name, _ in events]
        assert names[0] == "started"
        assert "progress" in names
        assert "result" in names
        assert names[-1] == "done"

        progress = [p for n, p in events if n == "progress"]
        # Real node events, not invented UI milestones.
        assert any(p.get("event") == "plan_generated" for p in progress)

    def test_result_carries_claim_to_evidence_provenance(self, patched_stream) -> None:
        """The differentiator the UI is built around: a claim must arrive
        with the evidence ids that support it."""
        patched_stream()
        with TestClient(create_app(demo_settings())) as client:
            body = client.post("/api/research", json={"query": "a genuine question here"}).text

        result = next(p for n, p in read_sse(body) if n == "result")
        claim = result["report"]["summary_claims"][0]
        assert claim["evidence_ids"] == ["S1-e1"]
        assert claim["citation_ids"] == ["S1"]
        assert claim["kind"] == "factual"

        evidence = {e["id"]: e for e in result["evidence"]}
        assert evidence["S1-e1"]["quote"] == "a verbatim quote"
        assert evidence["S1-e1"]["citable"] is True

    def test_source_body_text_is_not_shipped_to_the_client(self, patched_stream) -> None:
        """State holds full page text; serialising it wholesale would bloat
        the payload and risk leaking configuration alongside it."""
        patched_stream()
        with TestClient(create_app(demo_settings())) as client:
            body = client.post("/api/research", json={"query": "a genuine question here"}).text
        assert "must not be exposed" not in body

    def test_failures_do_not_leak_internals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def explode(query, settings, **kwargs):
            raise RuntimeError("connection to https://api.internal/v1 with key sk-abc failed")
            yield  # pragma: no cover

        monkeypatch.setattr("agentic_research.web.api.stream_research", explode)
        with TestClient(create_app(demo_settings())) as client:
            body = client.post("/api/research", json={"query": "a genuine question here"}).text

        assert "sk-abc" not in body
        assert "api.internal" not in body
        assert "failed" in body.lower()

    def test_capacity_is_released_after_a_failed_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A leaked concurrency slot tells the next visitor the demo is busy
        forever."""

        async def explode(query, settings, **kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover

        monkeypatch.setattr("agentic_research.web.api.stream_research", explode)
        app = create_app(demo_settings())
        with TestClient(app) as client:
            client.post("/api/research", json={"query": "a genuine question here"}).read()
            assert client.get("/api/health").json()["capacity"]["active_runs"] == 0
