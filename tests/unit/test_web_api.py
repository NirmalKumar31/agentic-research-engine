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
        # These exercise the *live* path, so they opt into it explicitly.
        # It ships off by default because the hosted demo serves recorded
        # runs; the gating itself is covered in test_replay_mode.py.
        "live_research_enabled": True,
        # Deliberately secret-shaped, and allowlisted in .gitleaks.toml.
        "tavily_api_key": "tvly-test-key",
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
        # One run allowed, already consumed, so the daily cap is the thing
        # that refuses -- distinct from the zero-capacity case below.
        app.state.research.limits = DemoLimits(global_runs_per_day=1)
        from agentic_research.web.limits import RateLimiter

        app.state.research.limiter = RateLimiter(app.state.research.limits)
        with TestClient(app) as client:
            client.post("/api/research", json={"query": "a real question here"}).read()
            response = client.post("/api/research", json={"query": "another real question"})
        assert response.status_code == 429
        error = response.json()["error"]
        assert "daily budget" in error or "demo runs" in error


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


class TestCapacityMatchesProviderQuota:
    """The demo's daily cap must follow from the provider quota.

    Found live: the default cap of 60 runs/day sat above an account limit of
    50 provider requests/day. Three visitors would have drained the quota
    and everyone after would have hit an opaque mid-run 429 rather than an
    honest capacity message.
    """

    def test_daily_cap_is_derived_from_the_worst_case_not_the_average(self) -> None:
        from agentic_research.web.limits import runs_affordable

        # 50 quota against a 40-request worst case affords exactly one run.
        assert runs_affordable(50, 40) == 1
        assert runs_affordable(500, 40) == 12

    def test_a_quota_too_small_for_one_run_affords_zero(self) -> None:
        """max(1, ...) would promise a run the quota cannot pay for and
        turn a predictable refusal into a mid-run 429."""
        from agentic_research.web.limits import runs_affordable

        assert runs_affordable(30, 40) == 0
        assert runs_affordable(0, 40) == 0

    def test_settings_quota_flows_into_the_limits(self) -> None:
        from agentic_research.web.limits import limits_from_settings

        limits = limits_from_settings(
            Settings(
                llm_mode="local",
                demo_provider_requests_per_day=400,
                max_cloud_calls=40,
                _env_file=None,
            )
        )
        # 400 / 20, not 400 / 40: the account quota flows through unclamped,
        # but max_cloud_calls is clamped to the demo ceiling, so a run's
        # real worst case is 20 provider requests rather than the
        # configured 40.
        assert limits.global_runs_per_day == 20

    def test_default_cap_does_not_exceed_the_measured_quota(self) -> None:
        from agentic_research.web.limits import DemoLimits

        limits = DemoLimits()
        assert limits.global_runs_per_day * 40 <= limits.max_provider_requests_per_day

    def test_zero_capacity_is_an_honest_distinct_state(self) -> None:
        """Not 'try again later' -- live runs are off until the quota
        changes, and the message should say so."""
        import asyncio

        from agentic_research.web.limits import CapacityError, DemoLimits, RateLimiter

        limiter = RateLimiter(DemoLimits(global_runs_per_day=0))
        with pytest.raises(CapacityError) as info:
            asyncio.run(limiter.acquire("1.2.3.4"))
        assert "disabled" in info.value.reason
        assert info.value.retry_after_seconds is None


class TestDegradedCorpusRefusedInLibrary:
    """The CLI refused a degraded corpus; the library did not, so a Python
    caller could produce exactly the misleading comparison the CLI was
    protected from."""

    def test_compare_refuses_by_default(self) -> None:
        import asyncio

        from agentic_research.evaluation.ab import (
            DegradedCorpusError,
            EvidenceCorpus,
            compare,
        )
        from agentic_research.models import SourceDocument, SubQuestion

        corpus = EvidenceCorpus(
            question="q",
            sub_questions=[SubQuestion(id="SQ1", text="t", rationale="r")],
            sources=[
                SourceDocument(
                    id="S1",
                    url="https://x/a",
                    canonical_url="https://x/a",
                    title="T",
                    domain="x",
                    text="",  # stripped
                )
            ],
            evidence=[],
        )
        with pytest.raises(DegradedCorpusError, match="no text"):
            asyncio.run(compare(corpus, demo_settings(), {"a": {}}))

    def test_the_escape_hatch_must_be_explicit(self) -> None:
        import inspect

        from agentic_research.evaluation.ab import compare

        signature = inspect.signature(compare)
        parameter = signature.parameters["allow_degraded"]
        assert parameter.default is False
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
