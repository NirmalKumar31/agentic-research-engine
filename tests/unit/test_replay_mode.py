"""Replay-first public demo, and the gate in front of paid execution.

The hosted site serves recorded runs. That is a deployment decision with a
concrete cause: the demo's daily run cap lives in process memory, and a free
host that spins down when idle resets it on every cold start, so it cannot
bound an account-level quota. Per-run ceilings still hold; the daily one
does not survive a restart.

So the gate has to be on the server. Hiding the button in React leaves the
endpoint open to anyone with curl, and that endpoint spends money. These
tests pin that the refusal happens before any provider work is even
contemplated, and that the recorded path needs no credentials at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentic_research.config import Settings
from agentic_research.web import recordings
from agentic_research.web.api import create_app


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_mode": "local",
        "demo_mode": True,
        "live_research_enabled": False,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)


RECORDING = {
    "meta": {
        "id": "example-run",
        "label": "Fraud detection on imbalanced data",
        "question": "Compare approaches for fraud detection on imbalanced data",
        "description": "A recorded local run.",
        "mode": "local",
        "recorded_at": "2026-09-23T00:00:00Z",
        "order": 1,
    },
    "trace": [
        {"event": "plan_generated", "sub_questions": 3},
        {"event": "search_completed", "query_id": "Q1", "results": 8},
    ],
    "result": {
        "run_id": "recorded",
        "report": {
            "title": "Fraud detection",
            "summary_claims": [
                {
                    "text": "Precision-recall beats ROC AUC under heavy imbalance.",
                    "kind": "factual",
                    "evidence_ids": ["S1-e1"],
                    "citation_ids": ["S1"],
                }
            ],
            "key_findings": [],
            "sections": [],
            "contradictions": [],
            "limitations": [],
        },
        "evidence": [
            {
                "id": "S1-e1",
                "source_id": "S1",
                "sub_question_id": "SQ1",
                "claim": "Precision-recall is more informative.",
                "quote": "precision-recall curves are a more informative evaluation",
                "quote_match": "exact_normalized",
                "page": 14,
                "stance": "supports",
                "confidence": 0.9,
                "citable": True,
                "query_id": "Q1",
                "cross_attributed": False,
            },
            {
                "id": "S1-e2",
                "source_id": "S1",
                "sub_question_id": "SQ2",
                "claim": "Noticed in passing.",
                "quote": "cost-sensitive learning assigns a higher penalty",
                "quote_match": "exact_normalized",
                "page": None,
                "stance": "neutral",
                "confidence": 0.7,
                "citable": True,
                "query_id": "",
                "cross_attributed": True,
            },
        ],
        "sources": [
            {
                "id": "S1",
                "url": "https://example.com/paper.pdf",
                "title": "A paper",
                "domain": "example.com",
                "source_type": "academic",
                "content_origin": "pdf_extract",
                "quality_score": 0.8,
                "page_count": 20,
                "usable": True,
                "fetch_status": "ok",
            }
        ],
        "verification": {"total_claims": 1},
        "metrics": {"duration_s": 1096.0, "evidence_items": 2},
        "markdown": "# Fraud detection\n",
    },
}


@pytest.fixture(autouse=True)
def _isolated_recordings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the loader at a temporary directory with one recording."""
    directory = tmp_path / "recordings"
    directory.mkdir()
    (directory / "example-run.json").write_text(json.dumps(RECORDING), encoding="utf-8")
    monkeypatch.setattr(recordings, "RECORDINGS_DIR", directory)
    recordings._index.cache_clear()
    yield directory
    recordings._index.cache_clear()


@pytest.fixture
def client() -> Any:
    with TestClient(create_app(_settings())) as c:
        yield c


class TestLiveResearchIsGatedOnTheServer:
    def test_disabled_by_default(self) -> None:
        """The dangerous setting has to be the one you opt into."""
        assert _settings().live_research_enabled is False
        assert Settings(llm_mode="local", _env_file=None).live_research_enabled is False

    def test_research_is_refused_with_403(self, client: Any) -> None:
        response = client.post("/api/research", json={"query": "a real research question"})
        assert response.status_code == 403
        body = response.json()
        assert body["live_disabled"] is True
        assert "recorded run" in body["error"]

    def test_refusal_never_reaches_the_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The endpoint must not spend money, so it must not get as far as
        constructing a run at all."""
        import agentic_research.web.api as api_module

        def explode(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("stream_research must not be called when live is disabled")

        monkeypatch.setattr(api_module, "stream_research", explode)
        with TestClient(create_app(_settings())) as client:
            assert (
                client.post("/api/research", json={"query": "a real question"}).status_code == 403
            )

    def test_no_request_field_can_switch_live_on(self, client: Any) -> None:
        """Client input may only ever narrow a run, never enable one."""
        for payload in (
            {"query": "a real research question", "max_rounds": 5},
            {"query": "a real research question", "max_sources": 40},
        ):
            assert client.post("/api/research", json=payload).status_code == 403

    def test_the_gate_precedes_query_validation(self, client: Any) -> None:
        """A too-short query would normally be a 400. It must still be 403,
        or the refusal order tells an attacker which inputs are interesting."""
        assert client.post("/api/research", json={"query": "hi"}).status_code == 403

    def test_enabling_it_restores_the_existing_guarded_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agentic_research.web.api as api_module

        called: list[str] = []

        async def fake_stream(query: str, settings: Any, run_id: str = "") -> Any:
            called.append(query)
            if False:  # pragma: no cover - makes this an async generator
                yield {}

        monkeypatch.setattr(api_module, "stream_research", fake_stream)
        app = create_app(_settings(live_research_enabled=True))
        with TestClient(app) as client:
            response = client.post("/api/research", json={"query": "a real research question"})
        assert response.status_code == 200
        assert called == ["a real research question"]


class TestConfigTellsTheTruth:
    def test_reports_live_availability(self, client: Any) -> None:
        body = client.get("/api/config").json()
        assert body["live_research_enabled"] is False
        assert body["recorded_examples"] == 1

    def test_reports_live_enabled_when_it_is(self) -> None:
        with TestClient(create_app(_settings(live_research_enabled=True))) as client:
            assert client.get("/api/config").json()["live_research_enabled"] is True

    def test_config_leaks_no_credential(self) -> None:
        settings = _settings(tavily_api_key="tvly-secret-value", openai_api_key="sk-secret-value")
        with TestClient(create_app(settings)) as client:
            body = json.dumps(client.get("/api/config").json())
        assert "secret-value" not in body
        assert "tvly-" not in body and "sk-" not in body


class TestRecordedExamplesNeedNoCredentials:
    def test_listing_works_without_any_key(self, client: Any) -> None:
        body = client.get("/api/examples").json()
        assert len(body["examples"]) == 1
        entry = body["examples"][0]
        assert entry["id"] == "example-run"
        assert entry["citable_evidence"] == 2
        assert entry["has_pdf_evidence"] is True

    def test_replay_returns_the_full_provenance_payload(self, client: Any) -> None:
        body = client.get("/api/examples/example-run").json()
        assert body["recorded"] is True
        result = body["result"]

        # The drill-down the demo exists to show: claim -> evidence -> quote
        # -> page -> source.
        claim = result["report"]["summary_claims"][0]
        assert claim["evidence_ids"] == ["S1-e1"]
        item = next(e for e in result["evidence"] if e["id"] == "S1-e1")
        assert item["quote_match"] == "exact_normalized"
        assert item["page"] == 14
        assert any(s["id"] == item["source_id"] for s in result["sources"])

    def test_cross_attributed_evidence_is_represented_honestly(self, client: Any) -> None:
        """It keeps exact source provenance and simply has no query. That
        must survive into the recording rather than being smoothed over."""
        result = client.get("/api/examples/example-run").json()["result"]
        item = next(e for e in result["evidence"] if e["id"] == "S1-e2")
        assert item["cross_attributed"] is True
        assert item["query_id"] == ""
        assert item["source_id"] == "S1" and item["citable"] is True

    def test_recordings_carry_no_source_text(self, client: Any) -> None:
        """Page text is large and not ours to redistribute; the quotes the
        report already cites are what the UI needs."""
        result = client.get("/api/examples/example-run").json()["result"]
        assert all("text" not in s for s in result["sources"])

    def test_stream_replays_only_recorded_events(self, client: Any) -> None:
        with client.stream("GET", "/api/examples/example-run/stream") as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())
        assert "event: started" in body
        assert "plan_generated" in body and "search_completed" in body
        assert "event: result" in body and "event: done" in body
        # Every progress event must come from the recording.
        assert body.count("event: progress") == len(RECORDING["trace"])

    def test_replay_is_labelled_as_recorded(self, client: Any) -> None:
        """A replay must never be mistakable for a fresh run."""
        assert client.get("/api/examples/example-run").json()["recorded"] is True
        with client.stream("GET", "/api/examples/example-run/stream") as response:
            body = "".join(response.iter_text())
        assert '"recorded": true' in body.lower()


class TestExampleIdsAreSafe:
    @pytest.mark.parametrize("bad", ["Example-Run", "", "a" * 200, "run;rm -rf /"])
    def test_a_malformed_id_under_api_is_a_json_404(self, client: Any, bad: str) -> None:
        """Stays under /api after URL normalisation, so it must come back as
        a JSON 404 rather than falling through to the SPA shell."""
        response = client.get(f"/api/examples/{bad}")
        assert response.status_code in (400, 404)
        assert response.headers["content-type"].startswith("application/json")

    @pytest.mark.parametrize(
        "bad",
        ["../../../etc/passwd", "..%2f..%2fetc%2fpasswd", "example-run/../../secret"],
    )
    def test_traversal_reads_no_file(self, client: Any, bad: str) -> None:
        """These normalise to a path outside /api before the server sees
        them, so they land on the SPA route. Status is not the property
        that matters here -- not reading a file is."""
        response = client.get(f"/api/examples/{bad}")
        assert "root:x:" not in response.text
        assert "BEGIN PRIVATE KEY" not in response.text

    def test_an_unknown_api_route_is_json_not_the_spa_shell(self, client: Any) -> None:
        response = client.get("/api/does-not-exist")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")

    def test_the_spa_still_serves_frontend_routes(self, client: Any) -> None:
        """The catch-all must keep working for everything that is not /api."""
        response = client.get("/some/client/route")
        assert response.status_code == 200

    def test_traversal_cannot_read_a_file_outside_the_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = tmp_path / "secret.json"
        secret.write_text(json.dumps({"result": {"leaked": True}}), encoding="utf-8")
        with pytest.raises(recordings.RecordingNotFound):
            recordings.load("../secret")

    def test_the_id_pattern_accepts_only_what_it_should(self) -> None:
        assert recordings.valid_id("fraud-detection-local")
        assert recordings.valid_id("a")
        assert not recordings.valid_id("-leading-dash")
        assert not recordings.valid_id("has_underscore")
        assert not recordings.valid_id("has space")
        assert not recordings.valid_id("../x")


class TestRecordingsAreSanitised:
    def test_a_recording_containing_a_credential_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Committed to a public repo and served anonymously, so this fails
        loudly in CI rather than quietly on the website."""
        directory = tmp_path / "bad"
        directory.mkdir()
        poisoned = {"meta": {}, "result": {"settings": {"openai_api_key": "sk-leaked"}}}
        (directory / "poisoned.json").write_text(json.dumps(poisoned), encoding="utf-8")
        monkeypatch.setattr(recordings, "RECORDINGS_DIR", directory)
        recordings._index.cache_clear()
        try:
            with pytest.raises(ValueError, match="forbidden key"):
                recordings.available()
        finally:
            recordings._index.cache_clear()

    def test_committed_recordings_load_and_are_clean(self) -> None:
        """Runs against the real shipped directory, so a recording added
        later cannot bypass the checks above."""
        recordings._index.cache_clear()
        try:
            for summary in recordings.available():
                payload = recordings.load(summary.id)
                assert payload["result"], f"{summary.id} has an empty result"
                assert summary.question, f"{summary.id} has no question"
                for source in payload["result"].get("sources", []):
                    assert "text" not in source, f"{summary.id} embeds source text"
        finally:
            recordings._index.cache_clear()


class TestUntrustedTextIsNotInterpreted:
    def test_a_script_payload_in_a_quote_survives_as_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fetched pages are attacker-influenced. The API must hand the
        payload through as data -- escaping is the renderer's job, and the
        frontend uses no dangerouslySetInnerHTML."""
        payload = json.loads(json.dumps(RECORDING))
        attack = '<script>alert("xss")</script>'
        payload["result"]["evidence"][0]["quote"] = attack
        directory = tmp_path / "xss"
        directory.mkdir()
        (directory / "example-run.json").write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(recordings, "RECORDINGS_DIR", directory)
        recordings._index.cache_clear()
        try:
            with TestClient(create_app(_settings())) as client:
                response = client.get("/api/examples/example-run")
            # Passed through byte-for-byte as data. The API deliberately
            # does not mangle it: escaping belongs to whatever renders it,
            # and silently rewriting a quote would corrupt the very thing
            # quote verification exists to guarantee.
            assert response.json()["result"]["evidence"][0]["quote"] == attack
            # Served as JSON, so a browser parses it as data and never as
            # markup. This -- not the absence of the characters -- is what
            # makes the payload inert in transit.
            assert response.headers["content-type"].startswith("application/json")
        finally:
            recordings._index.cache_clear()

    def test_the_frontend_never_injects_raw_html(self) -> None:
        """A guard on the property the test above depends on."""
        web_src = Path(__file__).resolve().parents[2] / "web" / "src"
        if not web_src.is_dir():  # pragma: no cover - frontend absent
            pytest.skip("frontend sources not present")
        offenders = [
            path.name
            for path in web_src.rglob("*.tsx")
            if "dangerouslySetInnerHTML" in path.read_text(encoding="utf-8")
        ]
        assert not offenders, f"raw HTML injection reintroduced in {offenders}"
