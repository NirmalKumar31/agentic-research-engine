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
from agentic_research.web.recordings import RECORDING_SCHEMA_VERSION


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_mode": "local",
        "demo_mode": True,
        "live_research_enabled": False,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)


def _fake_credential(*parts: str) -> str:
    """Assemble a credential-shaped string at runtime.

    Never written as a whole literal. A committed secret-shaped string is a
    permanent finding in the full-history scan even when it was always
    fake, and rewriting shared history to remove one is far worse than
    joining two fragments here. Three of these had to be allowlisted after
    exactly that mistake.
    """
    return "".join(parts)


# Shapes only. None of these is, or ever was, a real credential.
FAKE_CREDENTIALS = (
    _fake_credential("sk-", "proj-", "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"),
    _fake_credential("tvly", "-", "AbCdEfGh01234567890123456789"),
    _fake_credential("ghp", "_", "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"),
    _fake_credential("AKIA", "IOSFODNN7EXAMPLE"),
    _fake_credential("-----BEGIN ", "RSA PRIVATE KEY-----"),
)


RECORDING = {
    "recording_schema_version": RECORDING_SCHEMA_VERSION,
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
        "plan": {
            "sub_questions": [
                {
                    "id": "SQ1",
                    "text": "Which metrics suit heavy class imbalance?",
                    "is_followup": False,
                },
                {
                    "id": "SQ2",
                    "text": "Which resampling methods are used?",
                    "is_followup": False,
                },
            ],
        },
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
        assert body["service_mode"] == "replay"
        assert body["recorded_examples"] == 1

    def test_reports_live_enabled_when_it_is(self) -> None:
        with TestClient(create_app(_settings(live_research_enabled=True))) as client:
            body = client.get("/api/config").json()
        assert body["live_research_enabled"] is True
        assert body["service_mode"] == "live"

    def test_replay_mode_never_advertises_a_local_model(self) -> None:
        """The hosted instance boots with LLM_MODE=local only because cloud
        mode refuses to start without an API key it will never use. There
        is no Ollama on that host, so claiming a local model is available
        would be false. Boot configuration is not a capability."""
        with TestClient(create_app(_settings(llm_mode="local"))) as client:
            body = client.get("/api/config").json()
        assert body["service_mode"] == "replay"
        assert body["local_models_available"] is False
        assert body["mode"] is None, "internal LLM_MODE must not be advertised while replaying"

    def test_local_models_are_advertised_only_when_live_and_local(self) -> None:
        with TestClient(
            create_app(_settings(llm_mode="local", live_research_enabled=True))
        ) as client:
            body = client.get("/api/config").json()
        assert body["local_models_available"] is True
        assert body["mode"] == "local"

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
    @pytest.mark.parametrize("bad", ["Example-Run", "a" * 200, "run;rm -rf /"])
    def test_a_malformed_id_under_api_is_a_json_404(self, client: Any, bad: str) -> None:
        """Stays under /api after URL normalisation, so it must come back as
        a JSON 404 rather than falling through to the SPA shell.

        The empty string is deliberately absent: ``/api/examples/`` is the
        collection's trailing-slash form, not an invalid id. Asserting 404
        for it demanded the collection break. See TestTheCollectionRoute.
        """
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


class TestTheCollectionRoute:
    """``/api/examples/`` is the collection, not an empty example id.

    That distinction is why an earlier malformed-id test was wrong, and the
    trailing-slash form once behaved differently depending on whether a
    frontend build happened to exist.
    """

    def test_the_collection_returns_the_listing(self, client: Any) -> None:
        response = client.get("/api/examples")
        assert response.status_code == 200
        assert response.json()["examples"][0]["id"] == "example-run"

    def test_the_trailing_slash_form_reaches_the_same_collection(self, client: Any) -> None:
        response = client.get("/api/examples/")
        assert response.status_code == 200
        assert response.json()["examples"][0]["id"] == "example-run"

    def test_the_trailing_slash_redirects_rather_than_404s(self, client: Any) -> None:
        """Starlette's redirect must survive. A catch-all route matches every
        path and suppresses it, which is exactly how this broke."""
        response = client.get("/api/examples/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"].endswith("/api/examples")


class TestRoutingIsIdenticalWithAndWithoutAFrontendBuild:
    """The Python CI job never builds the frontend, so ``web/dist`` is absent
    there and present locally.

    Two tests once passed locally and failed in CI for precisely that
    reason. These fix the frontend state explicitly rather than inheriting
    whatever the working tree happens to contain.
    """

    @pytest.fixture
    def built_frontend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        dist = tmp_path / "web" / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<!doctype html><title>app</title>", encoding="utf-8")
        (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
        monkeypatch.setattr("agentic_research.web.api._FRONTEND_DIST", dist)
        return dist

    @pytest.fixture
    def no_frontend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        missing = tmp_path / "absent"
        monkeypatch.setattr("agentic_research.web.api._FRONTEND_DIST", missing)
        return missing

    def test_with_a_build_a_client_route_gets_the_app_shell(self, built_frontend: Path) -> None:
        with TestClient(create_app(_settings())) as client:
            response = client.get("/some/client/route")
        assert response.status_code == 200
        assert "<title>app</title>" in response.text

    def test_with_a_build_a_real_static_file_is_served(self, built_frontend: Path) -> None:
        with TestClient(create_app(_settings())) as client:
            assert client.get("/favicon.svg").status_code == 200

    def test_with_a_build_unknown_api_paths_are_still_json(self, built_frontend: Path) -> None:
        with TestClient(create_app(_settings())) as client:
            unknown = client.get("/api/does-not-exist")
            bad_id = client.get("/api/examples/Not-Valid")
            collection = client.get("/api/examples/")
        assert unknown.status_code == 404
        assert unknown.headers["content-type"].startswith("application/json")
        assert bad_id.status_code == 404
        assert bad_id.headers["content-type"].startswith("application/json")
        # The collection resolves whether or not a build exists.
        assert collection.status_code == 200

    def test_without_a_build_nothing_pretends_to_serve_a_frontend(self, no_frontend: Path) -> None:
        with TestClient(create_app(_settings())) as client:
            client_route = client.get("/some/client/route")
            collection = client.get("/api/examples/")
        assert client_route.status_code == 404
        assert client_route.headers["content-type"].startswith("application/json")
        assert collection.status_code == 200

    def test_a_traversal_outside_the_build_is_refused(self, built_frontend: Path) -> None:
        secret = built_frontend.parent.parent / "secret.txt"
        secret.write_text("TOP SECRET", encoding="utf-8")
        with TestClient(create_app(_settings())) as client:
            response = client.get("/../../secret.txt")
        assert "TOP SECRET" not in response.text


class TestOneCanonicalSerialiser:
    """A recording must be the same shape as a live result.

    They are produced in different places -- the CLI recorder and the SSE
    endpoint -- and if the two drifted the UI would need a second render
    path, which is how a replay stops matching the thing it is imitating.
    """

    def test_the_api_and_the_recorder_share_one_implementation(self) -> None:
        import agentic_research.web.api as api_module
        from agentic_research.web.recordings import serialise_result

        assert api_module._serialise_result is serialise_result

    def test_a_recording_has_the_same_result_shape_as_a_live_run(self, client: Any) -> None:
        # Imported the way `fakes` is, by module name rather than through a
        # `tests.` package path. There is no tests/__init__.py, so the
        # dotted form only resolves when the repo root happens to be on
        # sys.path -- true under `python -m pytest`, false under a bare
        # `pytest`, which is what CI runs.
        from test_web_api import sample_result

        from agentic_research.web.recordings import serialise_result

        live = serialise_result(sample_result())
        recorded = client.get("/api/examples/example-run").json()["result"]

        assert set(recorded) == set(live), "recording and live result differ at the top level"
        for key in ("evidence", "sources"):
            if live[key] and recorded[key]:
                assert set(recorded[key][0]) == set(live[key][0]), f"{key} entries differ"
        if live["report"] and recorded["report"]:
            assert set(recorded["report"]) == set(live["report"])
            live_claim = live["report"]["summary_claims"][0]
            recorded_claim = recorded["report"]["summary_claims"][0]
            assert set(recorded_claim) == set(live_claim)

    def test_the_schema_version_is_recorded_and_checked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A recording outlives the code that made it, so an incompatible
        one must disappear rather than render as a subtly broken demo."""
        from agentic_research.web.recordings import RECORDING_SCHEMA_VERSION

        stale = json.loads(json.dumps(RECORDING))
        stale["recording_schema_version"] = RECORDING_SCHEMA_VERSION + 1
        directory = tmp_path / "stale"
        directory.mkdir()
        (directory / "example-run.json").write_text(json.dumps(stale), encoding="utf-8")
        monkeypatch.setattr(recordings, "RECORDINGS_DIR", directory)
        recordings._index.cache_clear()
        try:
            assert recordings.available() == []
        finally:
            recordings._index.cache_clear()


class TestTheTraceIsAnAllowlist:
    """Progress events land in a committed, publicly served file, so what
    survives is listed rather than filtered."""

    def test_unlisted_fields_are_dropped(self) -> None:
        cleaned = recordings.sanitise_trace(
            [
                {
                    "event": "search_completed",
                    "query_id": "Q1",
                    "query": "fraud detection",
                    "results": 8,
                    "api_key": "leaked",
                    "internal_state": {"anything": "at all"},
                }
            ]
        )
        assert cleaned == [
            {
                "event": "search_completed",
                "query": "fraud detection",
                "query_id": "Q1",
                "results": 8,
            }
        ]

    def test_an_unknown_event_type_is_dropped_entirely(self) -> None:
        """A new graph event has to be considered before it can appear in a
        public file, rather than arriving there by default."""
        assert recordings.sanitise_trace([{"event": "brand_new_stage", "secret": "x"}]) == []

    def test_started_keeps_the_question_but_not_the_model_map(self) -> None:
        cleaned = recordings.sanitise_trace(
            [{"event": "started", "run_id": "r1", "query": "q", "models": {"planner": "gpt-6"}}]
        )
        assert cleaned == [{"event": "started", "query": "q"}]

    def test_public_provenance_drops_settings_and_packages(self) -> None:
        """Identifiers, not a deployment dump."""
        from agentic_research.config import Settings
        from agentic_research.environment import capture as capture_environment

        full = capture_environment(Settings(llm_mode="local", _env_file=None))
        public = recordings.public_provenance(full)

        assert "settings" not in public and "packages" not in public and "ollama" not in public
        assert public["prompt_version"] and public["schema_version"]
        assert "commit" in public and "dirty" in public


class TestRecordingsAreSanitised:
    def _write(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
        directory = tmp_path / "bad"
        directory.mkdir(exist_ok=True)
        (directory / "poisoned.json").write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(recordings, "RECORDINGS_DIR", directory)
        recordings._index.cache_clear()

    def test_a_forbidden_key_name_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Committed to a public repo and served anonymously, so this fails
        loudly in CI rather than quietly on the website."""
        payload = json.loads(json.dumps(RECORDING))
        payload["result"]["settings"] = {"openai_api_key": ""}
        self._write(tmp_path, monkeypatch, payload)
        try:
            with pytest.raises(ValueError, match="forbidden key"):
                recordings.available()
        finally:
            recordings._index.cache_clear()

    @pytest.mark.parametrize("secret", FAKE_CREDENTIALS)
    def test_a_credential_shaped_value_is_refused_whatever_it_is_called(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, secret: str
    ) -> None:
        """A key-name check alone passes {"note": "sk-proj-..."} straight
        through, which is the realistic way one of these gets committed."""
        payload = json.loads(json.dumps(RECORDING))
        payload["result"]["evidence"][0]["claim"] = f"An innocuous note: {secret}"
        self._write(tmp_path, monkeypatch, payload)
        try:
            with pytest.raises(ValueError, match="credential-shaped"):
                recordings.available()
        finally:
            recordings._index.cache_clear()

    def test_the_refusal_never_quotes_the_secret_it_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This runs in CI logs. A message containing the credential has
        published it a second time."""
        secret = FAKE_CREDENTIALS[0]
        payload = json.loads(json.dumps(RECORDING))
        payload["result"]["markdown"] = secret
        self._write(tmp_path, monkeypatch, payload)
        try:
            with pytest.raises(ValueError) as caught:
                recordings.available()
            assert secret not in str(caught.value)
            assert secret[:8] not in str(caught.value)
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


class TestNoInternalReasoningIsPublished:
    """Public payloads carry research, not the model's deliberation.

    The planner's strategy note is free prose a model writes to itself
    while deciding how to decompose a question, and it reliably opens with
    "The user wants me to...". It was committed in all three recordings
    and served to anonymous visitors before this check existed.
    """

    SELF_TALK = [
        "The user wants me to decompose this question",
        "I need to create sub-questions that cover different angles",
        "Let me think through this carefully",
        "I should focus on the technical dimensions first",
        "As an AI language model, I will break this down",
        "I will now generate the sub-questions",
        "I'll approach this by splitting the question",
        "My task is to plan the research",
    ]

    def test_the_canonical_recordings_are_clean(self) -> None:
        """The shipped artifacts, scanned as committed."""
        from agentic_research.web.recordings import (
            RECORDINGS_DIR,
            assert_no_internal_reasoning,
        )

        paths = sorted(RECORDINGS_DIR.glob("*.json"))
        assert paths, "no recordings found to scan"
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert_no_internal_reasoning(path.stem, payload)

    def test_no_recording_carries_a_plan_strategy(self) -> None:
        from agentic_research.web.recordings import RECORDINGS_DIR

        for path in sorted(RECORDINGS_DIR.glob("*.json")):
            plan = json.loads(path.read_text(encoding="utf-8"))["result"].get("plan") or {}
            assert "strategy" not in plan, path.name
            assert plan.get("sub_questions"), f"{path.name} lost its sub-questions"

    @pytest.mark.parametrize("phrase", SELF_TALK)
    def test_each_self_talk_phrase_is_rejected(self, phrase: str) -> None:
        """Placed in an ordinary field, not the removed one: the check has
        to catch a *new* free-text field, not just the known offender."""
        from agentic_research.web.recordings import assert_no_internal_reasoning

        payload = {"result": {"plan": {"sub_questions": [{"id": "SQ1", "text": phrase}]}}}
        with pytest.raises(ValueError, match="self-talk"):
            assert_no_internal_reasoning("probe", payload)

    def test_a_reinstated_strategy_field_is_rejected_by_name(self) -> None:
        """Even holding innocuous text. The key itself is the defect."""
        from agentic_research.web.recordings import assert_no_internal_reasoning

        payload = {"result": {"plan": {"strategy": "Compare the two approaches."}}}
        with pytest.raises(ValueError, match="internal reasoning"):
            assert_no_internal_reasoning("probe", payload)

    def test_a_quoted_source_passage_is_not_a_false_positive(self) -> None:
        """Evidence quotes are verbatim third-party text. A page that says
        "let me explain" must not take the recording out of the demo."""
        from agentic_research.web.recordings import assert_no_internal_reasoning

        payload = {
            "result": {
                "evidence": [{"quote": "Let me explain why recall matters here."}],
                "sources": [{"title": "I need to understand caching"}],
                "markdown": "As an AI practitioner, let me note the trade-off.",
            },
            "meta": {"question": "I need to compare vector databases"},
        }
        assert_no_internal_reasoning("probe", payload)

    def test_the_live_serialiser_omits_the_strategy_note(self) -> None:
        """Replay and live share one serialiser, so the live path must be
        clean for the same reason -- asserted directly rather than
        inferred from the recordings."""
        from test_web_api import sample_result

        from agentic_research.models import QueryAnalysis, ResearchPlan, SubQuestion
        from agentic_research.web.recordings import (
            assert_no_internal_reasoning,
            serialise_result,
        )

        result = sample_result()
        # A plan carrying exactly the prose this check exists to stop, so
        # the assertion cannot pass merely because the plan was absent.
        result.state["plan"] = ResearchPlan(
            analysis=QueryAnalysis(original_query="q", normalized_query="q", intent="compare"),
            strategy_note="The user wants me to compare these approaches. I need to split it.",
            sub_questions=[
                SubQuestion(
                    id="SQ1",
                    text="Which metrics suit imbalance?",
                    # Internal field, deliberately not published.
                    rationale="I can find this in the retrieved pages.",
                )
            ],
        )

        payload = serialise_result(result)
        assert payload["plan"] is not None
        assert "strategy" not in payload["plan"]
        assert payload["plan"]["sub_questions"]
        assert_no_internal_reasoning("live", {"result": payload})

    def test_the_served_api_payload_is_clean(self) -> None:
        """End to end, through the route an anonymous visitor hits."""
        from agentic_research.web.recordings import assert_no_internal_reasoning

        settings = Settings(
            llm_mode="local", demo_mode=True, live_research_enabled=False, _env_file=None
        )
        with TestClient(create_app(settings)) as client:
            for example in client.get("/api/examples").json()["examples"]:
                body = client.get(f"/api/examples/{example['id']}").json()
                assert_no_internal_reasoning(example["id"], body)
