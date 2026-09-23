"""Measurement provenance and version consistency.

A metric is only meaningful alongside the code that produced it. The first
cloud run surfaced three bugs and the implementation changed immediately
afterwards, silently invalidating the numbers that run had just produced --
and nothing in the artifact said which code it came from.

These tests pin that every artifact carries enough to identify its own
origin, and that the fingerprints actually move when the thing they
fingerprint moves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_research.config import Settings
from agentic_research.provenance import (
    EVALUATOR_VERSION,
    capture,
    config_fingerprint,
    describe,
    git_state,
    prompt_version,
    schema_version,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestFingerprints:
    def test_prompt_version_is_stable_across_calls(self) -> None:
        assert prompt_version() == prompt_version()
        assert prompt_version() != "unknown"

    def test_schema_version_is_stable_across_calls(self) -> None:
        assert schema_version() == schema_version()
        assert schema_version() != "unknown"

    def test_prompt_and_schema_versions_are_distinct(self) -> None:
        assert prompt_version() != schema_version()

    def test_schema_version_tracks_the_json_contract(self) -> None:
        """Derived from the JSON Schema the model is held to, so a field
        change moves it and a comment change does not."""
        import hashlib

        from pydantic import BaseModel

        from agentic_research import schemas

        shapes = {
            name: getattr(schemas, name).model_json_schema()
            for name in sorted(schemas.__all__)
            if isinstance(getattr(schemas, name), type)
            and issubclass(getattr(schemas, name), BaseModel)
        }
        expected = hashlib.sha256(json.dumps(shapes, sort_keys=True).encode("utf-8")).hexdigest()[
            :12
        ]
        assert schema_version() == expected

    def test_config_fingerprint_moves_with_behaviour(self) -> None:
        base = Settings(llm_mode="local", max_research_rounds=2, _env_file=None)
        same = Settings(llm_mode="local", max_research_rounds=2, _env_file=None)
        different = Settings(llm_mode="local", max_research_rounds=5, _env_file=None)

        assert config_fingerprint(base) == config_fingerprint(same)
        assert config_fingerprint(base) != config_fingerprint(different)

    def test_config_fingerprint_moves_with_model_assignment(self) -> None:
        a = Settings(llm_mode="local", ollama_model="qwen3:4b", _env_file=None)
        b = Settings(llm_mode="local", ollama_model="llama3.2:3b", _env_file=None)
        assert config_fingerprint(a) != config_fingerprint(b)

    def test_config_fingerprint_never_embeds_a_credential(self) -> None:
        """A fingerprint that moved on key rotation would be useless, and
        one that embedded a key would be dangerous."""
        with_key = Settings(llm_mode="local", tavily_api_key="tvly-test-key", _env_file=None)
        without = Settings(llm_mode="local", _env_file=None)
        assert config_fingerprint(with_key) == config_fingerprint(without)


class TestGitState:
    def test_reports_commit_and_dirty_flag(self) -> None:
        state = git_state()
        assert "commit" in state
        assert "dirty" in state
        if state["commit"] != "unavailable":
            assert len(state["commit"]) == 40
            assert state["short_commit"] == state["commit"][:8]
            assert isinstance(state["dirty"], bool)

    def test_absent_git_is_reported_not_guessed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An installed wheel has no repository; saying so beats inventing
        a commit."""
        import agentic_research.provenance as provenance

        monkeypatch.setattr(provenance, "_git", lambda *args: None)
        for name in provenance._COMMIT_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        git_state.cache_clear()
        try:
            state = git_state()
        finally:
            git_state.cache_clear()
        assert state["commit"] == "unavailable"
        assert state["dirty"] is None

    def test_a_deployed_image_reports_its_build_commit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The container ships no git binary and no .git, so without this
        fallback every deployed instance would report "unavailable" -- the
        one place the commit is actually needed."""
        import agentic_research.provenance as provenance

        monkeypatch.setattr(provenance, "_git", lambda *args: None)
        for name in provenance._COMMIT_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("RENDER_GIT_COMMIT", "abcdef1234567890")
        git_state.cache_clear()
        try:
            state = git_state()
        finally:
            git_state.cache_clear()
        assert state["commit"] == "abcdef1234567890"
        assert state["short_commit"] == "abcdef12"
        # No working tree in an image, so this is known-clean rather than
        # unknown.
        assert state["dirty"] is False

    def test_git_state_is_cached_so_health_checks_are_cheap(self) -> None:
        """/api/health calls this on every probe; it was spawning a git
        subprocess each time."""
        import agentic_research.provenance as provenance

        git_state.cache_clear()
        calls: list[tuple[str, ...]] = []
        original = provenance._git

        def counting(*args: str) -> str | None:
            calls.append(args)
            return original(*args)

        provenance._git = counting  # type: ignore[assignment]
        try:
            git_state()
            first = len(calls)
            git_state()
            assert len(calls) == first, "second call re-shelled out to git"
        finally:
            provenance._git = original  # type: ignore[assignment]
            git_state.cache_clear()


class TestCaptureBlock:
    def test_contains_every_required_field(self) -> None:
        block = capture(Settings(llm_mode="local", _env_file=None))
        for field in (
            "engine_version",
            "evaluator_version",
            "prompt_version",
            "schema_version",
            "git",
            "config_fingerprint",
            "model_assignments",
        ):
            assert field in block, f"provenance is missing {field}"

    def test_works_without_settings(self) -> None:
        block = capture()
        assert block["engine_version"]
        assert "config_fingerprint" not in block

    def test_is_json_serialisable(self) -> None:
        """It is written into artifact files, so it has to survive JSON."""
        block = capture(Settings(llm_mode="local", _env_file=None))
        assert json.loads(json.dumps(block))

    def test_describe_is_a_single_readable_line(self) -> None:
        line = describe(capture(Settings(llm_mode="local", _env_file=None)))
        assert "\n" not in line
        assert "prompts" in line and "schemas" in line

    def test_environment_capture_embeds_provenance(self) -> None:
        from agentic_research.environment import capture as capture_environment

        snapshot = capture_environment(Settings(llm_mode="local", _env_file=None))
        assert "provenance" in snapshot
        assert snapshot["provenance"]["prompt_version"] == prompt_version()


class TestVersionConsistency:
    """One version, declared once. The package and the API drifted apart
    once already (0.1.0 against 0.2.0)."""

    def test_package_and_pyproject_agree(self) -> None:
        import tomllib

        from agentic_research import __version__

        declared = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]["version"]
        assert declared == __version__

    def test_api_reports_the_package_version(self) -> None:
        from agentic_research import __version__
        from agentic_research.web.api import create_app

        app = create_app(Settings(llm_mode="local", _env_file=None))
        assert app.version == __version__

    def test_frontend_package_json_agrees(self) -> None:
        from agentic_research import __version__

        manifest = json.loads((REPO_ROOT / "web" / "package.json").read_text(encoding="utf-8"))
        assert manifest["version"] == __version__

    def test_health_endpoint_identifies_the_build(self) -> None:
        """A deployed instance should say which code it is running rather
        than leaving it to be inferred from the deploy time."""
        from fastapi.testclient import TestClient

        from agentic_research import __version__
        from agentic_research.web.api import create_app

        with TestClient(create_app(Settings(llm_mode="local", _env_file=None))) as client:
            body = client.get("/api/health").json()
        assert body["version"] == __version__
        assert "commit" in body

    def test_evaluator_version_is_set(self) -> None:
        assert EVALUATOR_VERSION and EVALUATOR_VERSION.strip()


class TestArtifactsCarryProvenance:
    def test_run_metrics_include_the_environment_block(self) -> None:
        from agentic_research.environment import capture as capture_environment
        from agentic_research.llm.base import UsageTracker
        from agentic_research.metrics import build_metrics
        from agentic_research.retrieval.fetcher import FetchStats
        from agentic_research.search.service import SearchStats

        settings = Settings(llm_mode="local", _env_file=None)
        metrics = build_metrics(
            run_id="r1",
            query="q",
            mode="local",
            model_assignments={},
            duration_s=1.0,
            state={},
            usage=UsageTracker(10),
            search_stats=SearchStats(),
            fetch_stats=FetchStats(),
            environment=capture_environment(settings),
        )
        provenance = metrics.environment["provenance"]
        assert provenance["git"]["commit"]
        assert provenance["prompt_version"]
        assert provenance["schema_version"]

    @pytest.mark.parametrize("field", ["prompt_version", "schema_version", "git"])
    def test_provenance_survives_a_json_round_trip(self, field: str) -> None:
        block = json.loads(json.dumps(capture(Settings(llm_mode="local", _env_file=None))))
        assert block[field]
