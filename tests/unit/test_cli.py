"""CLI behaviour, driven through Typer's runner.

The package and Docker smoke tests prove the entry point exists; they do
not cover flag handling, exit codes, or what happens when configuration is
wrong. Those are what a first-time user actually hits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_research.cli.main import app
from agentic_research.config import get_settings
from agentic_research.metrics import RunMetrics
from agentic_research.runner import RunResult

runner = CliRunner()


def combined(result: object) -> str:
    """stdout plus stderr.

    Errors are written to stderr, which is correct, and Click's runner keeps
    the streams separate; a test asserting only on stdout would silently
    pass whatever the message said.
    """
    out = getattr(result, "stdout", "") or ""
    try:
        err = result.stderr or ""  # type: ignore[attr-defined]
    except (ValueError, AttributeError):
        err = ""
    return out + err


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never read the developer's real .env during tests."""
    for key in (
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "BRAVE_API_KEY",
        "LLM_MODE",
        "MAX_RESEARCH_ROUNDS",
        "OUTPUT_DIR",
        "CHECKPOINT_BACKEND",
        "PERSIST_RUNS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM_MODE", "local")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("CHECKPOINT_BACKEND", "memory")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()


def fake_result(markdown: str = "# Report\n\nBody.\n", **metric_kwargs: object) -> RunResult:
    metrics = RunMetrics(
        run_id="test-run",
        query="q",
        mode="local",
        unique_sources=3,
        evidence_items=5,
        llm_calls=4,
        duration_s=1.0,
        **metric_kwargs,  # type: ignore[arg-type]
    )
    return RunResult(
        run_id="test-run",
        markdown=markdown,
        metrics=metrics,
        state={"verification": {"issues": []}},
        output_dir=None,
    )


def patch_stream(monkeypatch: pytest.MonkeyPatch, result: RunResult) -> None:
    async def fake_stream(query, settings, **kwargs):
        yield {"event": "started", "run_id": "test-run", "query": query, "models": {}}
        yield {"event": "plan_generated", "count": 2, "questions": ["a", "b"]}
        yield {"event": "completed", "stop_reason": "coverage sufficient"}
        yield {"event": "result", "result": result}

    monkeypatch.setattr("agentic_research.cli.main.stream_research", fake_stream)


class TestHelpAndDiscovery:
    def test_help_lists_every_command(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("research", "check", "show", "graph", "evaluate"):
            assert command in result.stdout

    def test_no_arguments_shows_help_rather_than_failing_obscurely(self) -> None:
        assert runner.invoke(app, []).exit_code in (0, 2)


class TestCheck:
    def test_check_reports_configuration_without_running_research(self) -> None:
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0
        assert "local" in result.stdout
        assert "Tavily key" in result.stdout

    def test_check_reports_a_missing_key_rather_than_crashing(self) -> None:
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0
        assert "missing" in result.stdout

    def test_invalid_configuration_exits_with_a_clear_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_MODE", "cloud")  # needs OPENAI_API_KEY
        get_settings.cache_clear()
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 1
        assert "OPENAI_API_KEY" in combined(result)


class TestGraph:
    def test_graph_emits_valid_mermaid(self) -> None:
        result = runner.invoke(app, ["graph"])
        assert result.exit_code == 0
        assert result.stdout.startswith("graph TD")
        assert "finalize --> END" in result.stdout

    def test_node_labels_survive_rich(self) -> None:
        """Rich treats [node] as markup and strips it, so this output must
        bypass it entirely."""
        result = runner.invoke(app, ["graph"])
        assert "analyze_query[analyze_query]" in result.stdout

    def test_graph_can_be_written_to_a_file(self, tmp_path: Path) -> None:
        target = tmp_path / "graph.mmd"
        result = runner.invoke(app, ["graph", "--output", str(target)])
        assert result.exit_code == 0
        assert target.read_text().startswith("graph TD")


class TestResearch:
    def test_runs_and_prints_a_report(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_stream(monkeypatch, fake_result())
        result = runner.invoke(app, ["research", "a question"])
        assert result.exit_code == 0
        assert "Report" in result.stdout

    def test_output_flag_writes_the_report(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        patch_stream(monkeypatch, fake_result())
        target = tmp_path / "nested" / "out.md"
        result = runner.invoke(app, ["research", "q", "--output", str(target), "--no-show"])
        assert result.exit_code == 0
        assert target.read_text().startswith("# Report")

    def test_quiet_emits_only_the_report(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_stream(monkeypatch, fake_result())
        result = runner.invoke(app, ["research", "q", "--quiet"])
        assert result.exit_code == 0
        assert "Researching:" not in result.stdout
        assert "Run metrics" not in result.stdout

    def test_verbose_shows_progress_detail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_stream(monkeypatch, fake_result())
        result = runner.invoke(app, ["research", "q", "--verbose", "--no-show"])
        assert result.exit_code == 0
        assert "Research plan" in result.stdout

    def test_unresolvable_citations_exit_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A degraded report must be detectable by a script, not only by
        reading the prose."""
        bad = fake_result()
        bad.state = {
            "verification": {"issues": [{"severity": "error", "type": "unknown_evidence"}]}
        }
        patch_stream(monkeypatch, bad)
        result = runner.invoke(app, ["research", "q", "--no-show"])
        assert result.exit_code == 2

    def test_mode_flag_overrides_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        async def capture(query, settings, **kwargs):
            seen["mode"] = settings.llm_mode.value
            seen["rounds"] = settings.max_research_rounds
            yield {"event": "result", "result": fake_result()}

        monkeypatch.setattr("agentic_research.cli.main.stream_research", capture)
        result = runner.invoke(
            app, ["research", "q", "--mode", "local", "--max-rounds", "1", "--no-show"]
        )
        assert result.exit_code == 0
        assert seen == {"mode": "local", "rounds": 1}

    def test_missing_search_key_exits_with_guidance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agentic_research.search.service import SearchProviderNotConfigured

        async def explode(query, settings, **kwargs):
            raise SearchProviderNotConfigured("SEARCH_PROVIDER=tavily requires TAVILY_API_KEY")
            yield  # pragma: no cover

        monkeypatch.setattr("agentic_research.cli.main.stream_research", explode)
        result = runner.invoke(app, ["research", "q"])
        assert result.exit_code == 1
        assert "TAVILY_API_KEY" in combined(result)


class TestShow:
    def test_show_reports_when_there_are_no_runs(self) -> None:
        result = runner.invoke(app, ["show"])
        assert result.exit_code == 1

    def test_show_latest_renders_a_stored_run(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "outputs" / "20260101-000000-abc"
        run_dir.mkdir(parents=True)
        (run_dir / "report.md").write_text("# Stored report\n\nBody.\n")
        (run_dir / "metrics.json").write_text(
            json.dumps(RunMetrics(run_id="r", query="q", mode="local").model_dump(mode="json"))
        )
        result = runner.invoke(app, ["show", "latest"])
        assert result.exit_code == 0
        assert "Stored report" in result.stdout

    def test_show_rejects_an_unknown_run_id(self, tmp_path: Path) -> None:
        (tmp_path / "outputs").mkdir(parents=True)
        result = runner.invoke(app, ["show", "does-not-exist"])
        assert result.exit_code == 1
