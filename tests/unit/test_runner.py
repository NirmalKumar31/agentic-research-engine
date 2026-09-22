"""The runner: event stream, metrics assembly and artifact writing.

This file exists because of a real escape. The graph was covered by driving
`app.astream` directly in tests, so nothing exercised `stream_research`
itself, and a change from a list to a tuple for `stream_mode` — accepted by
the type checker — broke every run while the suite stayed green. LangGraph
switches on `isinstance(stream_mode, list)` to decide whether to yield
(mode, chunk) pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_research.config import Settings
from agentic_research.runner import RunResult, new_run_id, run_research, stream_research
from fakes import FakeFetcher, FakeRouter, FakeSearchService


@pytest.fixture(autouse=True)
def _wire_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the three I/O boundaries, leaving the runner itself real."""
    import agentic_research.runner as runner

    monkeypatch.setattr(runner, "ModelRouter", lambda settings, tracker=None: FakeRouter())

    class _Search(FakeSearchService):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    class _Fetcher(FakeFetcher):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(runner, "build_provider", lambda settings: object())
    monkeypatch.setattr(runner, "SearchService", lambda provider, settings: _Search())
    monkeypatch.setattr(runner, "PageFetcher", lambda settings: _Fetcher())


@pytest.fixture
def run_settings(settings: Settings, tmp_path: Path) -> Settings:
    settings.max_research_rounds = 1
    settings.persist_runs = False
    settings.checkpoint_backend = "memory"
    settings.output_dir = tmp_path / "outputs"
    return settings


class TestRunIds:
    def test_ids_are_unique_and_time_sortable(self) -> None:
        ids = [new_run_id() for _ in range(5)]
        assert len(set(ids)) == 5
        # Time-prefixed, so `ls outputs/` is chronological without extra tooling.
        assert ids == sorted(ids) or len({i[:8] for i in ids}) == 1


class TestStreaming:
    async def test_yields_progress_then_a_result(self, run_settings: Settings) -> None:
        events = [e async for e in stream_research("q", run_settings)]

        assert events[0]["event"] == "started"
        assert events[-1]["event"] == "result"
        assert isinstance(events[-1]["result"], RunResult)

        names = [e["event"] for e in events]
        for expected in (
            "plan_generated",
            "search_completed",
            "coverage_evaluated",
            "synthesized",
            "citations_verified",
            "completed",
        ):
            assert expected in names, f"missing {expected}; got {names}"

    async def test_events_are_dicts_not_raw_chunks(self, run_settings: Settings) -> None:
        """Guards the stream_mode regression directly: a non-list stream_mode
        yields bare chunks and unpacking (mode, chunk) fails."""
        async for event in stream_research("q", run_settings):
            assert isinstance(event, dict)
            assert "event" in event


class TestResult:
    async def test_produces_markdown_and_metrics(self, run_settings: Settings) -> None:
        result = await run_research("q", run_settings)

        assert result.markdown.startswith("# ")
        assert "## Sources" in result.markdown
        assert result.metrics.unique_sources > 0
        assert result.metrics.evidence_items > 0
        assert result.metrics.research_rounds == 1
        assert result.run_id

    async def test_metrics_reflect_the_run(self, run_settings: Settings) -> None:
        result = await run_research("q", run_settings)
        metrics = result.metrics
        assert metrics.llm_calls > 0
        assert metrics.search_queries > 0
        assert metrics.duration_s > 0
        assert metrics.citation_validity_rate == 1.0

    async def test_progress_callback_receives_events(self, run_settings: Settings) -> None:
        seen: list[str] = []
        await run_research("q", run_settings, on_progress=lambda e: seen.append(e["event"]))
        assert "completed" in seen
        assert "result" not in seen, "the result is returned, not delivered as progress"


class TestArtifacts:
    async def test_writes_inspectable_artifacts(self, run_settings: Settings) -> None:
        run_settings.persist_runs = True
        result = await run_research("q", run_settings)

        assert result.output_dir is not None
        written = {p.name for p in result.output_dir.iterdir()}
        assert written == {"report.md", "metrics.json", "sources.json", "evidence.json", "run.json"}

        evidence = json.loads((result.output_dir / "evidence.json").read_text())
        assert evidence and "quote" in evidence[0] and "quote_verified" in evidence[0]

        # Source text is excluded: it is large and already summarised by the
        # evidence quotes that reference it.
        sources = json.loads((result.output_dir / "sources.json").read_text())
        assert sources and "text" not in sources[0]
        assert "url" in sources[0] and "quality_score" in sources[0]

        run = json.loads((result.output_dir / "run.json").read_text())
        assert run["plan"] is not None
        assert run["queries"]

    async def test_persistence_can_be_disabled(self, run_settings: Settings) -> None:
        run_settings.persist_runs = False
        result = await run_research("q", run_settings)
        assert result.output_dir is None
