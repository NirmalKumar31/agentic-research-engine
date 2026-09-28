"""Shared fixtures.

Default runs never touch a paid API. Anything that needs real credentials is
marked ``integration`` and deselected unless explicitly requested.
"""

from __future__ import annotations

import pytest

from agentic_research.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings that need no credentials."""
    return Settings(
        llm_mode="local",
        ollama_model="test-model:1b",
        tavily_api_key="tvly-test-key",
        max_research_rounds=2,
        max_search_queries=8,
        max_sources=10,
        max_sources_per_round=5,
        max_llm_calls=20,
        max_parallel_searches=3,
        max_parallel_fetches=4,
        persist_runs=False,
        checkpoint_backend="memory",
        _env_file=None,
    )


@pytest.fixture(autouse=True)
def _quiet_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "CRITICAL")


PUBLIC_TEST_IP = "93.184.216.34"


@pytest.fixture
def stub_dns(monkeypatch: pytest.MonkeyPatch) -> str:
    """Resolve every hostname to one public address.

    The fetcher pins the validated IP into the connection to close DNS
    rebinding, so the request URL it issues contains an address rather than
    a hostname. Tests therefore match on path and rely on this to make the
    pinned address deterministic.
    """
    import socket as _socket

    import agentic_research.retrieval.safety as safety

    def fake(host, *args, **kwargs):
        return [(_socket.AF_INET, 0, 0, "", (PUBLIC_TEST_IP, 0))]

    monkeypatch.setattr(safety.socket, "getaddrinfo", fake)
    return PUBLIC_TEST_IP


@pytest.fixture(autouse=True)
def fake_nli(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Never load the real 1.4GB checkpoint in a unit test.

    Verification runs inside the graph, so without this every test that
    drives a run would download and load DeBERTa -- minutes per run, and
    a 1.4GB pull in CI for tests that are not about semantics at all.

    The fake entails by default, so pipeline tests see claims survive
    the gate and can assert on what a report contains. Tests about the
    gate itself build their own scorers with specific scores, and the
    real checkpoint runs in the `nli` suite. Guards are NOT faked: they
    are pure functions and run for real against this.
    """
    if request.node.get_closest_marker("nli"):
        return
    from agentic_research.citations import nli_preflight
    from agentic_research.citations.fake_nli import FakeScorer
    from agentic_research.graph.nodes import reporting

    def fake(_settings: object) -> FakeScorer:
        return FakeScorer(default=(0.99, 0.01, 0.0))

    monkeypatch.setattr(reporting, "build_verifier", fake)

    # The runner now probes the verifier before spending anything, so
    # the preflight path needs the fake too -- otherwise every test that
    # drives a run downloads and loads a 1.4GB checkpoint.
    monkeypatch.setattr(
        nli_preflight,
        "check_nli_ready",
        lambda _settings: nli_preflight.NLIReadiness(True, "fake verifier ready"),
    )
    monkeypatch.setattr(
        "agentic_research.runner.check_nli_ready",
        lambda _settings: nli_preflight.NLIReadiness(True, "fake verifier ready"),
    )


@pytest.fixture(autouse=True)
def no_ollama_probe(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Environment capture must not reach the network in a unit test.

    ``capture()`` records the Ollama server version and model digests,
    which is genuinely useful for reproducing a local benchmark and is
    an HTTP request. Every test that drives a run made it, so the unit
    suite's behaviour depended on whether Ollama happened to be running
    on the developer's machine -- fast and populated locally, a timeout
    in CI. Tests that are about the capture itself opt out with the
    ``ollama`` marker.
    """
    if request.node.get_closest_marker("ollama") or request.node.get_closest_marker("integration"):
        return
    from agentic_research import environment

    monkeypatch.setattr(
        environment,
        "_ollama_info",
        lambda _base_url: {"version": "stubbed-in-tests"},
    )
