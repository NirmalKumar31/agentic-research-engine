"""A broken verifier must cost nothing.

Verification used to run last. Planning, searching, fetching,
extraction and synthesis all completed -- spending OpenAI tokens and
Tavily credits -- and only then was the NLI verifier constructed. A
misconfigured endpoint cost real money to discover, and the run that
discovered it published nothing anyway, because a claim that cannot be
checked is withheld.

These tests instrument the provider and search boundaries and assert
the counts are exactly zero when the verifier is not ready. That is the
release gate: not "fails gracefully", but "does not spend".
"""

from __future__ import annotations

import contextlib

import pytest

from agentic_research.citations.nli_preflight import NLIReadiness, check_nli_ready
from agentic_research.config import LLMMode, Settings
from agentic_research.runner import stream_research


class CountingRouter:
    """Fails the test if anything asks it for a model."""

    def __init__(self) -> None:
        self.preflights = 0
        self.model_requests = 0
        self.tracker = None
        self.raise_on_get = True

    def describe(self) -> dict[str, str]:
        return {"planner": "counted/none"}

    async def preflight(self) -> list[str]:
        self.preflights += 1
        return []

    def get(self, role: object) -> object:
        self.model_requests += 1
        if self.raise_on_get:
            raise AssertionError("a model was requested after a failed NLI preflight")
        raise RuntimeError("no model in this test")  # stops the graph harmlessly


class CountingSearch:
    instances = 0

    def __init__(self, *a: object, **k: object) -> None:
        CountingSearch.instances += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: object):
        return None


@pytest.fixture
def instrumented(monkeypatch: pytest.MonkeyPatch) -> CountingRouter:
    import agentic_research.runner as runner

    CountingSearch.instances = 0
    router = CountingRouter()
    monkeypatch.setattr(runner, "ModelRouter", lambda settings, tracker=None: router)
    monkeypatch.setattr(runner, "build_provider", lambda settings: object())
    monkeypatch.setattr(runner, "SearchService", CountingSearch)
    monkeypatch.setattr(runner, "PageFetcher", CountingSearch)
    return router


def settings() -> Settings:
    return Settings(
        llm_mode=LLMMode.LOCAL,
        persist_runs=False,
        checkpoint_backend="none",
        # A developer running the live endpoint has NLI_MODE=remote in
        # .env; without this the probe tests would exercise their
        # configuration instead of the one written here.
        _env_file=None,
    )


async def drive(s: Settings) -> list[dict]:
    return [event async for event in stream_research("a question", s)]


class TestAFailedPreflightSpendsNothing:
    async def test_no_model_is_ever_requested(
        self, instrumented: CountingRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agentic_research.runner.check_nli_ready",
            lambda _s: NLIReadiness(False, "endpoint unreachable"),
        )
        await drive(settings())
        assert instrumented.model_requests == 0, "a generative call was made"

    async def test_the_generative_preflight_never_runs(
        self, instrumented: CountingRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even the provider reachability check is a request. The
        verifier is checked first precisely so nothing else starts."""
        monkeypatch.setattr(
            "agentic_research.runner.check_nli_ready",
            lambda _s: NLIReadiness(False, "endpoint unreachable"),
        )
        await drive(settings())
        assert instrumented.preflights == 0

    async def test_no_search_service_is_constructed(
        self, instrumented: CountingRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agentic_research.runner.check_nli_ready",
            lambda _s: NLIReadiness(False, "endpoint unreachable"),
        )
        await drive(settings())
        assert CountingSearch.instances == 0, "a search provider was built"

    async def test_the_run_reports_a_controlled_failure(
        self, instrumented: CountingRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agentic_research.runner.check_nli_ready",
            lambda _s: NLIReadiness(False, "endpoint unreachable"),
        )
        events = await drive(settings())
        errors = [e for e in events if e.get("event") == "error"]
        assert errors, "the run ended without telling anyone why"
        assert errors[0].get("verifier_unavailable") is True
        assert not [e for e in events if e.get("event") == "result"]

    async def test_no_internal_detail_reaches_the_visitor(
        self, instrumented: CountingRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The operator gets the endpoint and the reason in logs; the
        visitor gets neither."""
        monkeypatch.setattr(
            "agentic_research.runner.check_nli_ready",
            lambda _s: NLIReadiness(False, "401 from https://nli.internal.example/score"),
        )
        events = await drive(settings())
        body = " ".join(str(e.get("error", "")) for e in events)
        for leak in ("401", "nli.internal.example", "http"):
            assert leak not in body, f"leaked {leak!r} to the visitor"


class TestAReadyPreflightLetsTheRunProceed:
    async def test_the_generative_preflight_then_runs(
        self, instrumented: CountingRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        instrumented.raise_on_get = False
        monkeypatch.setattr(
            "agentic_research.runner.check_nli_ready", lambda _s: NLIReadiness(True, "ready")
        )
        # The graph fails later for want of a real model; that is fine.
        # What matters is the ordering -- the generative preflight is
        # reached only once the verifier has been proven ready.
        with contextlib.suppress(Exception):
            await drive(settings())
        assert instrumented.preflights == 1


class TestTheProbeItself:
    """What the probe accepts and rejects, without a real model."""

    def _settings(self) -> Settings:
        return Settings(llm_mode=LLMMode.LOCAL, _env_file=None)

    def test_an_unavailable_verifier_is_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agentic_research.citations import nli_preflight
        from agentic_research.citations.fake_nli import FakeScorer

        monkeypatch.setattr(
            nli_preflight, "build_verifier", lambda _s: FakeScorer(fail_with="no checkpoint")
        )
        assert not check_nli_ready(self._settings()).ready

    def test_a_scorer_that_cannot_tell_entailment_from_contradiction_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Catches a non-NLI endpoint, or an inverted label mapping --
        the failure that would publish exactly the wrong claims."""
        from agentic_research.citations import nli_preflight
        from agentic_research.citations.fake_nli import FakeScorer
        from agentic_research.citations.nli_pin import (
            NLI_DEFAULT_MODEL_ID,
            NLI_DEFAULT_REVISION,
        )

        scorer = FakeScorer(default=(0.33, 0.34, 0.33))
        scorer.model_id = NLI_DEFAULT_MODEL_ID  # type: ignore[attr-defined]
        scorer.revision = NLI_DEFAULT_REVISION  # type: ignore[attr-defined]
        monkeypatch.setattr(nli_preflight, "build_verifier", lambda _s: scorer)
        result = check_nli_ready(self._settings())
        assert not result.ready
        assert "entailment" in result.detail

    def test_a_working_scorer_is_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agentic_research.citations import nli_preflight
        from agentic_research.citations.fake_nli import FakeScorer
        from agentic_research.citations.nli_pin import (
            NLI_DEFAULT_MODEL_ID,
            NLI_DEFAULT_REVISION,
        )

        entailed = ("The build completed successfully.", "The build succeeded.")
        contradicted = ("The build completed successfully.", "The build failed.")
        scorer = FakeScorer({entailed: (0.97, 0.02, 0.01), contradicted: (0.01, 0.04, 0.95)})
        scorer.model_id = NLI_DEFAULT_MODEL_ID  # type: ignore[attr-defined]
        scorer.revision = NLI_DEFAULT_REVISION  # type: ignore[attr-defined]
        monkeypatch.setattr(nli_preflight, "build_verifier", lambda _s: scorer)
        assert check_nli_ready(self._settings()).ready


class TestTheVerifierWakeIsNarrated:
    """A minute of invisible work reads as a hung page.

    A remote verifier at minimum replicas 0 takes roughly a minute to
    wake, and it happens after "started" and before the first graph
    stage. The stream carried heartbeats through that window and
    nothing else, so the UI sat on step 1 with no explanation for over
    a third of the run. The work was always real; it was never
    announced.
    """

    async def _events(self, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
        from agentic_research.citations import nli_preflight

        seen: list[dict] = []
        monkeypatch.setattr(
            "agentic_research.runner.check_nli_ready",
            lambda _s: nli_preflight.NLIReadiness(True, "ready"),
        )
        # The run fails later, at provider construction, because these
        # settings carry no search key. That is fine and deliberate:
        # everything under test here is emitted before then, and
        # letting the failure escape would discard it.
        with contextlib.suppress(Exception):
            async for event in stream_research("q", settings()):
                seen.append(event)
                if len(seen) > 8:
                    break
        return seen

    async def test_the_wake_is_announced_before_it_starts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events = await self._events(monkeypatch)
        names = [e.get("data", {}).get("event") or e.get("event") for e in events]
        assert "verifier_waking" in names, f"wake never announced; got {names}"

    async def test_it_comes_before_any_pipeline_stage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Announcing it afterwards would describe a wait that is over."""
        events = await self._events(monkeypatch)
        names = [e.get("data", {}).get("event") or e.get("event") for e in events]
        wake = names.index("verifier_waking")
        for stage in ("analyzing_query", "planning", "plan_generated"):
            if stage in names:
                assert wake < names.index(stage)

    async def test_it_carries_the_configured_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """So the UI can say how long the wait may be rather than
        leaving the reader to guess whether it has stalled."""
        events = await self._events(monkeypatch)
        wake = next(
            e["data"] for e in events if e.get("data", {}).get("event") == "verifier_waking"
        )
        assert "expected_seconds" in wake
        assert "detail" in wake
