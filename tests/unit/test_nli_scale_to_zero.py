"""Recovering from a verifier that was asleep.

The deployment runs the endpoint at minimum replicas 0, so the first
request after a quiet period arrives at nothing. Hugging Face documents
the behaviour: while a replica initialises "the HTTP server responds
with a 502 Bad Gateway", and requests are rejected rather than queued.
A measured cold start on this endpoint took 49.2s.

Waiting for that is the difference between a demo that works after
lunch and one that withholds every claim in its first run. Waiting for
anything else is how a wedged container, a revoked token or a typo in
the URL consumes the whole warm-up budget and then fails anyway.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from agentic_research.citations import nli as nli_module
from agentic_research.citations.nli import NLIUnavailable, RemoteNLIVerifier

ENDPOINT = "https://nli.example.test"
PAIRS = [("The build completed successfully.", "The build succeeded.")]
TOKEN = "hf_secret_token_value_do_not_log"


class _Encoding:
    def __init__(self, count: int) -> None:
        self.ids = [0] * count


class _Tokenizer:
    def encode(self, premise: str, hypothesis: str) -> _Encoding:
        return _Encoding(10)


@pytest.fixture(autouse=True)
def _stub_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nli_module, "load_pair_tokenizer", lambda *a, **k: _Tokenizer())


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Records what the retry loop asked to wait, and advances a fake
    clock by that much.

    Recording without advancing looks harmless and is not: the loop
    compares against a deadline taken from ``time.monotonic``, so a
    sleep that does not move the clock leaves it spinning until the
    real one catches up. A 200s budget then costs 200 real seconds of
    busy-waiting, which is how this suite went from 23s to four and a
    half minutes.
    """
    slept: list[float] = []
    clock = [0.0]

    def advance(seconds: float) -> None:
        slept.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(nli_module.time, "sleep", advance)
    monkeypatch.setattr(nli_module.time, "monotonic", lambda: clock[0])
    return slept


def verifier(**overrides: object) -> RemoteNLIVerifier:
    kwargs: dict[str, object] = {
        "api_key": TOKEN,
        "timeout": 1.0,
        "dialect": "hf",
        "scale_up_timeout": 90.0,
    }
    kwargs.update(overrides)
    return RemoteNLIVerifier(ENDPOINT, "test/model", "abc123", **kwargs)  # type: ignore[arg-type]


def ok_row() -> list[dict]:
    return [
        [
            {"label": "entailment", "score": 0.97},
            {"label": "neutral", "score": 0.02},
            {"label": "contradiction", "score": 0.01},
        ]
    ]


class TestAColdEndpointIsWaitedFor:
    @respx.mock
    def test_the_documented_502_is_retried_until_it_answers(self) -> None:
        """The scale-to-zero case exactly: nothing there, then a replica."""
        route = respx.post(ENDPOINT)
        route.side_effect = [
            httpx.Response(502),
            httpx.Response(502),
            httpx.Response(502),
            httpx.Response(200, json=ok_row()),
        ]
        [prediction] = verifier().score(PAIRS)
        assert prediction.scores.entailment == pytest.approx(0.97)
        assert route.call_count == 4

    @respx.mock
    def test_a_503_is_also_treated_as_warming(self) -> None:
        route = respx.post(ENDPOINT)
        route.side_effect = [httpx.Response(503), httpx.Response(200, json=ok_row())]
        assert verifier().score(PAIRS)
        assert route.call_count == 2

    @respx.mock
    def test_a_refused_connection_while_booting_is_retried(self) -> None:
        route = respx.post(ENDPOINT)
        route.side_effect = [
            httpx.ConnectError("connection refused"),
            httpx.Response(200, json=ok_row()),
        ]
        assert verifier().score(PAIRS)
        assert route.call_count == 2


class TestWaitingIsBounded:
    @respx.mock
    def test_endless_warming_gives_up_at_the_deadline(self, _no_real_sleeping: list[float]) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(502))
        with pytest.raises(NLIUnavailable):
            verifier(scale_up_timeout=90.0).score(PAIRS)
        waited = sum(_no_real_sleeping)
        assert waited <= 90.0, f"waited {waited}s against a 90s budget"

    @respx.mock
    def test_backoff_is_capped_rather_than_doubling_forever(
        self, _no_real_sleeping: list[float]
    ) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(502))
        with pytest.raises(NLIUnavailable):
            verifier(scale_up_timeout=200.0).score(PAIRS)
        assert _no_real_sleeping, "no backoff happened at all"
        assert max(_no_real_sleeping) <= 15.0, (
            f"backoff grew to {max(_no_real_sleeping)}s between attempts"
        )

    @respx.mock
    def test_zero_budget_means_exactly_one_attempt(self) -> None:
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(502))
        with pytest.raises(NLIUnavailable):
            verifier(scale_up_timeout=0.0).score(PAIRS)
        assert route.call_count == 1


class TestOnlyWarmingIsRetried:
    """Everything here answers the same after another thirty seconds."""

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422, 429, 500, 504])
    @respx.mock
    def test_a_permanent_failure_is_not_retried(self, status: int) -> None:
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(status))
        with pytest.raises(NLIUnavailable):
            verifier(scale_up_timeout=90.0).score(PAIRS)
        assert route.call_count == 1, f"HTTP {status} was retried as if warming"

    @respx.mock
    def test_a_gateway_timeout_is_not_a_cold_start(self) -> None:
        """504 means the request was accepted and took too long, which
        is a wedged replica rather than a starting one."""
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(504))
        with pytest.raises(NLIUnavailable):
            verifier(scale_up_timeout=90.0).score(PAIRS)
        assert route.call_count == 1

    @respx.mock
    def test_a_malformed_success_fails_closed_without_retrying(self) -> None:
        route = respx.post(ENDPOINT).mock(
            return_value=httpx.Response(200, content=b"not json at all")
        )
        with pytest.raises(NLIUnavailable):
            verifier(scale_up_timeout=90.0).score(PAIRS)
        assert route.call_count == 1

    @respx.mock
    def test_wrong_labels_fail_closed_after_warming(self) -> None:
        """A woken endpoint serving the wrong thing is still wrong."""
        route = respx.post(ENDPOINT)
        route.side_effect = [
            httpx.Response(502),
            httpx.Response(
                200,
                json=[[{"label": "supported", "score": 0.9}, {"label": "no", "score": 0.1}]],
            ),
        ]
        with pytest.raises(NLIUnavailable, match="expected"):
            verifier().score(PAIRS)


class TestCancellation:
    @respx.mock
    def test_an_interrupted_warm_up_propagates_rather_than_withholding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancelled run must not be reported as an unverifiable
        claim: the caller stopped, the verifier did not fail."""

        def cancel(_seconds: float) -> None:
            raise KeyboardInterrupt

        # Replaces the fixture's clock-advancing sleep on purpose.
        monkeypatch.setattr(nli_module.time, "sleep", cancel)
        respx.post(ENDPOINT).mock(return_value=httpx.Response(502))
        with pytest.raises(KeyboardInterrupt):
            verifier(scale_up_timeout=90.0).score(PAIRS)


class TestNoSecretsEscape:
    @respx.mock
    def test_the_token_is_absent_from_every_error(self) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(401))
        with pytest.raises(NLIUnavailable) as caught:
            verifier(scale_up_timeout=90.0).score(PAIRS)
        assert TOKEN not in str(caught.value)
        assert TOKEN not in repr(caught.value)

    @respx.mock
    def test_the_token_is_absent_from_warm_up_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(502))
        with caplog.at_level("INFO"), pytest.raises(NLIUnavailable):
            verifier(scale_up_timeout=10.0).score(PAIRS)
        assert TOKEN not in caplog.text


class TestTheBlueprintMatchesTheMeasurement:
    def test_the_configured_budget_clears_the_measured_cold_start(self) -> None:
        """49.2s measured. A budget below it would fail a normal wake."""
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parents[2]
        spec = yaml.safe_load((root / "deploy" / "render-live.yaml").read_text())
        env = {e["key"]: e for e in spec["services"][0]["envVars"]}
        budget = float(env["NLI_SCALE_UP_TIMEOUT_SECONDS"]["value"])
        assert budget >= 49.2 * 1.5, (
            f"budget {budget}s leaves too little margin over the measured 49.2s"
        )

    def test_the_budget_leaves_room_inside_the_run_deadline(self) -> None:
        """The wake happens inside the run's wall-clock ceiling, so the
        two cannot be chosen independently."""
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parents[2]
        spec = yaml.safe_load((root / "deploy" / "render-live.yaml").read_text())
        env = {e["key"]: e for e in spec["services"][0]["envVars"]}
        budget = float(env["NLI_SCALE_UP_TIMEOUT_SECONDS"]["value"])
        deadline = float(env["DEMO_MAX_RUNTIME_SECONDS"]["value"])
        assert budget < deadline / 2, (
            f"a {budget}s wake inside a {deadline}s run leaves "
            f"{deadline - budget}s for the research itself"
        )
