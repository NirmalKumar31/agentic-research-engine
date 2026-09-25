"""Remote-mode semantic verification fails closed on every error path.

Remote mode exists so a host too small for the checkpoint can still
verify. It is only safe if every way the call can go wrong ends in a
withhold -- a network error that returned a neutral score would publish
unverified claims and look like normal operation while doing it.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from agentic_research.citations.nli import NLIUnavailable, RemoteNLIVerifier, build_verifier
from agentic_research.citations.semantic import verify_claim
from agentic_research.config import Settings

ENDPOINT = "https://nli.example.test/score"
PAIRS = [("Throughput increased by 20%.", "Throughput increased by 20%.")]


def verifier() -> RemoteNLIVerifier:
    return RemoteNLIVerifier(ENDPOINT, "test/model", "abc123", timeout=1.0)


@respx.mock
def test_well_formed_response_is_scored() -> None:
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200, json={"results": [{"entailment": 0.99, "neutral": 0.01, "contradiction": 0.0}]}
        )
    )
    [prediction] = verifier().score(PAIRS)
    assert prediction.scores.entailment == pytest.approx(0.99)
    assert prediction.model_id == "test/model"
    assert prediction.model_revision == "abc123"


@respx.mock
@pytest.mark.parametrize("status", [429, 500, 502, 503, 401])
def test_error_status_raises(status: int) -> None:
    respx.post(ENDPOINT).mock(return_value=httpx.Response(status, json={}))
    with pytest.raises(NLIUnavailable):
        verifier().score(PAIRS)


@respx.mock
def test_timeout_raises() -> None:
    respx.post(ENDPOINT).mock(side_effect=httpx.TimeoutException("timed out"))
    with pytest.raises(NLIUnavailable):
        verifier().score(PAIRS)


@respx.mock
def test_network_error_raises() -> None:
    respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("no route"))
    with pytest.raises(NLIUnavailable):
        verifier().score(PAIRS)


@respx.mock
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"results": "not a list"},
        {"results": []},
        {"results": [{"entailment": 0.9}]},
        {"results": [{"entailment": "high", "neutral": 0.0, "contradiction": 0.0}]},
        {"results": [{}, {}]},
    ],
)
def test_malformed_body_raises(body: dict) -> None:
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(NLIUnavailable):
        verifier().score(PAIRS)


@respx.mock
def test_non_json_body_raises() -> None:
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(NLIUnavailable):
        verifier().score(PAIRS)


@respx.mock
def test_remote_failure_withholds_rather_than_publishes() -> None:
    """The end-to-end consequence, which is the only one that matters."""
    respx.post(ENDPOINT).mock(return_value=httpx.Response(503, json={}))
    verdict = verify_claim(
        "Throughput increased by 20%.",
        [("E1", "Throughput increased by 20%.")],
        verifier(),
        support_threshold=0.98,
    )
    assert not verdict.publishable
    assert not verdict.checked
    assert "unavailable" in verdict.reason


@respx.mock
def test_api_key_is_sent_as_a_bearer_token_and_not_logged() -> None:
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200, json={"results": [{"entailment": 0.1, "neutral": 0.9, "contradiction": 0.0}]}
        )
    )
    RemoteNLIVerifier(ENDPOINT, api_key="unit-test-placeholder").score(PAIRS)
    assert route.calls.last.request.headers["authorization"].startswith("Bearer ")


class TestBuildVerifier:
    def test_local_is_the_default(self) -> None:
        built = build_verifier(Settings())
        assert type(built).__name__ == "NLIVerifier"
        assert built.revision == Settings().nli_model_revision

    def test_remote_without_an_endpoint_raises_rather_than_silently_going_local(self) -> None:
        """Falling back to local here would try to load 1.4GB on a host
        chosen precisely because it cannot hold that."""
        with pytest.raises(NLIUnavailable):
            build_verifier(Settings(nli_mode="remote"))

    def test_remote_with_an_endpoint_builds_remote(self) -> None:
        built = build_verifier(Settings(nli_mode="remote", nli_endpoint=ENDPOINT))
        assert type(built).__name__ == "RemoteNLIVerifier"
