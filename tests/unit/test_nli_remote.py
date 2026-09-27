"""Remote-mode semantic verification fails closed on every error path.

Remote mode exists so a host too small for the checkpoint can still
verify. It is only safe if every way the call can go wrong ends in a
withhold -- a network error that returned a neutral score would publish
unverified claims and look like normal operation while doing it.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from agentic_research.citations.nli import NLIUnavailable, RemoteNLIVerifier, build_verifier
from agentic_research.citations.semantic import verify_claim
from agentic_research.config import LLMMode, Settings

ENDPOINT = "https://nli.example.test/score"


def local_settings(**overrides: object) -> Settings:
    """Settings that do not depend on a developer's .env.

    The default LLM_MODE is hybrid, which requires an API key. A local
    checkout supplies one through .env and CI does not, so a bare
    Settings() passes here and fails there -- which is exactly how this
    reached CI green locally and red remotely.
    """
    return Settings(llm_mode=LLMMode.LOCAL, **overrides)  # type: ignore[arg-type]


def well_formed(results: list[dict]) -> dict:
    """A response satisfying the remote contract.

    The endpoint must state which checkpoint served the request and
    whether each pair was truncated. Neither is inferred: a silently
    redeployed remote answering with a different model would otherwise
    publish claims against a threshold never measured for it.
    """
    return {
        "contract_version": 1,
        "model_id": "test/model",
        "model_revision": "abc123",
        "results": [{"pair_id": str(i), "truncated": False, **r} for i, r in enumerate(results)],
    }


PAIRS = [("Throughput increased by 20%.", "Throughput increased by 20%.")]


def verifier() -> RemoteNLIVerifier:
    return RemoteNLIVerifier(ENDPOINT, "test/model", "abc123", timeout=1.0)


@respx.mock
def test_well_formed_response_is_scored() -> None:
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200, json=well_formed([{"entailment": 0.99, "neutral": 0.01, "contradiction": 0.0}])
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
        # contract metadata missing entirely
        {"results": [{"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}]},
        # served by a different checkpoint than the one configured
        {
            "contract_version": 1,
            "model_id": "other/model",
            "model_revision": "abc123",
            "results": [
                {
                    "pair_id": "0",
                    "entailment": 1.0,
                    "neutral": 0.0,
                    "contradiction": 0.0,
                    "truncated": False,
                }
            ],
        },
        {
            "contract_version": 1,
            "model_id": "test/model",
            "model_revision": "deadbee",
            "results": [
                {
                    "pair_id": "0",
                    "entailment": 1.0,
                    "neutral": 0.0,
                    "contradiction": 0.0,
                    "truncated": False,
                }
            ],
        },
        # truncation status omitted: must not be assumed false
        {
            "contract_version": 1,
            "model_id": "test/model",
            "model_revision": "abc123",
            "results": [{"pair_id": "0", "entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}],
        },
        # not a probability distribution
        {
            "contract_version": 1,
            "model_id": "test/model",
            "model_revision": "abc123",
            "results": [
                {
                    "pair_id": "0",
                    "entailment": 1.0,
                    "neutral": 1.0,
                    "contradiction": 1.0,
                    "truncated": False,
                }
            ],
        },
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
            200, json=well_formed([{"entailment": 0.1, "neutral": 0.9, "contradiction": 0.0}])
        )
    )
    RemoteNLIVerifier(ENDPOINT, "test/model", "abc123", api_key="unit-test-placeholder").score(
        PAIRS
    )
    assert route.calls.last.request.headers["authorization"].startswith("Bearer ")


class TestBuildVerifier:
    def test_local_is_the_default(self) -> None:
        built = build_verifier(local_settings())
        assert type(built).__name__ == "NLIVerifier"
        assert built.revision == local_settings().nli_model_revision

    def test_remote_without_an_endpoint_raises_rather_than_silently_going_local(self) -> None:
        """Falling back to local here would try to load 1.4GB on a host
        chosen precisely because it cannot hold that."""
        with pytest.raises(NLIUnavailable):
            build_verifier(local_settings(nli_mode="remote"))

    def test_remote_with_an_endpoint_builds_remote(self) -> None:
        built = build_verifier(local_settings(nli_mode="remote", nli_endpoint=ENDPOINT))
        assert type(built).__name__ == "RemoteNLIVerifier"


GOOD = {"entailment": 0.99, "neutral": 0.01, "contradiction": 0.0, "truncated": False}


def envelope(results: list[dict], **over: object) -> dict:
    return {
        "contract_version": 1,
        "model_id": "test/model",
        "model_revision": "abc123",
        "results": results,
        **over,
    }


class TestResultsAreMatchedByIdNotPosition:
    """A reordered response must not attach one claim's score to another.

    Results used to be zipped against the request by list position. A
    remote that reorders, drops or duplicates them would have produced
    confident, well-formed, wrong verdicts -- the worst failure shape
    available, because nothing downstream could detect it.
    """

    PAIRS = [("premise one", "hypothesis one"), ("premise two", "hypothesis two")]

    @respx.mock
    def test_ids_are_sent_with_every_pair(self) -> None:
        route = respx.post(ENDPOINT).mock(
            return_value=httpx.Response(
                200, json=envelope([{"pair_id": "0", **GOOD}, {"pair_id": "1", **GOOD}])
            )
        )
        verifier().score(self.PAIRS)
        sent = json.loads(route.calls.last.request.content)
        assert [p["pair_id"] for p in sent["pairs"]] == ["0", "1"]
        assert sent["contract_version"] == 1

    @respx.mock
    def test_reordered_results_are_reassociated_correctly(self) -> None:
        """Returned out of order, matched back by id."""
        respx.post(ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json=envelope(
                    [
                        {
                            "pair_id": "1",
                            "entailment": 0.10,
                            "neutral": 0.90,
                            "contradiction": 0.0,
                            "truncated": False,
                        },
                        {
                            "pair_id": "0",
                            "entailment": 0.95,
                            "neutral": 0.05,
                            "contradiction": 0.0,
                            "truncated": False,
                        },
                    ]
                ),
            )
        )
        out = verifier().score(self.PAIRS)
        assert out[0].premise == "premise one"
        assert out[0].scores.entailment == pytest.approx(0.95)
        assert out[1].scores.entailment == pytest.approx(0.10)

    @respx.mock
    @pytest.mark.parametrize(
        ("results", "why"),
        [
            ([{**GOOD}, {"pair_id": "1", **GOOD}], "a result without an id"),
            ([{"pair_id": "0", **GOOD}, {"pair_id": "0", **GOOD}], "duplicate ids"),
            ([{"pair_id": "0", **GOOD}, {"pair_id": "7", **GOOD}], "an unknown id"),
            ([{"pair_id": "0", **GOOD}], "fewer results than pairs"),
        ],
    )
    def test_broken_association_withholds(self, results: list[dict], why: str) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=envelope(results)))
        with pytest.raises(NLIUnavailable):
            verifier().score(self.PAIRS)


class TestContractVersionIsEnforcedByValue:
    """Presence was checked; the value was not.

    An endpoint speaking a different wire version is not compatible,
    and accepting it because the field merely exists is how a silent
    protocol change becomes a silent scoring change.
    """

    @respx.mock
    @pytest.mark.parametrize("version", [0, 2, 99, "1"])
    def test_unsupported_contract_version_withholds(self, version: object) -> None:
        respx.post(ENDPOINT).mock(
            return_value=httpx.Response(
                200, json=envelope([{"pair_id": "0", **GOOD}], contract_version=version)
            )
        )
        with pytest.raises(NLIUnavailable):
            verifier().score(PAIRS)

    @respx.mock
    def test_supported_version_is_accepted(self) -> None:
        respx.post(ENDPOINT).mock(
            return_value=httpx.Response(200, json=envelope([{"pair_id": "0", **GOOD}]))
        )
        assert len(verifier().score(PAIRS)) == 1


class TestThePinIsSingleSourced:
    def test_settings_and_adapter_agree(self) -> None:
        """Two copies drifted once: the adapter's said "main"."""
        from agentic_research.citations import nli
        from agentic_research.config import LLMMode, Settings

        settings = Settings(llm_mode=LLMMode.LOCAL)
        assert settings.nli_model_revision == nli.DEFAULT_REVISION
        assert settings.nli_model_id == nli.DEFAULT_MODEL_ID

    def test_there_is_no_unpinned_default(self) -> None:
        from agentic_research.citations import nli

        assert nli.DEFAULT_REVISION != "main"
        assert len(nli.DEFAULT_REVISION) == 40
