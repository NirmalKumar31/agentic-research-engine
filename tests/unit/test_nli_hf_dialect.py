"""The managed-endpoint dialect fails closed on every malformed answer.

The project's own scoring service proves what served each request. A
managed Hugging Face endpoint on the stock text-classification handler
proves nothing: it returns labels and scores, in an order it chooses,
with no model id, no revision, no per-pair id and no truncation flag.

Every guarantee that dialect cannot provide has to be recovered
somewhere, and these tests pin down where. Order is handled by reading
labels by name. Truncation is measured in this process. Identity moves
to preflight, against the control plane. What is left is the set of
malformed responses that must withhold rather than publish.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from agentic_research.citations import nli as nli_module
from agentic_research.citations.nli import NLIUnavailable, RemoteNLIVerifier, build_verifier
from agentic_research.config import LLMMode, Settings

ENDPOINT = "https://nli.example.test"
PAIRS = [("Throughput increased by 20%.", "Throughput increased by 20%.")]


class _Encoding:
    def __init__(self, count: int) -> None:
        self.ids = [0] * count


class _Tokenizer:
    """Stands in for the pinned tokenizer, with a settable length."""

    def __init__(self, count: int = 10) -> None:
        self.count = count

    def encode(self, premise: str, hypothesis: str) -> _Encoding:
        return _Encoding(self.count)


@pytest.fixture(autouse=True)
def _stub_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """No network and no 8MB download in unit tests."""
    monkeypatch.setattr(nli_module, "load_pair_tokenizer", lambda *a, **k: _Tokenizer())


def verifier(**overrides: object) -> RemoteNLIVerifier:
    kwargs: dict[str, object] = {
        "api_key": "token",
        "timeout": 1.0,
        "dialect": "hf",
    }
    kwargs.update(overrides)
    return RemoteNLIVerifier(ENDPOINT, "test/model", "abc123", **kwargs)  # type: ignore[arg-type]


def row(entailment: float, neutral: float, contradiction: float) -> list[dict]:
    """A row in the order the endpoint actually returns: by score."""
    scored = [
        {"label": "entailment", "score": entailment},
        {"label": "neutral", "score": neutral},
        {"label": "contradiction", "score": contradiction},
    ]
    return sorted(scored, key=lambda d: d["score"], reverse=True)


class TestLabelHandling:
    @respx.mock
    def test_labels_are_read_by_name_not_position(self) -> None:
        """The endpoint sorts each row by descending score, so the first
        entry is whichever label won. Reading position 0 as entailment
        would report a flat contradiction as near-certain support."""
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=[row(0.01, 0.04, 0.95)]))
        [prediction] = verifier().score(PAIRS)
        assert prediction.scores.contradiction == pytest.approx(0.95)
        assert prediction.scores.entailment == pytest.approx(0.01)

    @respx.mock
    def test_uppercase_labels_are_accepted(self) -> None:
        respx.post(ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json=[
                    [
                        {"label": "ENTAILMENT", "score": 0.97},
                        {"label": "Neutral", "score": 0.02},
                        {"label": "CONTRADICTION", "score": 0.01},
                    ]
                ],
            )
        )
        [prediction] = verifier().score(PAIRS)
        assert prediction.scores.entailment == pytest.approx(0.97)

    @respx.mock
    def test_a_missing_label_is_refused(self) -> None:
        respx.post(ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json=[[{"label": "entailment", "score": 0.9}, {"label": "neutral", "score": 0.1}]],
            )
        )
        with pytest.raises(NLIUnavailable, match="expected"):
            verifier().score(PAIRS)

    @respx.mock
    def test_an_unexpected_label_is_refused(self) -> None:
        """A zero-shot endpoint answers with the caller's own class
        names. Its two-way normalisation is not the distribution the
        0.98 threshold was calibrated against."""
        respx.post(ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json=[
                    [
                        {"label": "supported", "score": 0.8},
                        {"label": "unsupported", "score": 0.2},
                    ]
                ],
            )
        )
        with pytest.raises(NLIUnavailable, match="expected"):
            verifier().score(PAIRS)

    @respx.mock
    def test_a_repeated_label_is_refused(self) -> None:
        respx.post(ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json=[
                    [
                        {"label": "entailment", "score": 0.6},
                        {"label": "entailment", "score": 0.3},
                        {"label": "neutral", "score": 0.1},
                    ]
                ],
            )
        )
        with pytest.raises(NLIUnavailable, match="repeats label"):
            verifier().score(PAIRS)


class TestShape:
    @respx.mock
    def test_the_request_sends_a_text_pair(self) -> None:
        """Sending the two sides as a bare list instead of a text pair
        is answered with HTTP 200 and two unrelated single-text scores,
        so the mistake never surfaces as an error."""
        route = respx.post(ENDPOINT).mock(
            return_value=httpx.Response(200, json=[row(0.99, 0.01, 0.0)])
        )
        verifier().score(PAIRS)
        body = route.calls[0].request.content
        import json

        payload = json.loads(body)
        assert payload["inputs"] == [{"text": PAIRS[0][0], "text_pair": PAIRS[0][1]}]
        assert payload["parameters"]["top_k"] is None

    @respx.mock
    def test_a_top_label_only_response_is_refused(self) -> None:
        """Without top_k the handler returns one dict per pair rather
        than a row of labels. One score cannot be checked against a
        threshold that needs the distribution."""
        respx.post(ENDPOINT).mock(
            return_value=httpx.Response(200, json=[{"label": "entailment", "score": 0.99}])
        )
        with pytest.raises(NLIUnavailable, match="top label"):
            verifier().score(PAIRS)

    @respx.mock
    def test_a_row_count_mismatch_is_refused(self) -> None:
        """Results are matched positionally in this dialect, so a
        differing count means the mapping from claims to scores is
        unknown."""
        respx.post(ENDPOINT).mock(
            return_value=httpx.Response(200, json=[row(0.9, 0.1, 0.0), row(0.1, 0.9, 0.0)])
        )
        with pytest.raises(NLIUnavailable, match="rows for"):
            verifier().score(PAIRS)

    @respx.mock
    def test_a_non_list_body_is_refused(self) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"error": "nope"}))
        with pytest.raises(NLIUnavailable, match="not a list"):
            verifier().score(PAIRS)

    @respx.mock
    def test_scores_that_are_not_a_distribution_are_refused(self) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=[row(1.0, 1.0, 1.0)]))
        with pytest.raises(NLIUnavailable, match="distribution"):
            verifier().score(PAIRS)


class TestTruncation:
    @respx.mock
    def test_an_overlong_pair_is_marked_truncated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The handler cuts at 512 tokens and says nothing about it. A
        premise cut mid-way is a different premise, and the clause that
        reverses a claim is usually the one that gets dropped."""
        monkeypatch.setattr(nli_module, "load_pair_tokenizer", lambda *a, **k: _Tokenizer(513))
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=[row(0.99, 0.01, 0.0)]))
        [prediction] = verifier().score(PAIRS)
        assert prediction.truncated is True

    @respx.mock
    def test_a_short_pair_is_not_marked_truncated(self) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=[row(0.99, 0.01, 0.0)]))
        [prediction] = verifier().score(PAIRS)
        assert prediction.truncated is False

    @respx.mock
    def test_an_unavailable_tokenizer_withholds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not knowing whether the premise was cut is not the same as
        knowing it was not."""

        def boom(*a: object, **k: object) -> object:
            raise nli_module.TokenizerUnavailable("no tokenizer")

        monkeypatch.setattr(nli_module, "load_pair_tokenizer", boom)
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=[row(0.99, 0.01, 0.0)]))
        with pytest.raises(NLIUnavailable, match="truncation"):
            verifier().score(PAIRS)


class TestWarmUp:
    @respx.mock
    def test_a_booting_endpoint_is_waited_for(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An endpoint scaled to zero answers 502 while its replica
        starts. Failing there would withhold every claim in the first
        run after an idle period."""
        monkeypatch.setattr(nli_module.time, "sleep", lambda _s: None)
        route = respx.post(ENDPOINT)
        route.side_effect = [
            httpx.Response(502),
            httpx.Response(503),
            httpx.Response(200, json=[row(0.99, 0.01, 0.0)]),
        ]
        [prediction] = verifier(scale_up_timeout=60.0).score(PAIRS)
        assert prediction.scores.entailment == pytest.approx(0.99)
        assert route.call_count == 3

    @respx.mock
    def test_waiting_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = iter([0.0, 0.0, 100.0, 100.0, 100.0])
        monkeypatch.setattr(nli_module.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(nli_module.time, "sleep", lambda _s: None)
        respx.post(ENDPOINT).mock(return_value=httpx.Response(502))
        with pytest.raises(NLIUnavailable):
            verifier(scale_up_timeout=30.0).score(PAIRS)

    @respx.mock
    def test_no_waiting_by_default(self) -> None:
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(502))
        with pytest.raises(NLIUnavailable):
            verifier().score(PAIRS)
        assert route.call_count == 1

    @respx.mock
    def test_a_rejected_token_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 401 answers the same after thirty more seconds. Retrying it
        spends the warm-up window learning nothing."""
        monkeypatch.setattr(nli_module.time, "sleep", lambda _s: None)
        route = respx.post(ENDPOINT).mock(return_value=httpx.Response(401))
        with pytest.raises(NLIUnavailable):
            verifier(scale_up_timeout=60.0).score(PAIRS)
        assert route.call_count == 1


class TestWiring:
    def test_an_unknown_dialect_is_refused(self) -> None:
        with pytest.raises(NLIUnavailable, match="unknown remote dialect"):
            RemoteNLIVerifier(ENDPOINT, "test/model", "abc123", dialect="zero-shot")

    def test_settings_select_the_dialect(self) -> None:
        settings = Settings(
            llm_mode=LLMMode.LOCAL,
            nli_mode="remote",
            nli_endpoint=ENDPOINT,
            nli_api_key="token",
            nli_dialect="hf",
            _env_file=None,
        )  # type: ignore[arg-type]
        built = build_verifier(settings)
        assert isinstance(built, RemoteNLIVerifier)
        assert built.dialect == "hf"

    def test_the_contract_dialect_remains_the_default(self) -> None:
        """_env_file=None on purpose. A developer running the live
        endpoint has NLI_DIALECT=hf in .env, and without this the test
        would read their configuration and assert nothing."""
        settings = Settings(
            llm_mode=LLMMode.LOCAL,
            nli_mode="remote",
            nli_endpoint=ENDPOINT,
            nli_api_key="token",
            _env_file=None,
        )  # type: ignore[arg-type]
        built = build_verifier(settings)
        assert isinstance(built, RemoteNLIVerifier)
        assert built.dialect == "contract"
