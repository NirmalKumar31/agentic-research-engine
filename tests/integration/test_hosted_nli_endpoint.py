"""Parity against the real pinned endpoint.

Skipped unless NLI_TEST_ENDPOINT and NLI_TEST_TOKEN are set, because it
calls a live private endpoint that bills by the hour. Hermetic CI uses
mocked transports; this is the only test that proves the deployed
endpoint behaves like the checkpoint the threshold was calibrated
against.

    NLI_TEST_ENDPOINT=https://<id>.endpoints.huggingface.cloud/score \
    NLI_TEST_TOKEN=hf_... pytest -m hosted_nli
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentic_research.citations.nli import RemoteNLIVerifier
from agentic_research.citations.nli_pin import NLI_DEFAULT_MODEL_ID, NLI_DEFAULT_REVISION

pytestmark = pytest.mark.hosted_nli

ENDPOINT = os.environ.get("NLI_TEST_ENDPOINT")
TOKEN = os.environ.get("NLI_TEST_TOKEN")
CALIBRATION = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "verifier-calibration"
    / "nli-deberta-v3-large-scores.json"
)
BLIND = CALIBRATION.parent / "blind_cases.json"

# Scores come from a different machine and runtime, so exact equality is
# the wrong test. What must hold is that no publish/withhold decision
# changes at the threshold; the numeric tolerance is a secondary signal.
SCORE_TOLERANCE = 1e-3
THRESHOLD = 0.98


@pytest.fixture(scope="module")
def verifier() -> RemoteNLIVerifier:
    if not ENDPOINT or not TOKEN:
        pytest.skip("set NLI_TEST_ENDPOINT and NLI_TEST_TOKEN to run hosted parity")
    return RemoteNLIVerifier(
        ENDPOINT, NLI_DEFAULT_MODEL_ID, NLI_DEFAULT_REVISION, api_key=TOKEN, timeout=120
    )


def calibration_pairs() -> list[tuple[str, str, float]]:
    scores = json.loads(CALIBRATION.read_text())["scores"]
    blind = {c["case_id"]: c for c in json.loads(BLIND.read_text())["cases"]}
    out = []
    for case in scores:
        quotes = {e["evidence_id"]: e["quote"] for e in blind[case["case_id"]]["evidence"]}
        for entry in case["evidence"]:
            out.append((quotes[entry["evidence_id"]], case["claim"], entry["entailment"]))
    return out


class TestIdentity:
    def test_it_serves_the_pinned_revision(self, verifier: RemoteNLIVerifier) -> None:
        """A different checkpoint is a different verifier, and the
        client refuses it. Reaching this assertion means it did not."""
        [prediction] = verifier.score([("The build succeeded.", "The build succeeded.")])
        assert prediction.model_revision == NLI_DEFAULT_REVISION
        assert prediction.model_id == NLI_DEFAULT_MODEL_ID


class TestKnownPairs:
    def test_entailment(self, verifier: RemoteNLIVerifier) -> None:
        [p] = verifier.score([("The build completed successfully.", "The build succeeded.")])
        assert p.scores.entailment > 0.5, p.scores

    def test_contradiction(self, verifier: RemoteNLIVerifier) -> None:
        """Also catches an inverted label mapping, which would publish
        exactly the claims that should be withheld."""
        [p] = verifier.score([("The build completed successfully.", "The build failed.")])
        assert p.scores.contradiction > 0.5, p.scores

    def test_scores_are_a_distribution(self, verifier: RemoteNLIVerifier) -> None:
        [p] = verifier.score([("A premise.", "A hypothesis.")])
        assert p.scores.is_distribution()


class TestCalibrationParity:
    def test_no_publish_decision_changes(self, verifier: RemoteNLIVerifier) -> None:
        """The release property. Raw scores may drift slightly across
        runtimes; which claims publish must not."""
        pairs = calibration_pairs()
        got = verifier.score([(p, h) for p, h, _ in pairs])
        changed = [
            (i, expected, g.scores.entailment)
            for i, ((_p, _h, expected), g) in enumerate(zip(pairs, got, strict=True))
            if (expected >= THRESHOLD) != (g.scores.entailment >= THRESHOLD)
        ]
        assert not changed, f"{len(changed)} publish decisions differ from local: {changed[:5]}"

    def test_scores_are_numerically_close(self, verifier: RemoteNLIVerifier) -> None:
        pairs = calibration_pairs()
        got = verifier.score([(p, h) for p, h, _ in pairs])
        worst = max(abs(g.scores.entailment - e) for (_p, _h, e), g in zip(pairs, got, strict=True))
        assert worst < SCORE_TOLERANCE, f"largest divergence {worst:.2e}"


class TestAdversarialSafety:
    def test_no_new_unsafe_publish(self, verifier: RemoteNLIVerifier) -> None:
        """Safety must not regress against the accepted local baseline."""
        from agentic_research.citations.semantic import verify_claim

        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "semantic_stress.json"
        cases = json.loads(fixture.read_text())["cases"]
        leaked = [
            case["id"]
            for case in cases
            if case["expected"] == "withhold"
            and verify_claim(
                case["hypothesis"],
                [("E1", case["premise"])],
                verifier,
                support_threshold=THRESHOLD,
            ).publishable
        ]
        assert not leaked, f"overclaims published by the hosted endpoint: {leaked}"
