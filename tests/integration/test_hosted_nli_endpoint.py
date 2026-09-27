"""Parity against the real pinned endpoint.

Skipped unless NLI_TEST_ENDPOINT and NLI_TEST_TOKEN are set, because it
calls a live private endpoint that bills by the hour. Hermetic CI uses
mocked transports; this is the only test that proves the deployed
endpoint behaves like the checkpoint the threshold was calibrated
against.

    NLI_TEST_ENDPOINT=https://<id>.endpoints.huggingface.cloud \
    NLI_TEST_TOKEN=hf_... \
    NLI_TEST_DIALECT=hf pytest -m hosted_nli

NLI_TEST_DIALECT selects the wire format: "hf" for a managed Inference
Endpoint on the stock handler, "contract" for this project's own
service in deploy/nli-service.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentic_research.citations.nli import RemoteNLIVerifier
from agentic_research.citations.nli_endpoint_meta import verify_endpoint_pin
from agentic_research.citations.nli_pin import NLI_DEFAULT_MODEL_ID, NLI_DEFAULT_REVISION

pytestmark = pytest.mark.hosted_nli

ENDPOINT = os.environ.get("NLI_TEST_ENDPOINT")
TOKEN = os.environ.get("NLI_TEST_TOKEN")
DIALECT = os.environ.get("NLI_TEST_DIALECT", "hf")
CALIBRATION = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "verifier-calibration"
    / "nli-deberta-v3-large-scores.json"
)
BLIND = CALIBRATION.parent / "blind_cases.json"

# The committed local scores were produced in float16 -- the checkpoint
# ships fp16 weights and the local verifier loads them as they are, so
# all 50 lie exactly on the fp16 grid. The managed endpoint runs the
# same commit in float32. The gap is therefore fp16 error accumulated
# through 24 layers, not endpoint inaccuracy: of the two numbers the
# remote one is the more faithful.
#
# Measured across the 50 calibration pairs: max 9.43e-3, mean 4.27e-4,
# 45 of 50 within 1e-3. The bound below sits above the measured maximum
# with room to spare, and is a regression detector rather than a claim
# about precision -- a redeployed endpoint serving different weights
# would blow through it by orders of magnitude.
SCORE_TOLERANCE = 1.5e-2
THRESHOLD = 0.98


@pytest.fixture(scope="module")
def verifier() -> RemoteNLIVerifier:
    if not ENDPOINT or not TOKEN:
        pytest.skip("set NLI_TEST_ENDPOINT and NLI_TEST_TOKEN to run hosted parity")
    return RemoteNLIVerifier(
        ENDPOINT,
        NLI_DEFAULT_MODEL_ID,
        NLI_DEFAULT_REVISION,
        api_key=TOKEN,
        timeout=120,
        dialect=DIALECT,
        # A scaled-to-zero endpoint takes about half a minute to wake,
        # and the first call in this suite is the one that pays it.
        scale_up_timeout=180.0,
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
        """What this proves depends on the dialect.

        Under "contract" it is a real check: the service echoes what
        served the request and the client refuses a mismatch, so
        reaching the assertion means the echo agreed.

        Under "hf" it is not. The stock handler reports no model or
        revision, so these fields are copied from configuration and the
        assertion cannot fail. The equivalent guarantee is
        test_control_plane_confirms_the_pin below, which asks Hugging
        Face rather than the inference path.
        """
        [prediction] = verifier.score([("The build succeeded.", "The build succeeded.")])
        assert prediction.model_revision == NLI_DEFAULT_REVISION
        assert prediction.model_id == NLI_DEFAULT_MODEL_ID

    def test_control_plane_confirms_the_pin(self) -> None:
        """The check that actually binds a managed endpoint.

        Without it, an endpoint edited to a newer revision would keep
        answering with a clean three-way distribution and silently
        invalidate the 0.98 threshold.
        """
        if not ENDPOINT or not TOKEN:
            pytest.skip("set NLI_TEST_ENDPOINT and NLI_TEST_TOKEN to run hosted parity")
        if DIALECT != "hf":
            pytest.skip("control-plane pin applies to managed endpoints only")
        pin = verify_endpoint_pin(ENDPOINT, TOKEN, NLI_DEFAULT_MODEL_ID, NLI_DEFAULT_REVISION)
        assert pin.verified, pin.detail
        assert pin.revision == NLI_DEFAULT_REVISION
        assert pin.repository == NLI_DEFAULT_MODEL_ID
        assert pin.task == "text-classification", (
            "zero-shot normalises over two labels, not the three the threshold was calibrated on"
        )


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
        """Bounded drift, not equality. See SCORE_TOLERANCE for why the
        two sides are not expected to agree exactly."""
        pairs = calibration_pairs()
        got = verifier.score([(p, h) for p, h, _ in pairs])
        worst = max(abs(g.scores.entailment - e) for (_p, _h, e), g in zip(pairs, got, strict=True))
        assert worst < SCORE_TOLERANCE, f"largest divergence {worst:.2e}"

    def test_the_endpoint_does_not_drift(self, verifier: RemoteNLIVerifier) -> None:
        """The property that makes a thin threshold margin survivable.

        One calibration pair sits 0.0005 above 0.98 in fp16, which is a
        single fp16 step, and the fp16-to-fp32 gap reaches 9.4e-3. That
        would be alarming if the endpoint wandered. It does not: repeat
        runs are bit-identical, so a pair that publishes today publishes
        tomorrow for the same reason.
        """
        pairs = calibration_pairs()[:12]
        inputs = [(p, h) for p, h, _ in pairs]
        first = verifier.score(inputs)
        second = verifier.score(inputs)
        drift = [
            abs(a.scores.entailment - b.scores.entailment)
            for a, b in zip(first, second, strict=True)
        ]
        assert max(drift) == 0.0, f"endpoint is not deterministic: max drift {max(drift):.2e}"

    def test_batching_does_not_change_scores(self, verifier: RemoteNLIVerifier) -> None:
        """Padding within a batch is the usual source of drift in
        sequence-pair scoring, and a score that depends on which other
        claims happened to be in flight is not reproducible."""
        pairs = calibration_pairs()[:8]
        inputs = [(p, h) for p, h, _ in pairs]
        verifier.batch_size = 8
        batched = verifier.score(inputs)
        verifier.batch_size = 1
        singly = verifier.score(inputs)
        verifier.batch_size = 8
        drift = [
            abs(a.scores.entailment - b.scores.entailment)
            for a, b in zip(batched, singly, strict=True)
        ]
        assert max(drift) == 0.0, f"batch size changes scores: max {max(drift):.2e}"


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
