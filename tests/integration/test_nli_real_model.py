"""The release gate: the adversarial suite against the real checkpoint.

Marked ``nli`` and excluded from default runs, because it downloads and
loads a 1.4GB model. CI runs it in a dedicated cached job, and it must
pass before release -- the fake-scorer tests prove the wiring and say
nothing about whether the classifier actually refuses an overclaim.

Run:  pytest -m nli
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_research.citations.nli import NLIVerifier
from agentic_research.citations.semantic import verify_claim
from agentic_research.config import Settings

pytestmark = pytest.mark.nli

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "semantic_stress.json"


@pytest.fixture(scope="module")
def gate() -> tuple[NLIVerifier, float]:
    settings = Settings()
    return (
        NLIVerifier(settings.nli_model_id, settings.nli_model_revision),
        settings.nli_support_threshold,
    )


def _cases() -> list[dict]:
    return json.loads(FIXTURE.read_text())["cases"]


def test_no_adversarial_case_is_published(gate: tuple[NLIVerifier, float]) -> None:
    """Zero unsafe publishes. This is the acceptance condition; a single
    overclaim reaching a report is the failure the project exists to
    prevent, so it is asserted absolutely rather than as a rate."""
    verifier, threshold = gate
    leaked = []
    for case in _cases():
        if case["expected"] != "withhold":
            continue
        verdict = verify_claim(
            case["hypothesis"], [("E1", case["premise"])], verifier, support_threshold=threshold
        )
        if verdict.publishable:
            leaked.append((case["id"], case["category"], round(verdict.best_entailment, 4)))
    assert not leaked, f"overclaims published at threshold {threshold}: {leaked}"


def test_pinned_revision_reports_the_expected_label_order(
    gate: tuple[NLIVerifier, float],
) -> None:
    """The two cross-encoder candidates put contradiction at index 0 and
    this one puts entailment there. Reading the order from the config is
    what keeps the gate from being exactly inverted, so the read itself
    is asserted rather than assumed."""
    verifier, _ = gate
    [entailed] = verifier.score([("The build failed.", "The build failed.")])
    [contradicted] = verifier.score([("The build failed.", "The build succeeded.")])
    assert entailed.scores.entailment > 0.9
    assert contradicted.scores.contradiction > 0.5


def test_faithful_restatement_is_not_withheld(gate: tuple[NLIVerifier, float]) -> None:
    verifier, threshold = gate
    quote = "Throughput increased by 20% after the change."
    verdict = verify_claim(
        "Throughput increased by 20% after the change.",
        [("E1", quote)],
        verifier,
        support_threshold=threshold,
    )
    assert verdict.publishable, verdict.reason
