"""The publication decision: fail-closed behaviour and guard interaction.

Scores come from a deterministic fake, so these test the gate's wiring
rather than any model's semantics. Nothing here measures calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_research.citations.fake_nli import BrokenScorer, FakeScorer
from agentic_research.citations.semantic import (
    PARTIALLY_SUPPORTED,
    SUPPORTED,
    UNSUPPORTED,
    verify_claim,
)

CLAIM = "Throughput increased by 20%."
QUOTE = "Throughput increased by 20% after the change."


def entailing(entailment: float = 0.99) -> FakeScorer:
    return FakeScorer({(QUOTE, CLAIM): (entailment, 1 - entailment, 0.0)})


class TestPublication:
    def test_entailment_above_threshold_publishes(self) -> None:
        verdict = verify_claim(CLAIM, [("E1", QUOTE)], entailing(), support_threshold=0.98)
        assert verdict.publishable
        assert verdict.verdict == SUPPORTED
        assert verdict.best_evidence_id == "E1"

    def test_entailment_below_threshold_withholds(self) -> None:
        verdict = verify_claim(CLAIM, [("E1", QUOTE)], entailing(0.97), support_threshold=0.98)
        assert not verdict.publishable
        assert verdict.checked
        assert "below the 0.98 support threshold" in verdict.reason

    def test_threshold_is_inclusive(self) -> None:
        verdict = verify_claim(CLAIM, [("E1", QUOTE)], entailing(0.98), support_threshold=0.98)
        assert verdict.publishable

    def test_strongest_evidence_carries_the_claim(self) -> None:
        """One quote out of several is enough. Quotes are never merged --
        joining unrelated snippets until they collectively imply
        something is the reasoning this gate exists to prevent."""
        other = "Unrelated background about the deployment."
        scorer = FakeScorer({(QUOTE, CLAIM): (0.99, 0.01, 0.0), (other, CLAIM): (0.01, 0.99, 0.0)})
        verdict = verify_claim(
            CLAIM, [("E1", other), ("E2", QUOTE)], scorer, support_threshold=0.98
        )
        assert verdict.publishable
        assert verdict.best_evidence_id == "E2"
        assert len(verdict.per_evidence) == 2

    def test_pairs_are_scored_in_one_batched_call(self) -> None:
        scorer = entailing()
        verify_claim(CLAIM, [("E1", QUOTE), ("E2", QUOTE)], scorer, support_threshold=0.98)
        assert scorer.calls == 1


class TestGuardsOverrideScore:
    def test_guard_failure_withholds_a_confidently_entailed_claim(self) -> None:
        """The whole point of the guards: a 0.99 entailment on a claim
        whose numbers do not match the quote still does not publish."""
        claim = "Throughput increased by 75%."
        scorer = FakeScorer({(QUOTE, claim): (0.99, 0.01, 0.0)})
        verdict = verify_claim(claim, [("E1", QUOTE)], scorer, support_threshold=0.98)
        assert not verdict.publishable
        assert "numeric" in verdict.reason
        assert verdict.per_evidence[0].failed_guard_names == ["numeric"]

    def test_a_guard_failing_item_cannot_be_the_best_evidence(self) -> None:
        """A high score on a quote the guards rejected must not be
        promoted over a lower score on a quote they accepted."""
        claim = "Throughput increased by 20%."
        bad = "Throughput increased by 75%."
        scorer = FakeScorer({(bad, claim): (0.999, 0.001, 0.0), (QUOTE, claim): (0.5, 0.5, 0.0)})
        verdict = verify_claim(claim, [("E1", bad), ("E2", QUOTE)], scorer, support_threshold=0.98)
        assert not verdict.publishable
        assert verdict.best_evidence_id == "E2"


class TestFailClosed:
    def test_no_evidence_withholds_and_is_unchecked(self) -> None:
        verdict = verify_claim(CLAIM, [], entailing(), support_threshold=0.98)
        assert not verdict.publishable
        assert not verdict.checked
        assert "no citable evidence" in verdict.reason

    def test_model_unavailable_withholds(self) -> None:
        """Never a fallback to another verifier: unavailable means
        unverified, and unverified must not reach the report."""
        scorer = FakeScorer(fail_with="could not load checkpoint")
        verdict = verify_claim(CLAIM, [("E1", QUOTE)], scorer, support_threshold=0.98)
        assert not verdict.publishable
        assert not verdict.checked
        assert "unavailable" in verdict.reason

    def test_truncated_scorer_response_withholds(self) -> None:
        verdict = verify_claim(
            CLAIM, [("E1", QUOTE), ("E2", QUOTE)], BrokenScorer(), support_threshold=0.98
        )
        assert not verdict.publishable
        assert not verdict.checked
        assert "returned 1 results for 2 pairs" in verdict.reason

    def test_out_of_range_probability_withholds(self) -> None:
        verdict = verify_claim(
            CLAIM, [("E1", QUOTE)], BrokenScorer(mode="out_of_range"), support_threshold=0.98
        )
        assert not verdict.publishable
        assert "malformed scores" in verdict.reason

    @pytest.mark.parametrize("threshold", [0.5, 0.9, 0.98, 1.0])
    def test_nothing_publishes_without_a_score(self, threshold: float) -> None:
        scorer = FakeScorer(fail_with="offline")
        assert not verify_claim(
            CLAIM, [("E1", QUOTE)], scorer, support_threshold=threshold
        ).publishable


class TestDiagnosticVerdict:
    def test_contradiction_reads_unsupported(self) -> None:
        scorer = FakeScorer({(QUOTE, CLAIM): (0.1, 0.2, 0.7)})
        verdict = verify_claim(CLAIM, [("E1", QUOTE)], scorer, support_threshold=0.98)
        assert verdict.verdict == UNSUPPORTED

    def test_partial_entailment_reads_partially_supported(self) -> None:
        scorer = FakeScorer({(QUOTE, CLAIM): (0.6, 0.4, 0.0)})
        verdict = verify_claim(CLAIM, [("E1", QUOTE)], scorer, support_threshold=0.98)
        assert verdict.verdict == PARTIALLY_SUPPORTED

    def test_negligible_entailment_reads_unsupported(self) -> None:
        scorer = FakeScorer({(QUOTE, CLAIM): (0.01, 0.99, 0.0)})
        verdict = verify_claim(CLAIM, [("E1", QUOTE)], scorer, support_threshold=0.98)
        assert verdict.verdict == UNSUPPORTED

    def test_diagnostic_label_does_not_gate_publication(self) -> None:
        """Only `publishable` is read by the gate. The three-way label is
        derived afterwards and controls nothing."""
        verdict = verify_claim(CLAIM, [("E1", QUOTE)], entailing(), support_threshold=0.98)
        assert verdict.publishable and verdict.verdict == SUPPORTED


class TestStressFixtureGuardCoverage:
    """The guard-decidable part of the adversarial suite, without a model.

    Cases the guards alone must refuse are refused here whatever the
    classifier says -- proven by handing the gate a scorer that entails
    everything at 1.0. The remaining cases need real semantics and are
    covered by the marked real-model test.
    """

    FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "semantic_stress.json"
    GUARD_DECIDABLE = {
        "modality-hedge-to-tendency",
        "modality-hedge-to-necessity",
        "modality-hedge-to-universal",
        "numeric-substituted",
        "numeric-inflated-multiplier",
        "numeric-invented",
        "ranking-invented",
        "ranking-invented-best",
        "causal-from-association",
        "causal-from-correlation",
        "exclusivity-invented",
        "exclusivity-invented-necessity",
    }
    # Deliberately absent: comparison-invented ("System A used 4 GB" ->
    # "System A used less memory than System B"). The ranking guard
    # covers superlatives, not comparatives between two named entities,
    # and widening it to "less than" would fire on ordinary numeric
    # bounds like "under 5 ms". That case is left to the classifier and
    # is checked in the real-model run.

    def cases(self) -> list[dict]:
        return json.loads(self.FIXTURE.read_text())["cases"]

    def test_fixture_covers_every_guard_category(self) -> None:
        categories = {c["category"] for c in self.cases()}
        assert {"modality", "numeric", "ranking", "causal", "exclusivity"} <= categories

    def test_guards_refuse_these_against_an_all_entailing_scorer(self) -> None:
        failures = []
        for case in self.cases():
            if case["id"] not in self.GUARD_DECIDABLE:
                continue
            scorer = FakeScorer(default=(1.0, 0.0, 0.0))
            verdict = verify_claim(
                case["hypothesis"], [("E1", case["premise"])], scorer, support_threshold=0.98
            )
            if verdict.publishable:
                failures.append(case["id"])
        assert not failures, f"guards let these through at entailment 1.0: {failures}"

    def test_faithful_restatements_are_not_blocked_by_guards(self) -> None:
        """Guards must not withhold a claim that copies its evidence."""
        blocked = []
        for case in self.cases():
            if case["expected"] != "publish":
                continue
            scorer = FakeScorer(default=(1.0, 0.0, 0.0))
            verdict = verify_claim(
                case["hypothesis"], [("E1", case["premise"])], scorer, support_threshold=0.98
            )
            if not verdict.publishable:
                blocked.append((case["id"], verdict.reason))
        assert not blocked, f"guards blocked faithful restatements: {blocked}"
