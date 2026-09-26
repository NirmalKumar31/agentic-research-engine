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


class TestAuditRecordSurvivesTheRebuild:
    """The judgments must reach the artifact, not just the local object.

    verify_citations rebuilds its result from the published report as a
    final step, carrying a named list of fields forward. ``judgments``
    was not on that list, so every candidate's full-fidelity record was
    appended and then thrown away, leaving only the 200-character
    CitationIssue entries -- which is exactly the truncation that made
    four calibration cases unusable.
    """

    def test_judgments_are_carried_through_the_rebuild(self) -> None:
        import inspect

        from agentic_research.graph.nodes import reporting

        source = inspect.getsource(reporting.verify_citations)
        rebuild = source[source.index("semantic = {") : source.index("result.final_published")]
        assert '"judgments": result.judgments' in rebuild, (
            "verify_citations rebuilds its result without carrying judgments forward; "
            "the audit record will be empty in every artifact"
        )

    def test_every_carried_field_is_a_real_model_field(self) -> None:
        """A typo in that dict would silently drop a field rather than
        raise, since model_copy(update=...) does not validate names."""
        import inspect
        import re

        from agentic_research.graph.nodes import reporting
        from agentic_research.models import CitationVerification

        source = inspect.getsource(reporting.verify_citations)
        rebuild = source[source.index("semantic = {") : source.index("result.final_published")]
        names = set(re.findall(r'"(\w+)":', rebuild))
        unknown = names - set(CitationVerification.model_fields)
        assert not unknown, f"not fields of CitationVerification: {sorted(unknown)}"


class TestCompoundClaimFailureClass:
    """The shape that produced the one development false positive.

    Three propositions fused into one sentence, where a single quote
    states two of them and not the third. No guard can reach it and the
    classifier scores the sentence on its supported majority, so the
    unsupported clause rides along. Generic wording throughout; nothing
    here special-cases any phrase.

    The verifier is not expected to catch this -- that is the finding.
    What these assert is that splitting the sentence into atomic claims
    is what fixes it, which is why the fix went into synthesis.
    """

    QUOTE = (
        "Pruning cut index memory by 60%. The authors report minimal impact "
        "on recall and high precision on the held-out set."
    )

    def test_the_fused_claim_is_refused_by_the_publication_path(self) -> None:
        """The whole path, not the helper: verify_claim -> run_guards ->
        atomicity_guard -> decision.

        This test previously asserted the opposite. A fused one-sentence
        claim published at 0.99, and that was recorded as a known
        limitation -- which it was, until the clause check was written
        and then not wired into the guard. Pinning the end-to-end
        behaviour is the only thing that would have caught that.
        """
        fused = "Pruning cut index memory by 60% while maintaining high recall."
        scorer = FakeScorer(default=(1.0, 0.0, 0.0))
        verdict = verify_claim(fused, [("E1", self.QUOTE)], scorer, support_threshold=0.98)
        assert not verdict.publishable
        assert "atomicity" in verdict.per_evidence[0].failed_guard_names

    def test_the_unsupported_half_alone_is_refused(self) -> None:
        """Split out, the overreaching proposition is scored on its own
        and has nothing carrying it."""
        atomic = "Pruning maintained high recall."
        scorer = FakeScorer({(self.QUOTE, atomic): (0.12, 0.87, 0.01)})
        verdict = verify_claim(atomic, [("E1", self.QUOTE)], scorer, support_threshold=0.98)
        assert not verdict.publishable

    def test_the_supported_half_alone_still_publishes(self) -> None:
        """Atomic synthesis must not cost the true propositions."""
        atomic = "Pruning cut index memory by 60%."
        scorer = FakeScorer({(self.QUOTE, atomic): (0.99, 0.01, 0.0)})
        verdict = verify_claim(atomic, [("E1", self.QUOTE)], scorer, support_threshold=0.98)
        assert verdict.publishable

    def test_the_fused_shape_is_detectable_for_the_audit(self) -> None:
        """Not a gate -- a flag, so a compound claim that does publish is
        visible in the release audit instead of silently passing."""
        from agentic_research.citations.atomicity import looks_compound

        assert looks_compound("Pruning cut index memory by 60% while maintaining high recall.")
        assert not looks_compound("Pruning cut index memory by 60%.")


class TestOneClaimOneSupportContract:
    """§F: one claim, one supporting evidence item.

    Stated explicitly rather than left implicit in the max(). The
    invariant is that a claim publishes because a *single* quote carries
    it, never because several partial quotes add up. Otherwise evidence
    A supporting one clause and evidence B supporting another compose
    into a sentence neither source made.

    Synthesis-kind claims go through the identical rule. There is no
    separate multi-evidence contract in v1: a conclusion that genuinely
    needs two sources should be written as two claims, and if it cannot
    be, it is not publishable here. That is a deliberate limitation, not
    an oversight.
    """

    # An atomic claim. Compound ones are refused earlier by the
    # atomicity guard, which would mask what this is testing.
    CLAIM = "Latency fell by half."
    LEFT = "Latency improved noticeably in the trial."
    RIGHT = "Memory use fell by a third."

    def test_two_partial_quotes_cannot_combine(self) -> None:
        """Neither quote carries the whole claim, so neither publishes
        it, however high both score."""
        scorer = FakeScorer(
            {
                (self.LEFT, self.CLAIM): (0.97, 0.03, 0.0),
                (self.RIGHT, self.CLAIM): (0.97, 0.03, 0.0),
            }
        )
        verdict = verify_claim(
            self.CLAIM,
            [("E1", self.LEFT), ("E2", self.RIGHT)],
            scorer,
            support_threshold=0.98,
        )
        assert not verdict.publishable

    def test_the_decision_rests_on_one_item_not_an_aggregate(self) -> None:
        """A single item at threshold publishes; the verdict names it."""
        scorer = FakeScorer(
            {
                (self.LEFT, "Latency fell."): (0.99, 0.01, 0.0),
                (self.RIGHT, "Latency fell."): (0.01, 0.99, 0.0),
            }
        )
        verdict = verify_claim(
            "Latency fell.", [("E1", self.LEFT), ("E2", self.RIGHT)], scorer, support_threshold=0.98
        )
        assert verdict.publishable
        assert verdict.best_evidence_id == "E1"

    def test_quotes_are_never_concatenated_into_one_premise(self) -> None:
        """The scorer must see each quote separately. Joining them is
        how a broad claim gets rescued by unrelated fragments."""
        scorer = FakeScorer(default=(0.0, 1.0, 0.0))
        verify_claim(
            self.CLAIM, [("E1", self.LEFT), ("E2", self.RIGHT)], scorer, support_threshold=0.98
        )
        premises = [premise for premise, _ in scorer.seen]
        assert premises == [self.LEFT, self.RIGHT]
        assert not any(self.LEFT in p and self.RIGHT in p for p in premises)

    def test_a_synthesis_claim_uses_the_same_rule(self) -> None:
        """No separate contract. If one quote does not carry it, it does
        not publish, whatever kind it declares itself to be."""
        scorer = FakeScorer(
            {
                (self.LEFT, self.CLAIM): (0.97, 0.03, 0.0),
                (self.RIGHT, self.CLAIM): (0.97, 0.03, 0.0),
            }
        )
        verdict = verify_claim(
            self.CLAIM, [("E1", self.LEFT), ("E2", self.RIGHT)], scorer, support_threshold=0.98
        )
        assert not verdict.publishable
        assert "below the" in verdict.reason, verdict.reason
