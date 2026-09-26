"""Every guard is actually reachable from the publication decision.

A guard that exists, is well tested and is never called is worse than
no guard: the tests are green, the report says the property holds, and
nothing enforces it. That happened here. `compound_propositions` was
written, tested against a 17-case matrix, and reported as gating
publication, while `atomicity_guard` still called `sentence_count` and
the clause check was reachable only from its own unit tests.

So these tests assert wiring rather than behaviour. Each one drives the
real path -- verify_claim -> run_guards -> the guard -> the decision --
with a scorer that entails everything at 1.0, so the only thing that
can withhold the claim is the guard. If a guard is ever disconnected
again, its line here fails.
"""

from __future__ import annotations

import pytest

from agentic_research.citations.fake_nli import FakeScorer
from agentic_research.citations.guards import ALL_GUARDS, GUARD_NAMES, SourceIdentity
from agentic_research.citations.semantic import CitedEvidence, verify_claim

# (guard name, claim, quote, source) chosen so that guard and no other
# is the reason the claim cannot publish.
CASES: list[tuple[str, str, str, SourceIdentity | None]] = [
    (
        "atomicity",
        "Latency fell by half and memory use rose.",
        "Latency fell by half and memory use rose.",
        None,
    ),
    (
        "numeric",
        "Throughput rose 75%.",
        "Throughput rose 20% after the change.",
        None,
    ),
    (
        "modality",
        "Deployments must provision 64GB.",
        "Deployments may need 64GB of dedicated RAM.",
        None,
    ),
    (
        "hedge",
        "The format is unsupported.",
        "The format may be unsupported.",
        None,
    ),
    (
        "framing",
        "Method A is the best choice.",
        "We demonstrate that method A is the best choice.",
        None,
    ),
    (
        "ranking",
        "System A recorded the lowest latency.",
        "System A recorded 5 ms median latency.",
        None,
    ),
    (
        "causal",
        "Larger caches caused lower tail latency.",
        "Larger caches were associated with lower tail latency.",
        None,
    ),
    (
        "exclusivity",
        "Only quantisation reduces index memory.",
        "Quantisation reduces index memory.",
        None,
    ),
    (
        "attribution",
        "NIST requires organizations to manage AI risks.",
        "Organizations must manage AI risks.",
        SourceIdentity("vendor.example", "A Vendor Blog"),
    ),
]


def entails_everything() -> FakeScorer:
    """Nothing can withhold a claim except a guard."""
    return FakeScorer(default=(1.0, 0.0, 0.0))


class TestEveryGuardIsWired:
    def test_the_case_list_covers_every_registered_guard(self) -> None:
        """Adding a guard without adding a wiring case fails here
        rather than shipping unenforced."""
        assert {name for name, *_ in CASES} == set(GUARD_NAMES)
        assert len(ALL_GUARDS) + 1 == len(GUARD_NAMES)  # +1: attribution takes a source

    @pytest.mark.parametrize(("name", "claim", "quote", "source"), CASES, ids=[c[0] for c in CASES])
    def test_the_guard_overrides_a_perfect_entailment_score(
        self, name: str, claim: str, quote: str, source: SourceIdentity | None
    ) -> None:
        verdict = verify_claim(
            claim,
            [CitedEvidence("E1", quote, source)],
            entails_everything(),
            support_threshold=0.98,
        )
        assert not verdict.publishable, f"{name} did not stop a 1.0 entailment"
        assert name in verdict.per_evidence[0].failed_guard_names, (
            f"{name} was not the guard that fired; got {verdict.per_evidence[0].failed_guard_names}"
        )

    @pytest.mark.parametrize(("name", "claim", "quote", "source"), CASES, ids=[c[0] for c in CASES])
    def test_the_reason_reaches_the_audit_record(
        self, name: str, claim: str, quote: str, source: SourceIdentity | None
    ) -> None:
        """A withheld claim has to be explainable, or the audit is a
        list of refusals without grounds."""
        verdict = verify_claim(
            claim,
            [CitedEvidence("E1", quote, source)],
            entails_everything(),
            support_threshold=0.98,
        )
        assert verdict.reason
        assert any(g.detail for g in verdict.per_evidence[0].guards if not g.passed)


class TestPublicationPathAtomicityMatrix:
    """§4: the same matrix as the helper tests, through the real gate.

    The helper was already proven. What was not proven -- and was false
    -- is that the gate consults it.
    """

    QUOTE = "Any quote at all; the scorer entails everything."

    @pytest.mark.parametrize(
        "claim",
        [
            "X increased, Y decreased.",
            "X improved accuracy and reduced latency.",
            "X improved accuracy while reducing memory.",
            "X was faster but less accurate.",
            "Accuracy was 92%, recall was 81%.",
            "Quantization reduces memory use, and accuracy remains high.",
        ],
    )
    def test_compound_claims_cannot_publish(self, claim: str) -> None:
        verdict = verify_claim(
            claim, [CitedEvidence("E1", claim)], entails_everything(), support_threshold=0.98
        )
        assert not verdict.publishable
        assert "atomicity" in verdict.per_evidence[0].failed_guard_names

    @pytest.mark.parametrize(
        "claim",
        [
            "Precision, recall, and F1 were reported.",
            "The benchmark reports precision, recall, and F1.",
            "The system supports CSV and Parquet.",
        ],
    )
    def test_enumerations_still_publish(self, claim: str) -> None:
        """Rejecting every conjunction would withhold most real
        sentences, which is the failure mode opposite to the one being
        fixed."""
        verdict = verify_claim(
            claim, [CitedEvidence("E1", claim)], entails_everything(), support_threshold=0.98
        )
        assert verdict.publishable, verdict.reason
