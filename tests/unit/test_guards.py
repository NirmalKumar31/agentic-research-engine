"""Deterministic overclaim guards.

Generic wording throughout: these test the transformation class, not the
thirty development calibration cases, so passing here is not evidence
about those.
"""

from __future__ import annotations

import pytest

from agentic_research.citations.guards import (
    causal_guard,
    exclusivity_guard,
    guards_pass,
    modality_band,
    modality_guard,
    numeric_guard,
    ranking_guard,
    run_guards,
)


class TestModalityBand:
    def test_hedged_necessity_reads_as_the_hedge(self) -> None:
        """ "may require" is a hedge over a necessity, not a necessity.

        Without this the guard fails a claim that copies its evidence
        word for word, which is the one thing it must never do.
        """
        assert modality_band("the system may require 64GB") == 1
        assert modality_band("might be required") == 1
        assert modality_band("could need extra memory") == 1

    def test_bare_necessity_reads_strong(self) -> None:
        assert modality_band("the system requires 64GB") == 3
        assert modality_band("this is always necessary") == 3

    def test_tendency_sits_between(self) -> None:
        assert modality_band("typically requires 64GB") == 3
        assert modality_band("this typically happens") == 2

    def test_absent_modality_is_zero(self) -> None:
        assert modality_band("the system used 64GB") == 0

    def test_substrings_do_not_match(self) -> None:
        """Word boundaries: "mayor" is not "may", "canvas" is not "can"."""
        assert modality_band("the mayor visited the canvas factory") == 0


class TestModalityGuard:
    EVIDENCE = "deployments may need 64GB of dedicated RAM"

    def test_matching_hedge_passes(self) -> None:
        assert modality_guard("deployments may need 64GB", self.EVIDENCE).passed

    @pytest.mark.parametrize(
        "claim",
        [
            "deployments typically require 64GB",
            "deployments must use 64GB",
            "deployments always need 64GB",
            "64GB is necessary for deployment",
        ],
    )
    def test_strengthened_modality_fails(self, claim: str) -> None:
        assert not modality_guard(claim, self.EVIDENCE).passed

    def test_weakening_passes(self) -> None:
        """Claiming less than the evidence says is not an overclaim."""
        assert modality_guard("deployments may need 64GB", "deployments require 64GB").passed

    def test_unmodalised_claim_abstains(self) -> None:
        assert modality_guard("the deployment used 64GB", self.EVIDENCE).passed


class TestNumericGuard:
    def test_literal_present_passes(self) -> None:
        assert numeric_guard("throughput rose 20%", "throughput rose 20% overall").passed

    @pytest.mark.parametrize(
        ("claim", "evidence"),
        [
            ("throughput rose 75%", "throughput rose 20% overall"),
            ("a 5x speedup", "a .5x speedup"),
            ("a 32-bit index", "a 3:2-bit index"),
            ("scored 1.0", "scored 1:0"),
        ],
    )
    def test_substituted_literal_fails(self, claim: str, evidence: str) -> None:
        """The corruptions the generative verifier produced in its own
        reasoning, each of which must now be caught in code."""
        assert not numeric_guard(claim, evidence).passed

    def test_comma_grouping_is_equivalent(self) -> None:
        assert numeric_guard("over 20,000 records", "over 20000 records").passed

    def test_unit_spacing_is_equivalent(self) -> None:
        assert numeric_guard("latency of 4ms", "latency of 4 ms").passed

    def test_differently_written_magnitude_fails_closed(self) -> None:
        """A true claim withheld. The safe direction, and deliberate --
        equating these needs unit arithmetic the guard does not do."""
        assert not numeric_guard("100M parameters", "100 million parameters").passed

    def test_claim_without_numbers_abstains(self) -> None:
        assert numeric_guard("the index is compact", "throughput rose 20%").passed


class TestRankingGuard:
    def test_reported_value_passes(self) -> None:
        assert ranking_guard("System A recorded 5ms", "System A: 5ms p50").passed

    def test_invented_ranking_fails(self) -> None:
        """A value is not a ranking, however good the value looks."""
        assert not ranking_guard("System A had the lowest latency", "System A: 5ms p50").passed

    def test_ranking_in_evidence_passes(self) -> None:
        assert ranking_guard(
            "System A had the lowest latency", "System A showed the lowest latency of the three"
        ).passed

    @pytest.mark.parametrize(
        "evidence",
        [
            "System A wins on raw throughput among the tested engines",
            "System A leads for throughput",
            "System A outperforms the alternatives on throughput",
            "System A beats the alternatives on throughput",
            "System A is the top choice for throughput",
        ],
    )
    def test_ranking_stated_in_other_words_passes(self, evidence: str) -> None:
        """A ranking expressed with a verb is still a ranking.

        Requiring the claim's own superlative to reappear verbatim made
        the guard a synonym test, blocking true claims whenever the
        source ranked in different words.
        """
        assert ranking_guard("System A achieves the highest throughput", evidence).passed

    def test_a_different_ranking_is_left_to_the_classifier(self) -> None:
        """The guard is a necessary condition, not a sufficient one.

        Evidence ranking latency does not establish a claim ranking
        throughput, but deciding that is a semantic question. The guard
        passes it through deliberately; the classifier refuses it, and
        the stress suite holds that end to end.
        """
        assert ranking_guard(
            "System A achieves the highest throughput",
            "System B leads for p99 latency",
        ).passed

    def test_causal_leads_to_is_not_read_as_a_ranking(self) -> None:
        """ "leads to" is causal. Matching bare "leads" would invent a
        ranking in the evidence and let an invented ranking publish."""
        assert not ranking_guard(
            "System A had the lowest latency", "Caching leads to better response times"
        ).passed


class TestCausalGuard:
    def test_association_does_not_become_cause(self) -> None:
        result = causal_guard(
            "compression caused the recall drop", "compression was associated with lower recall"
        )
        assert not result.passed

    def test_causal_evidence_passes(self) -> None:
        assert causal_guard(
            "compression caused the recall drop", "compression caused recall to fall"
        ).passed

    def test_no_causal_language_abstains(self) -> None:
        assert causal_guard("compression lowered recall", "recall fell under compression").passed


class TestExclusivityGuard:
    def test_invented_exclusivity_fails(self) -> None:
        assert not exclusivity_guard(
            "only quantisation reduces memory", "quantisation reduces memory"
        ).passed

    def test_exclusive_evidence_passes(self) -> None:
        assert exclusivity_guard(
            "only quantisation reduces memory",
            "quantisation is the only method that reduces memory",
        ).passed


class TestRunGuards:
    def test_every_guard_reports(self) -> None:
        """All five always run, so the audit record shows each decision
        rather than stopping at the first failure."""
        results = run_guards("System A had the lowest latency", "System A: 5ms")
        assert [r.name for r in results] == [
            "numeric",
            "modality",
            "hedge",
            "framing",
            "ranking",
            "causal",
            "exclusivity",
            "attribution",
        ]

    def test_one_failure_withholds(self) -> None:
        assert not guards_pass(run_guards("System A had the lowest latency", "System A: 5ms"))

    def test_faithful_restatement_passes_every_guard(self) -> None:
        results = run_guards(
            "Deployments may need 64GB of RAM", "Deployments may need 64GB of dedicated RAM"
        )
        assert guards_pass(results), [r for r in results if not r.passed]
