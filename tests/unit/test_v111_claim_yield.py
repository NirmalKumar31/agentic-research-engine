"""v1.11: the defects that discarded 91% of extracted evidence.

Three v1.10 live runs extracted 85 evidence items and cited 8, publishing
8 of 22 generated claims. These tests pin the causes that were found by
reading those runs, so a regression shows up here rather than in a paid
run.

Every test names the live symptom it came from. Tests live where the
behaviour *runs*, not where the constant is declared -- the pattern this
repo has produced a defect from eleven times.
"""

from __future__ import annotations

import pytest

from agentic_research.citations.atomicity import compound_propositions
from agentic_research.citations.guards import (
    _BARE_ASSERTION,
    failed_guards,
    guards_pass,
    hedge_guard,
    modality_band,
    modality_guard,
    run_guards,
)
from agentic_research.citations.repair import (
    _BARE_ASSERTION_LEVEL,
    REPAIRABLE_GUARDS,
    UNREPAIRABLE_GUARDS,
    _modality_level,
)


class TestModalityLaddersAgree:
    """`guards.py` and `repair.py` each rank modality, and must not diverge.

    The live defect: `repair.py` discovered that a bare assertion
    outranks a hedge and fixed it with its own `_BARE_ASSERTION_LEVEL`,
    while `guards.py` -- the module that actually gates publication --
    kept scoring it 0. One module fixed, the other not.
    """

    HEDGED = "the system may use 64GB"
    BARE = "the system uses 64GB"
    NECESSARY = "the system must use 64GB"

    def test_both_modules_place_a_bare_assertion_above_a_hedge(self) -> None:
        assert modality_band(self.BARE) > modality_band(self.HEDGED)
        assert _modality_level(self.BARE) > _modality_level(self.HEDGED)

    def test_both_modules_place_a_bare_assertion_below_a_necessity(self) -> None:
        assert modality_band(self.BARE) < modality_band(self.NECESSARY)
        assert _modality_level(self.BARE) < _modality_level(self.NECESSARY)

    def test_the_two_bare_assertion_constants_are_the_same_rung(self) -> None:
        """Not a style point. They are compared against the same hedges."""
        assert _BARE_ASSERTION == _BARE_ASSERTION_LEVEL


class TestHedgingBelowEvidenceIsAllowed:
    """The direction that is always safe, and was being refused.

    Flat evidence scored band 0 -- below every hedge -- so a claim that
    reported its source *more cautiously* than the source itself was
    refused. Three claims in one live run died this way.
    """

    def test_the_guard_set_as_a_whole_admits_a_hedged_claim(self) -> None:
        """Through `run_guards`, not `modality_guard` alone.

        A fix that satisfies one guard and trips another publishes
        nothing, so the end-to-end path is what the claim depends on.
        """
        claim = "SQLite deployment can consist of copying the database file."
        evidence = "SQLite deployment consists of copying the database file."
        results = run_guards(claim, evidence)
        assert guards_pass(results), [r.detail for r in failed_guards(results)]


class TestDeletionIsStillRefused:
    """The exemption must not become a loophole.

    `modality_guard` abstains on a bare claim so that deletion is
    classified by `hedge_guard`, which marks it unrepairable. If the
    exemption silently stopped anything from catching deletion, the fix
    above would have bought yield with integrity.
    """

    CLAIM = "the deployment uses 64GB"
    EVIDENCE = "deployments may need 64GB of dedicated RAM"

    def test_modality_abstains_so_the_classification_is_not_stolen(self) -> None:
        assert modality_guard(self.CLAIM, self.EVIDENCE).passed

    def test_but_hedge_guard_refuses_it(self) -> None:
        assert not hedge_guard(self.CLAIM, self.EVIDENCE).passed

    def test_so_the_claim_does_not_publish(self) -> None:
        results = run_guards(self.CLAIM, self.EVIDENCE)
        assert not guards_pass(results)
        assert "hedge" in {r.name for r in failed_guards(results)}

    def test_and_it_is_classified_unrepairable(self) -> None:
        """Reporting deletion as `modality` would send it to be reworded."""
        assert "hedge" in UNREPAIRABLE_GUARDS
        assert "hedge" not in REPAIRABLE_GUARDS


class TestStrengtheningIsStillRefused:
    """The band fix must not have loosened the direction that matters."""

    EVIDENCE = "deployments may need 64GB of dedicated RAM"

    @pytest.mark.parametrize(
        "claim",
        [
            "deployments typically require 64GB",
            "deployments must use 64GB",
            "deployments always need 64GB",
            "64GB is necessary for deployment",
        ],
    )
    def test_strengthened_modality_still_fails(self, claim: str) -> None:
        assert not modality_guard(claim, self.EVIDENCE).passed


class TestNounVerbAmbiguityDoesNotFakeACompound:
    """Plural nouns were being read as the predicate of their own clause.

    "stores", "reads", "writes" and "requires" are all noun and verb.
    The atomicity guard judged a coordinated segment by whether *any*
    token looked like a predicate, so a noun phrase read as a clause and
    the guard refused claims that were never compound.
    """

    LIVE_FALSE_POSITIVE = (
        "LangGraph's persistence layer gives agents short-term memory through "
        "checkpointers and long-term memory through stores."
    )

    def test_the_live_claim_reads_as_one_proposition(self) -> None:
        """The exact wording a live run lost, cited to official docs at 0.92.

        "long-term memory through stores" was counted as a clause
        because "stores" follows "through" -- where English does not put
        a finite verb.
        """
        assert compound_propositions(self.LIVE_FALSE_POSITIVE) == []

    @pytest.mark.parametrize(
        "text",
        [
            "the engine writes through stores",
            "it serves requests with reads",
            "the layer persists state via writes",
        ],
    )
    def test_a_predicate_after_a_preposition_is_its_object(self, text: str) -> None:
        assert compound_propositions(text) == []

    @pytest.mark.parametrize(
        "text",
        [
            # Two real clauses: each predicate follows a subject, not a
            # preposition. Narrowing the misparse must not reach these.
            "For a single process doing reads and small writes, SQLite is hard to "
            "beat, and it often outruns PostgreSQL on simple queries.",
            "The planner emitted three queries, and the critic requested another round.",
        ],
    )
    def test_genuine_compounds_still_fire(self, text: str) -> None:
        assert compound_propositions(text) != []

    def test_multiple_sentences_still_fire(self) -> None:
        """A separate limb of the guard, which this change must not touch."""
        assert compound_propositions("SQLite serializes writes. PostgreSQL does not.") != []
