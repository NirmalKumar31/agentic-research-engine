"""The two guards added to close the release audit failure.

Generic wording throughout. No case-specific token from the failing
recording appears in any rule these exercise, and the one real defect is
tested by its transformation class, not by its subject matter.
"""

from __future__ import annotations

import pytest

from agentic_research.citations.fake_nli import FakeScorer
from agentic_research.citations.guards import (
    SourceIdentity,
    attributed_entities,
    attribution_guard,
    framing_guard,
    hedge_guard,
    modality_guard,
    run_guards,
)
from agentic_research.citations.semantic import CitedEvidence, verify_claim


class TestEpistemicHedgeDeletion:
    """Evidence that hedges must not be published as fact.

    The band-based modality guard cannot see this: its rule is "claim
    band must not exceed evidence band", and a claim with no modality
    sits in band 0, the weakest. So "may" to "must" fails while "may"
    to nothing passes -- and deletion is the more common overclaim.
    """

    @pytest.mark.parametrize(
        ("claim", "evidence"),
        [
            ("X causes Y.", "X may cause Y."),
            ("X lacks feature Y.", "X might lack feature Y."),
            ("X is vulnerable.", "X is potentially vulnerable."),
            ("X is more expensive.", "X is possibly more expensive."),
            ("X reduces latency.", "Perhaps X reduces latency."),
            ("The format is unsupported.", "The format may be unsupported."),
        ],
    )
    def test_deleting_the_hedge_is_refused(self, claim: str, evidence: str) -> None:
        assert not hedge_guard(claim, evidence).passed

    @pytest.mark.parametrize(
        ("claim", "evidence"),
        [
            ("X might reduce latency.", "X might reduce latency."),
            ("X may reduce latency.", "X may reduce latency."),
            ("X may reduce latency.", "X might reduce latency."),
            ("X might be affected.", "X may possibly be affected."),
        ],
    )
    def test_keeping_the_hedge_is_allowed(self, claim: str, evidence: str) -> None:
        assert hedge_guard(claim, evidence).passed

    @pytest.mark.parametrize(
        ("claim", "evidence"),
        [
            ("X supports processing large datasets.", "X can process large datasets."),
            ("X processes large datasets.", "X can process large datasets."),
            ("X handles replication.", "X could handle replication."),
        ],
    )
    def test_capability_modals_are_not_treated_as_uncertainty(
        self, claim: str, evidence: str
    ) -> None:
        """ "can" and "could" usually express ability, not doubt.

        Treating every modal as epistemic was measured against the
        release audit and rejected a true claim, so they are excluded.
        """
        assert hedge_guard(claim, evidence).passed

    def test_strengthening_is_still_caught_by_the_band_rule(self) -> None:
        """The two rules compose rather than overlap: one covers
        promotion, the other deletion."""
        assert not modality_guard("X must be replaced.", "X may need replacing.").passed
        assert not hedge_guard("X needs replacing.", "X may need replacing.").passed

    def test_no_hedge_anywhere_abstains(self) -> None:
        assert hedge_guard("Throughput rose 20%.", "Throughput rose 20% overall.").passed


class TestHedgeScopeIsTheSupportingSentence:
    """A hedge about something else must not withhold a faithful claim."""

    PASSAGE = "System A may fail under load. System B uses AES-256 for encryption."

    def test_unrelated_hedge_does_not_block(self) -> None:
        assert hedge_guard("System B uses AES-256 for encryption.", self.PASSAGE).passed

    def test_the_hedged_sentence_still_blocks_its_own_claim(self) -> None:
        assert not hedge_guard("System A fails under load.", self.PASSAGE).passed

    def test_single_sentence_evidence_is_unaffected(self) -> None:
        assert not hedge_guard("X fails.", "X may fail.").passed


class TestAttributedEntityDetection:
    def test_detects_explicit_attribution(self) -> None:
        assert attributed_entities("According to Microsoft, latency halved.") == ["Microsoft"]

    def test_detects_acronym_attribution(self) -> None:
        assert attributed_entities("NIST requires organizations to manage risk.") == ["NIST"]

    @pytest.mark.parametrize(
        "claim",
        [
            "Vector databases require significant memory.",
            "Traditional databases offer robust encryption options.",
            "The study reports minimal impact on recall.",
            "The authors found no regression.",
            "Deep Learning models have shown strong performance.",
            "Compression reduces index size.",
        ],
    )
    def test_ordinary_claims_attribute_to_nobody(self, claim: str) -> None:
        """A sentence-initial capitalised common noun is a subject, not
        a publisher. Firing here would withhold most ordinary claims."""
        assert attributed_entities(claim) == []


class TestAttributionGuard:
    """The regression left by removing publisher identity from the premise.

    The old verifier saw who published the quote. The classifier does
    not, and must not -- source reputation is not entailment. So the
    check moves to deterministic code, using identity only.
    """

    QUOTE = "Organizations must manage AI risks across the lifecycle."
    CLAIM = "NIST requires organizations to manage AI risks across the lifecycle."

    def test_first_party_source_establishes_the_attribution(self) -> None:
        source = SourceIdentity("nist.gov", "NIST AI Risk Management Framework")
        assert attribution_guard(self.CLAIM, self.QUOTE, source).passed

    def test_third_party_source_does_not(self) -> None:
        """A vendor page asserting what a standards body requires is not
        that standards body saying it."""
        source = SourceIdentity("vendor.example", "A Vendor Blog")
        assert not attribution_guard(self.CLAIM, self.QUOTE, source).passed

    def test_attribution_inside_the_quote_is_enough(self) -> None:
        quote = "NIST requires organizations to manage AI risks across the lifecycle."
        source = SourceIdentity("vendor.example", "A Vendor Blog")
        assert attribution_guard(self.CLAIM, quote, source).passed

    def test_unattributed_claim_is_not_interfered_with(self) -> None:
        source = SourceIdentity("vendor.example", "A Vendor Blog")
        assert attribution_guard("Organizations should manage AI risks.", self.QUOTE, source).passed

    def test_named_party_absent_from_quote_and_source_is_refused(self) -> None:
        source = SourceIdentity("vendor.example", "A Vendor Blog")
        assert not attribution_guard(
            "According to Microsoft, latency fell by half.", "Latency fell by half.", source
        ).passed

    def test_source_title_can_establish_it(self) -> None:
        source = SourceIdentity("example.test", "IEEE Standards Association overview")
        assert attribution_guard(
            "IEEE defines the interchange format.", "The format is defined here.", source
        ).passed

    def test_missing_source_identity_fails_closed(self) -> None:
        """An attributed claim with nothing to check against is
        unverified, and unverified withholds."""
        assert not attribution_guard(self.CLAIM, self.QUOTE, None).passed


class TestNoMetadataReachesTheClassifier:
    """§10: the NLI premise is the exact quote and nothing else.

    Source identity exists for the attribution guard. If it ever leaked
    into the premise the classifier would be scoring reputation, which
    is precisely what moving to a dedicated entailment model was meant
    to stop.
    """

    QUOTE = "Latency fell by half in the vendor's own benchmark."

    def test_premise_is_exactly_the_quote(self) -> None:
        scorer = FakeScorer(default=(0.99, 0.01, 0.0))
        verify_claim(
            "Latency fell by half.",
            [
                CitedEvidence(
                    "E1",
                    self.QUOTE,
                    SourceIdentity("authoritative.example", "Highly Trusted Standards Body"),
                )
            ],
            scorer,
            support_threshold=0.98,
        )
        [(premise, hypothesis)] = scorer.seen
        assert premise == self.QUOTE
        assert hypothesis == "Latency fell by half."

    @pytest.mark.parametrize(
        "leak", ["authoritative.example", "Highly Trusted", "quality", "rank", "official_docs"]
    )
    def test_no_identity_or_ranking_token_appears_in_the_premise(self, leak: str) -> None:
        scorer = FakeScorer(default=(0.99, 0.01, 0.0))
        verify_claim(
            "Latency fell by half.",
            [
                CitedEvidence(
                    "E1",
                    self.QUOTE,
                    SourceIdentity("authoritative.example", "Highly Trusted Standards Body"),
                )
            ],
            scorer,
            support_threshold=0.98,
        )
        assert leak not in scorer.seen[0][0]

    def test_identity_still_reaches_the_attribution_guard(self) -> None:
        """The other half of the contract: withheld from the premise,
        but not discarded."""
        verdict = verify_claim(
            "NIST requires organizations to manage AI risks.",
            [
                CitedEvidence(
                    "E1",
                    "Organizations must manage AI risks.",
                    SourceIdentity("vendor.example", "A Vendor Blog"),
                )
            ],
            FakeScorer(default=(0.99, 0.01, 0.0)),
            support_threshold=0.98,
        )
        assert not verdict.publishable
        assert "attribution" in verdict.per_evidence[0].failed_guard_names


class TestTheKnownReleaseDefect:
    """The exact pair the manual audit rejected, as a transformation.

    Kept because it is the one case measured end to end against a real
    model and a human reviewer. The rule that catches it names no token
    from it.
    """

    QUOTE = (
        "Traditional databases offer robust encryption options for data, but some "
        "storage engines might lack standardized encryption features. This can make "
        "it challenging to secure records, particularly when they are stored in "
        "public cloud environments."
    )
    CLAIM = (
        "Traditional databases offer robust encryption options, but some storage "
        "engines lack standardized encryption features, making it challenging to "
        "secure records in public cloud environments."
    )

    def test_the_defect_is_now_refused(self) -> None:
        failed = [g.name for g in run_guards(self.CLAIM, self.QUOTE) if not g.passed]
        assert "hedge" in failed

    def test_it_is_refused_even_at_maximum_entailment(self) -> None:
        """The classifier scored the original 0.9946, so the guard must
        hold independently of any score."""
        verdict = verify_claim(
            self.CLAIM,
            [CitedEvidence("E1", self.QUOTE, SourceIdentity("example.test", "A Page"))],
            FakeScorer(default=(1.0, 0.0, 0.0)),
            support_threshold=0.98,
        )
        assert not verdict.publishable

    def test_the_hedge_preserving_version_still_publishes(self) -> None:
        """The fix must cost the faithful phrasing nothing."""
        faithful = "Some storage engines might lack standardized encryption features."
        verdict = verify_claim(
            faithful,
            [CitedEvidence("E1", self.QUOTE, SourceIdentity("example.test", "A Page"))],
            FakeScorer(default=(0.99, 0.01, 0.0)),
            support_threshold=0.98,
        )
        assert verdict.publishable, verdict.reason


class TestResearchVoiceFraming:
    """A source's own finding must not be published as settled fact.

    "We demonstrate that X" and a bare "X" are different assertions: one
    paper reporting a result, versus the field agreeing. The release
    audit published exactly this transformation -- a superlative about
    the best evaluation metric, lifted out of one paper's own voice --
    and it passed every other guard because the wording is otherwise
    verbatim.
    """

    @pytest.mark.parametrize(
        ("claim", "evidence"),
        [
            ("Method A is the best choice.", "We demonstrate that method A is the best choice."),
            ("Latency falls by half.", "We find that latency falls by half."),
            ("The approach generalises.", "Our results show the approach generalises."),
            ("Batching improves throughput.", "This paper shows batching improves throughput."),
            ("The index is smaller.", "We report that the index is smaller."),
        ],
    )
    def test_deleting_the_research_voice_is_refused(self, claim: str, evidence: str) -> None:
        assert not framing_guard(claim, evidence).passed

    @pytest.mark.parametrize(
        "claim",
        [
            "The authors demonstrate that method A is the best choice.",
            "The study reports that method A is the best choice.",
            "According to the paper, method A is the best choice.",
            "The researchers found that method A is the best choice.",
        ],
    )
    def test_keeping_any_attribution_is_enough(self, claim: str) -> None:
        """The claim need not copy the source's wording, only preserve
        that this is somebody's finding."""
        evidence = "We demonstrate that method A is the best choice."
        assert framing_guard(claim, evidence).passed

    @pytest.mark.parametrize(
        ("claim", "evidence"),
        [
            ("Latency is measured end to end.", "Latency is measured end to end."),
            ("The format was deprecated in 2021.", "The format was deprecated in 2021."),
            ("SMOTE balances datasets.", "Techniques like SMOTE can balance datasets."),
        ],
    )
    def test_ordinary_expository_evidence_is_untouched(self, claim: str, evidence: str) -> None:
        """Most sources are not reporting their own experiments. The
        guard must stay silent on them or it withholds everything."""
        assert framing_guard(claim, evidence).passed

    def test_scoped_to_the_supporting_sentence(self) -> None:
        """A methods sentence elsewhere in a long quote must not
        withhold an unrelated factual claim."""
        passage = (
            "We demonstrate that the sampler converges quickly. "
            "Latency is the time taken to process one transaction."
        )
        assert framing_guard(
            "Latency is the time taken to process one transaction.", passage
        ).passed
        assert not framing_guard("The sampler converges quickly.", passage).passed

    def test_the_audited_defect_is_refused(self) -> None:
        """The exact transformation the release audit rejected, stated
        generically -- no token from the original subject matter."""
        evidence = (
            "We demonstrate that a combined precision and recall score, in that "
            "specific order, is the best evaluation metric for this task."
        )
        claim = (
            "A combined precision and recall score, in that specific order, is the "
            "best evaluation metric for this task."
        )
        assert not framing_guard(claim, evidence).passed
