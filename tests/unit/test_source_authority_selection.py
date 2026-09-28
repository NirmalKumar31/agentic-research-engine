"""Which quote carries a claim, once several could.

A live run cited a tweet at entailment 0.994 over an arXiv paper at
0.985 for the same technical claim, because 0.994 is the larger
number. It is not the better source, and the engine had already
computed that and thrown it away: source quality was calculated for
display and never consulted when choosing evidence.

Authority orders evidence that has *already* qualified. It is not a
second opinion on whether a claim is true, and it never reaches the
classifier.
"""

from __future__ import annotations

from agentic_research.citations.fake_nli import FakeScorer
from agentic_research.citations.guards import SourceAuthority, SourceIdentity
from agentic_research.citations.semantic import CitedEvidence, verify_claim

THRESHOLD = 0.98
CLAIM = "Decoder-only transformers condition each token only on preceding tokens."
PAPER_QUOTE = "In a decoder-only transformer the representation of each token depends only on preceding tokens."
TWEET_QUOTE = "Decoder-only transformers condition each token only on preceding tokens."


def source(domain: str, authority: SourceAuthority, quality: float) -> SourceIdentity:
    return SourceIdentity(domain=domain, title="", authority=authority, quality=quality)


PAPER = CitedEvidence("S1-e1", PAPER_QUOTE, source("arxiv.org", SourceAuthority.PRIMARY, 0.96))
TWEET = CitedEvidence("S3-e1", TWEET_QUOTE, source("x.com", SourceAuthority.AGGREGATOR, 0.55))
WIKI = CitedEvidence(
    "S2-e1", PAPER_QUOTE, source("en.wikipedia.org", SourceAuthority.SECONDARY, 0.61)
)


def scorer(**by_quote: float) -> FakeScorer:
    return FakeScorer(
        {(quote, CLAIM): (score, round(1 - score, 6), 0.0) for quote, score in by_quote.items()}
    )


class TestTheStrongerSourceCarriesTheClaim:
    def test_a_paper_beats_a_tweet_that_scored_higher(self) -> None:
        """The exact live failure, as a test."""
        verdict = verify_claim(
            CLAIM,
            [TWEET, PAPER],
            scorer(**{TWEET_QUOTE: 0.994, PAPER_QUOTE: 0.985}),
            support_threshold=THRESHOLD,
        )
        assert verdict.publishable
        assert verdict.best_evidence_id == "S1-e1", "the tweet was cited again"

    def test_a_paper_beats_an_encyclopaedia(self) -> None:
        verdict = verify_claim(
            CLAIM,
            [WIKI, PAPER],
            scorer(**{PAPER_QUOTE: 0.99}),
            support_threshold=THRESHOLD,
        )
        assert verdict.best_evidence_id in {"S1-e1", "S2-e1"}
        chosen = next(s for s in verdict.per_evidence if s.evidence_id == verdict.best_evidence_id)
        assert chosen.source is not None
        assert chosen.source.authority is SourceAuthority.PRIMARY

    def test_quality_breaks_a_tie_within_one_authority_level(self) -> None:
        better = CitedEvidence("A", PAPER_QUOTE, source("a.org", SourceAuthority.SECONDARY, 0.90))
        worse = CitedEvidence("B", PAPER_QUOTE, source("b.org", SourceAuthority.SECONDARY, 0.40))
        verdict = verify_claim(
            CLAIM, [worse, better], scorer(**{PAPER_QUOTE: 0.99}), support_threshold=THRESHOLD
        )
        assert verdict.best_evidence_id == "A"

    def test_the_same_candidates_always_produce_the_same_citation(self) -> None:
        """Deterministic, so a report can be reproduced."""
        args = ([TWEET, PAPER, WIKI], scorer(**{TWEET_QUOTE: 0.994, PAPER_QUOTE: 0.99}))
        first = verify_claim(CLAIM, args[0], args[1], support_threshold=THRESHOLD)
        second = verify_claim(CLAIM, list(reversed(args[0])), args[1], support_threshold=THRESHOLD)
        assert first.best_evidence_id == second.best_evidence_id


class TestEntailmentRemainsAGate:
    def test_authority_cannot_rescue_an_unsupported_quote(self) -> None:
        """A primary source below the threshold is still below it."""
        verdict = verify_claim(
            CLAIM,
            [PAPER],
            scorer(**{PAPER_QUOTE: 0.42}),
            support_threshold=THRESHOLD,
        )
        assert verdict.publishable is False

    def test_a_weak_source_above_the_threshold_still_publishes_alone(self) -> None:
        """Authority orders alternatives; it does not veto the only
        evidence there is. Withholding a supported claim because its
        source is a blog would be a different gate, and not this one."""
        verdict = verify_claim(
            CLAIM, [TWEET], scorer(**{TWEET_QUOTE: 0.994}), support_threshold=THRESHOLD
        )
        assert verdict.publishable is True
        assert verdict.best_evidence_id == "S3-e1"

    def test_a_strong_source_below_the_threshold_loses_to_nothing(self) -> None:
        """When nothing qualifies, the report explains how close the
        best attempt came, so the highest score is still what is shown."""
        verdict = verify_claim(
            CLAIM,
            [PAPER, TWEET],
            scorer(**{PAPER_QUOTE: 0.40, TWEET_QUOTE: 0.70}),
            support_threshold=THRESHOLD,
        )
        assert verdict.publishable is False
        assert verdict.best_evidence_id == "S3-e1"


class TestTheClassifierNeverSeesTheSource:
    """The property the whole design rests on.

    A classifier told a quote came from an authoritative domain would
    be scoring reputation. Entailment is the only thing it may score,
    so authority and quality must not reach the premise.
    """

    def test_no_domain_authority_or_quality_reaches_the_premise(self) -> None:
        s = scorer(**{TWEET_QUOTE: 0.994, PAPER_QUOTE: 0.985})
        verify_claim(CLAIM, [TWEET, PAPER], s, support_threshold=THRESHOLD)
        seen = " ".join(premise for premise, _ in s.seen)
        for leak in ("arxiv", "x.com", "primary", "aggregator", "0.96", "0.55"):
            assert leak not in seen.lower(), f"{leak!r} reached the classifier"

    def test_the_premise_is_exactly_the_quote(self) -> None:
        s = scorer(**{PAPER_QUOTE: 0.99})
        verify_claim(CLAIM, [PAPER], s, support_threshold=THRESHOLD)
        assert [p for p, _ in s.seen] == [PAPER_QUOTE]


class TestUnknownSources:
    def test_evidence_without_a_source_still_works(self) -> None:
        """Plain (id, quote) pairs are accepted; they simply rank lowest."""
        verdict = verify_claim(
            CLAIM,
            [("X", PAPER_QUOTE)],
            scorer(**{PAPER_QUOTE: 0.99}),
            support_threshold=THRESHOLD,
        )
        assert verdict.publishable
        assert verdict.best_evidence_id == "X"

    def test_a_known_source_beats_an_unknown_one(self) -> None:
        verdict = verify_claim(
            CLAIM,
            [("X", PAPER_QUOTE), PAPER],
            scorer(**{PAPER_QUOTE: 0.99}),
            support_threshold=THRESHOLD,
        )
        assert verdict.best_evidence_id == "S1-e1"
