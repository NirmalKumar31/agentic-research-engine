"""Phase 0 regression fixtures for retrieval: query intent and selection.

The live overfitting run is the baseline. Asked for the main causes of
overfitting, the engine sent queries like

    "parametric knowledge long tail facts factual recall generalization
     failures language models hallucinations"

and selected, from 47 candidates, a tweet, a LinkedIn-style interview
blog, a Substack newsletter and an arXiv paper on *frozen
overparameterization in transfer learning*. One of five research
dimensions reached its evidence threshold and nothing published.

Two causes, both upstream of the search provider:

* the query-writer prompt instructed the model to "include the specific
  technical terms an authoritative page would use", with no length
  bound, so only research papers matched the query;
* selection ranked by ``round(provider_score, 1)`` first and consulted
  authority only to break ties *inside* a band, so a tweet scoring 0.87
  outranked a primary source scoring 0.84 and authority never applied.

These fixtures are hermetic: no provider is called, and the candidate
pools are written to reproduce the shape of the pool that failed.
"""

from __future__ import annotations

import pytest

from agentic_research.evidence.dedup import Candidate
from agentic_research.models import DiscoveryRef, SourceType


def candidate(
    url: str,
    *,
    score: float,
    sub_questions: tuple[str, ...] = ("SQ1",),
    title: str = "t",
) -> Candidate:
    domain = url.split("/")[2]
    return Candidate(
        canonical_url=url,
        url=url,
        title=title,
        snippet="s",
        domain=domain,
        best_score=score,
        discovered_by=[
            DiscoveryRef(query_id=f"Q{i + 1}", sub_question_id=sq)
            for i, sq in enumerate(sub_questions)
        ],
    )


class TestSocialAndNewsletterSourcesAreClassified:
    """A tweet must not sit in the same neutral bucket as an
    unrecognised page.

    ``other`` carries base quality 0.50 and SECONDARY authority, which
    is *above* a blog. So an unclassified tweet outranked a Medium post,
    and x.com was selected over primary sources on the live run.
    """

    def test_twitter_and_x_are_social(self) -> None:
        from agentic_research.evidence.quality import classify_source

        for url, domain in [
            ("https://x.com/someone/status/1", "x.com"),
            ("https://twitter.com/someone/status/1", "twitter.com"),
        ]:
            assert classify_source(url, domain) is SourceType.SOCIAL

    def test_linkedin_is_social(self) -> None:
        from agentic_research.evidence.quality import classify_source

        assert (
            classify_source("https://www.linkedin.com/pulse/x", "www.linkedin.com")
            is SourceType.SOCIAL
        )

    def test_a_newsletter_subdomain_is_a_blog(self) -> None:
        from agentic_research.evidence.quality import classify_source

        assert (
            classify_source(
                "https://aiweeklybriefing.substack.com/p/x", "aiweeklybriefing.substack.com"
            )
            is SourceType.BLOG
        )

    def test_social_ranks_below_every_other_class(self) -> None:
        from agentic_research.evidence.quality import base_quality_for

        worst_other = min(base_quality_for(t) for t in SourceType if t is not SourceType.SOCIAL)
        assert base_quality_for(SourceType.SOCIAL) < worst_other

    def test_an_encyclopaedic_reference_is_not_merely_other(self) -> None:
        """Wikipedia was `other`, scoring the same as an unrecognised
        SEO page, while being the best available source for exactly the
        explanatory questions that failed."""
        from agentic_research.evidence.quality import classify_source

        assert (
            classify_source("https://en.wikipedia.org/wiki/Overfitting", "en.wikipedia.org")
            is SourceType.REFERENCE
        )


class TestSelectionPrefersAuthorityAtComparableRelevance:
    """The ranking defect, as a pool.

    A tweet the provider rated 0.87 and a primary source it rated 0.84
    fall in different tenths, so the old banded key never reached
    authority. Relevance must still lead -- an authoritative page about
    the wrong subject is worse than a good page about the right one --
    but a 0.03 relevance difference cannot outweigh tweet-versus-paper.
    """

    def test_a_primary_source_beats_a_tweet_it_is_close_to(self) -> None:
        from agentic_research.retrieval.selection import select_candidates

        pool = [
            candidate("https://x.com/a/status/1", score=0.87),
            candidate("https://arxiv.org/abs/1234", score=0.84),
        ]
        chosen = select_candidates(pool, limit=1, explanatory=True)
        assert [c.domain for c in chosen] == ["arxiv.org"]

    def test_authority_alone_decides_between_two_unpenalised_classes(self) -> None:
        """Non-vacuity for the authority bonus itself.

        Mutation testing found that the tweet case above is decided by
        the social penalty, so removing the authority term entirely
        left every retrieval test passing. Here neither candidate is
        penalised -- a paper against a wire service -- so the bonus is
        the only thing that can reorder them, and the news source has
        the *higher* provider relevance.
        """
        from agentic_research.retrieval.selection import select_candidates

        pool = [
            candidate("https://www.reuters.com/tech/a", score=0.90),
            candidate("https://arxiv.org/abs/1234", score=0.84),
        ]
        chosen = select_candidates(pool, limit=1, explanatory=True)
        assert [c.domain for c in chosen] == ["arxiv.org"]

    def test_the_adjustment_cannot_overturn_a_wide_relevance_gap(self) -> None:
        """The cap, stated as a property: bonus 0.15 plus penalty 0.20
        means a gap wider than 0.35 is decisive whatever the classes."""
        from agentic_research.retrieval.selection import (
            DECIDING_RELEVANCE_GAP,
            select_candidates,
        )

        assert pytest.approx(0.35) == DECIDING_RELEVANCE_GAP
        pool = [
            candidate("https://x.com/a/status/1", score=0.99),
            candidate("https://arxiv.org/abs/1234", score=0.99 - DECIDING_RELEVANCE_GAP - 0.01),
        ]
        chosen = select_candidates(pool, limit=1, explanatory=True)
        assert [c.domain for c in chosen] == ["x.com"]

    def test_relevance_still_wins_when_the_gap_is_large(self) -> None:
        """Authority must not become a veto: a primary source that the
        provider rated far less relevant is probably about something
        else."""
        from agentic_research.retrieval.selection import select_candidates

        pool = [
            candidate("https://someblog.example.com/overfitting", score=0.95),
            candidate("https://arxiv.org/abs/9999", score=0.30),
        ]
        chosen = select_candidates(pool, limit=1, explanatory=True)
        assert [c.domain for c in chosen] == ["someblog.example.com"]

    def test_every_subquestion_gets_a_source_before_any_gets_two(self) -> None:
        """The live run spent all six slots on one dimension's
        candidates and left four dimensions with nothing."""
        from agentic_research.retrieval.selection import select_candidates

        pool = [
            candidate("https://a.example.com/1", score=0.99, sub_questions=("SQ1",)),
            candidate("https://b.example.com/2", score=0.98, sub_questions=("SQ1",)),
            candidate("https://c.example.com/3", score=0.97, sub_questions=("SQ1",)),
            candidate("https://d.example.com/4", score=0.50, sub_questions=("SQ2",)),
            candidate("https://e.example.com/5", score=0.40, sub_questions=("SQ3",)),
        ]
        chosen = select_candidates(pool, limit=3, explanatory=True)
        covered = {sq for c in chosen for sq in c.sub_question_ids}
        assert covered == {"SQ1", "SQ2", "SQ3"}

    def test_selection_explains_what_it_dropped(self) -> None:
        from agentic_research.retrieval.selection import select_with_diagnostics

        pool = [
            candidate("https://x.com/a/status/1", score=0.87),
            candidate("https://arxiv.org/abs/1234", score=0.84),
        ]
        result = select_with_diagnostics(pool, limit=1, explanatory=True)
        assert [c.domain for c in result.selected] == ["arxiv.org"]
        assert result.dropped
        reason = result.dropped[0].reason
        assert "social" in reason or "authority" in reason


class TestACandidateServingSeveralSubQuestions:
    """Selected once, credited to every sub-question it serves.

    The earlier version bucketed such a candidate under each
    sub-question and then recorded it against whichever bucket claimed
    it first. Two wrong outcomes followed: a sub-question whose own
    query returned the selected page was reported as having no source,
    so critique published a false "no suitable source found" gap; and
    the bucket that lost the race spent another fetch on a
    sub-question already represented by that same page.
    """

    def test_one_shared_candidate_credits_both_sub_questions(self) -> None:
        from agentic_research.retrieval.selection import select_with_diagnostics

        shared = candidate("https://a.example.com/1", score=0.90, sub_questions=("SQ1", "SQ2"))
        result = select_with_diagnostics([shared], limit=2, sub_question_order=["SQ1", "SQ2"])
        assert len(result.selected) == 1, "one page, one fetch"
        assert result.by_sub_question["SQ1"] == ["https://a.example.com/1"]
        assert result.by_sub_question["SQ2"] == ["https://a.example.com/1"]
        assert result.sub_questions_with_no_source == ()

    def test_no_false_starvation_is_reported(self) -> None:
        """The specific false gap cause: critique would have said no
        suitable source was found for SQ2."""
        from agentic_research.retrieval.selection import select_with_diagnostics

        shared = candidate("https://a.example.com/1", score=0.90, sub_questions=("SQ1", "SQ2"))
        result = select_with_diagnostics([shared], limit=1, sub_question_order=["SQ1", "SQ2"])
        assert "SQ2" not in result.sub_questions_with_no_source

    def test_a_shared_candidate_does_not_consume_a_second_slot(self) -> None:
        """A shared page plus a unique page covers three sub-questions
        in two fetches. Previously SQ2 would have taken a duplicate and
        SQ3 would have gone without."""
        from agentic_research.retrieval.selection import select_with_diagnostics

        shared = candidate("https://a.example.com/1", score=0.99, sub_questions=("SQ1", "SQ2"))
        second_for_sq2 = candidate("https://b.example.com/2", score=0.98, sub_questions=("SQ2",))
        unique = candidate("https://c.example.com/3", score=0.10, sub_questions=("SQ3",))
        result = select_with_diagnostics(
            [shared, second_for_sq2, unique],
            limit=2,
            sub_question_order=["SQ1", "SQ2", "SQ3"],
        )
        domains = sorted(c.domain for c in result.selected)
        assert domains == ["a.example.com", "c.example.com"]
        assert result.sub_questions_with_no_source == ()

    def test_planner_priority_still_orders_selection(self) -> None:
        """Determinism and priority: the first-listed sub-question is
        served first when the budget cannot cover every one."""
        from agentic_research.retrieval.selection import select_with_diagnostics

        first = candidate("https://first.example.com/1", score=0.10, sub_questions=("SQ1",))
        second = candidate("https://second.example.com/2", score=0.99, sub_questions=("SQ2",))
        result = select_with_diagnostics(
            [first, second], limit=1, sub_question_order=["SQ1", "SQ2"]
        )
        assert [c.domain for c in result.selected] == ["first.example.com"]

    def test_selection_is_deterministic_under_input_reordering(self) -> None:
        from agentic_research.retrieval.selection import select_with_diagnostics

        pool = [
            candidate("https://a.example.com/1", score=0.90, sub_questions=("SQ1", "SQ2")),
            candidate("https://b.example.com/2", score=0.50, sub_questions=("SQ3",)),
            candidate("https://c.example.com/3", score=0.40, sub_questions=("SQ2",)),
        ]
        order = ["SQ1", "SQ2", "SQ3"]
        forward = select_with_diagnostics(pool, limit=2, sub_question_order=order)
        backward = select_with_diagnostics(list(reversed(pool)), limit=2, sub_question_order=order)
        assert [c.canonical_url for c in forward.selected] == [
            c.canonical_url for c in backward.selected
        ]
        assert forward.by_sub_question == backward.by_sub_question

    def test_allocation_is_not_evidence_coverage(self) -> None:
        """Crediting a shared page to two sub-questions says it was
        retrieved for both and nothing more. Whether its text addresses
        either is decided by topicality against the extracted quote --
        so a page allocated to two sub-questions can still be on topic
        for only one, and coverage must reflect that rather than the
        allocation.
        """
        from agentic_research.evidence.topicality import lexically_plausible
        from agentic_research.retrieval.selection import select_with_diagnostics

        shared = candidate("https://a.example.com/1", score=0.9, sub_questions=("SQ1", "SQ2"))
        result = select_with_diagnostics([shared], limit=1, sub_question_order=["SQ1", "SQ2"])
        assert result.by_sub_question["SQ1"] and result.by_sub_question["SQ2"]

        capacity_sq = "How does excessive model capacity cause overfitting?"
        leakage_sq = "How does data leakage conceal overfitting in validation splits?"
        extracted = (
            "If the capacity is too high relative to the available data, the model "
            "fits random noise in the training set."
        )
        # One page, allocated to both, on topic for exactly one.
        assert lexically_plausible(capacity_sq, extracted)
        assert not lexically_plausible(leakage_sq, extracted)


class TestOnceEveryoneIsRepresentedQualityDecides:
    """Breadth is a coverage argument; depth is a quality one.

    The depth pass used to rotate by sub-question, and the synthetic
    manifest example showed what that cost: with every sub-question
    already represented, rotation took SQ1's second-best candidate --
    a tweet at 0.71 -- ahead of SQ2's second-best, a reference at 0.87.
    A social source displacing a more relevant accountable one is
    exactly what the authority adjustment exists to prevent, so it
    cannot be reintroduced by the allocation order.
    """

    def test_the_spare_slot_goes_to_the_best_remaining_candidate(self) -> None:
        from agentic_research.retrieval.selection import select_with_diagnostics

        pool = [
            # Serves both SQ1 and SQ2, so one fetch represents both.
            candidate("https://arxiv.org/abs/1", score=0.84, sub_questions=("SQ1", "SQ2")),
            candidate("https://x.com/a/status/1", score=0.91, sub_questions=("SQ1",)),
            candidate("https://en.wikipedia.org/wiki/X", score=0.79, sub_questions=("SQ2",)),
        ]
        result = select_with_diagnostics(
            pool, limit=2, explanatory=True, sub_question_order=["SQ1", "SQ2"]
        )
        domains = sorted(c.domain for c in result.selected)
        assert domains == ["arxiv.org", "en.wikipedia.org"], (
            "the spare slot went to a social source over a better reference"
        )

    def test_breadth_still_wins_while_a_subquestion_has_nothing(self) -> None:
        """Non-vacuity in the other direction: a lower-scoring candidate
        is still taken when it is the only thing serving its
        sub-question."""
        from agentic_research.retrieval.selection import select_with_diagnostics

        pool = [
            candidate("https://arxiv.org/abs/1", score=0.99, sub_questions=("SQ1",)),
            candidate("https://arxiv.org/abs/2", score=0.98, sub_questions=("SQ1",)),
            candidate("https://tiny.example.com/p", score=0.10, sub_questions=("SQ2",)),
        ]
        result = select_with_diagnostics(
            pool, limit=2, explanatory=True, sub_question_order=["SQ1", "SQ2"]
        )
        covered = {sq for c in result.selected for sq in c.sub_question_ids}
        assert covered == {"SQ1", "SQ2"}

    def test_it_stays_deterministic(self) -> None:
        from agentic_research.retrieval.selection import select_with_diagnostics

        pool = [
            candidate("https://a.example.com/1", score=0.80, sub_questions=("SQ1",)),
            candidate("https://b.example.com/2", score=0.80, sub_questions=("SQ2",)),
            candidate("https://c.example.com/3", score=0.80, sub_questions=("SQ1",)),
        ]
        order = ["SQ1", "SQ2"]
        first = select_with_diagnostics(pool, limit=3, sub_question_order=order)
        second = select_with_diagnostics(list(reversed(pool)), limit=3, sub_question_order=order)
        assert [c.canonical_url for c in first.selected] == [
            c.canonical_url for c in second.selected
        ]
