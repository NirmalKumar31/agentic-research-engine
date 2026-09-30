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
