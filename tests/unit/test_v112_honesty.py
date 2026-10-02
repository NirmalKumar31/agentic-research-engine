"""Two defects that made a truthful report look untrustworthy.

Both were read off the v1.11 hosted run on "langchain vs langgraph
differences?". Neither changed a verdict; both changed whether a reader
could believe the verdict.

Tested through the functions a report is actually built from, not
against the formatting helpers alone -- the helper is where the value is
declared, the verdict is where it runs.
"""

from __future__ import annotations

import pytest

from agentic_research.citations.atomicity import _carries_predicate, compound_propositions
from agentic_research.citations.fake_nli import FakeScorer
from agentic_research.citations.semantic import _below, _threshold, verify_claim

CLAIM = "LangChain agents do not provide state management."
QUOTE = "LangChain agents do not provide state management."


def reason_at(entailment: float, *, threshold: float = 0.98) -> str:
    scorer = FakeScorer(default=(entailment, 0.0, 1.0 - entailment))
    verdict = verify_claim(CLAIM, [("E1", QUOTE)], scorer, support_threshold=threshold)
    assert not verdict.publishable
    return verdict.reason


class TestANearMissDoesNotRenderAsAboveTheThreshold:
    """The live wording was:

        best entailment 0.980 ... is below the 0.98 support threshold

    which reads as though the engine cannot compare two floats. The true
    score was under 0.98; `f"{0.97951:.3f}"` is `"0.980"`. It appears on
    near-miss claims, which are the ones a reader checks hardest.
    """

    @pytest.mark.parametrize("entailment", [0.97951, 0.9799, 0.979999, 0.9799999999])
    def test_the_printed_score_never_reaches_the_printed_threshold(self, entailment: float) -> None:
        reason = reason_at(entailment)
        assert "0.980 " not in reason, reason
        assert "below the 0.98 support threshold" in reason

    def test_the_score_is_truncated_not_rounded(self) -> None:
        """Rounding is what produced the defect; truncation cannot."""
        assert _below(0.97951) == "0.9795"
        assert _below(0.9799999) == "0.9799"

    def test_closeness_is_still_reported(self) -> None:
        """The number exists to say how close it came; keep the detail."""
        assert "0.9795" in reason_at(0.97951)

    def test_a_distant_score_still_reads_naturally(self) -> None:
        assert "0.0130" in reason_at(0.013)

    def test_the_threshold_prints_at_its_own_precision(self) -> None:
        """The other half of the defect.

        At `.2f` a threshold of 0.985 printed as "0.98", so a truthful
        0.9840 read as though it were above the bar.
        """
        assert _threshold(0.985) == "0.985"
        assert _threshold(0.98) == "0.98"
        assert "below the 0.985 support threshold" in reason_at(0.984, threshold=0.985)


class TestARelativeClauseIsNotASecondProposition:
    """`compound_propositions` split a compound *noun phrase* at "and".

    "a graph of nodes and edges that supports flexible data flow" became
    "edges that supports flexible data flow", and "supports" -- whose
    subject is the relative pronoun -- was counted as a second clause's
    verb. The v1.11 run refused a true claim on it. The preposition fix
    shipped in v1.11 was narrower than the problem.
    """

    LIVE_FALSE_POSITIVE = (
        "LangGraph's core abstraction is a graph of nodes and edges that "
        "supports flexible data flow and dynamic decision paths."
    )

    def test_the_live_claim_reads_as_one_proposition(self) -> None:
        assert compound_propositions(self.LIVE_FALSE_POSITIVE) == []

    @pytest.mark.parametrize(
        "segment",
        [
            "edges that requires review",
            "a store which requires configuration",
            "a maintainer who requires approval",
        ],
    )
    def test_a_mid_segment_relative_pronoun_carries_no_assertion(self, segment: str) -> None:
        assert not _carries_predicate(segment)

    @pytest.mark.parametrize(
        "segment",
        [
            # "that" leading the segment is a subject in its own right,
            # not a relative pronoun attached to a preceding noun.
            "that requires a separate service",
            "which requires a separate service",
        ],
    )
    def test_a_leading_relative_pronoun_still_carries_one(self, segment: str) -> None:
        assert _carries_predicate(segment)

    def test_the_v111_preposition_fix_still_holds(self) -> None:
        assert (
            compound_propositions(
                "LangGraph's persistence layer gives agents short-term memory through "
                "checkpointers and long-term memory through stores."
            )
            == []
        )

    @pytest.mark.parametrize(
        "text",
        [
            "For a single process doing reads and small writes, SQLite is hard to "
            "beat, and it often outruns PostgreSQL on simple queries.",
            "The planner emitted three queries, and the critic requested another round.",
        ],
    )
    def test_genuine_compounds_still_fire(self, text: str) -> None:
        assert compound_propositions(text) != []

    def test_multiple_sentences_still_fire(self) -> None:
        assert compound_propositions("SQLite serializes writes. PostgreSQL does not.") != []


class TestTheQueryWriterIsToldToAskPrimarySources:
    """Retrieval, not verification, is what now limits the answer.

    Two hosted runs on "langchain vs langgraph differences?" six hours
    apart: the first retrieved `docs.langchain.com` three times and
    `reference.langchain.com` once (0.92-0.96); the second retrieved six
    commentary articles, ceiling 0.64, and the report quoted a blog on
    LangGraph's state model while the first-party docs were never a
    candidate. Same prompt, same question -- so primary sources were
    being reached by luck.

    **This is a prompt rule, and these tests prove only that the rule is
    asked for.** Whether it changes what comes back is a live-retrieval
    question that no offline test can answer.

    A deterministic version was built first and withdrawn: it injected
    "<subject> official documentation" for every entity, which produced
    "cost-sensitive learning official documentation" on a methods
    question. Narrowing it to comparisons, then to single-token
    capitalised subjects, still fired on `SMOTE` -- an algorithm, not a
    product. Deciding which subjects have a maintainer is not reliably
    solvable in code, which is the conclusion this repo already reached
    about first-party-docs detection; the model writing the query can
    judge it, so it is asked to.
    """

    def test_the_rule_is_in_the_prompt(self) -> None:
        from agentic_research.graph.prompts import QUERY_WRITER_SYSTEM

        lowered = QUERY_WRITER_SYSTEM.lower()
        assert "maintainer" in lowered
        assert "documentation" in lowered

    def test_the_counter_rule_is_there_too(self) -> None:
        """Without it the model asks a technique for its documentation."""
        from agentic_research.graph.prompts import QUERY_WRITER_SYSTEM

        lowered = QUERY_WRITER_SYSTEM.lower()
        assert "technique" in lowered
        assert "cost-sensitive learning" in lowered

    def test_site_filters_are_still_banned(self) -> None:
        """The rule changes what is *asked for*, not where it is asked.

        `site:` filters and a domain allowlist are both still refused --
        what comes back is ranked on its merits by the authority
        adjustment that already exists.
        """
        from agentic_research.graph.prompts import QUERY_WRITER_SYSTEM

        assert "site: filters" in QUERY_WRITER_SYSTEM

    def test_the_question_vocabulary_rule_survives(self) -> None:
        """The new rule is a counterweight to it, not a replacement.

        Dropping it would bring back the failure the prompt opens with:
        a query of specialist terms that returns only specialist papers.
        """
        from agentic_research.graph.prompts import QUERY_WRITER_SYSTEM

        assert "Prefer the question's own vocabulary" in QUERY_WRITER_SYSTEM
