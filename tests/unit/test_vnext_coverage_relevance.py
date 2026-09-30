"""A quote being exact does not make it an answer.

`quote_verified` is a provenance property: the quote appears verbatim in
the source that cites it. Coverage treated it as sufficient -- two
exact-match items across two sources marked a sub-question covered --
so a sub-question was "covered" by quotes that addressed something
else.

The live run this comes from asked for the main causes of overfitting.
Retrieval returned papers on double descent and frozen
overparameterization; the extractor pulled exact quotes from them and
attributed them to sub-questions about training-data size and data
leakage. The quotes below are verbatim from that run's artifacts.
"""

from __future__ import annotations

from agentic_research.evidence.topicality import is_topical, overlap, salient_terms

# Verbatim from the live run's preserved source excerpts.
CAPACITY_SQ = (
    "How does excessive model capacity or complexity relative to the amount "
    "of training data cause overfitting?"
)
LEAKAGE_SQ = (
    "How can data leakage or repeated use of validation/test data create "
    "misleading performance estimates and conceal overfitting?"
)
ON_TOPIC_QUOTE = (
    "If the capacity is too high relative to the amount of available data, the "
    "model may not only fit the underlying trend but also the random noise "
    "present in the training set."
)
DOUBLE_DESCENT_QUOTE = (
    "the number of frozen layers can determine whether the transfer learning is "
    "effectively underparameterized or overparameterized and, in turn, this may "
    "induce a freezing-wise double descent phenomenon"
)


class TestTheLiveRunsEvidence:
    def test_the_capacity_question_accepts_its_own_evidence(self) -> None:
        assert is_topical(CAPACITY_SQ, ON_TOPIC_QUOTE)
        assert overlap(CAPACITY_SQ, ON_TOPIC_QUOTE) >= 2

    def test_the_capacity_question_rejects_the_double_descent_paper(self) -> None:
        assert not is_topical(CAPACITY_SQ, DOUBLE_DESCENT_QUOTE)

    def test_the_leakage_question_rejects_both(self) -> None:
        """Neither quote is about data leakage. Both were counted."""
        assert not is_topical(LEAKAGE_SQ, ON_TOPIC_QUOTE)
        assert not is_topical(LEAKAGE_SQ, DOUBLE_DESCENT_QUOTE)


class TestTheFloorIsDeliberatelyLow:
    def test_one_coincidental_term_is_not_enough(self) -> None:
        """Almost any machine-learning text shares one term with any
        other, and that coincidence is what let the papers through."""
        assert overlap("How large should the training data be?", "data centres use power") >= 1
        assert not is_topical("How large should the training data be?", "data centres use power")

    def test_a_question_with_no_salient_terms_abstains(self) -> None:
        """It cannot call evidence off-topic without knowing the topic.
        Rejecting here would punish a run for a degenerate question."""
        assert is_topical("q", "anything at all")
        assert is_topical("", "anything at all")

    def test_evidence_with_no_content_words_is_rejected(self) -> None:
        """The asymmetry, on purpose: abstaining here would reinstate
        the defect for the evidence that carries least."""
        assert not is_topical(CAPACITY_SQ, "")
        assert not is_topical(CAPACITY_SQ, "a b c")

    def test_stopwords_do_not_count_as_shared_subject_matter(self) -> None:
        assert "the" not in salient_terms("the model and the data")
        assert "model" in salient_terms("the model and the data")


class TestCoverageReportsWhyAGapExists:
    def _store(self, claim: str, quote: str, sources: int = 2):
        from agentic_research.evidence.store import EvidenceStore
        from agentic_research.models import (
            EvidenceItem,
            QuoteMatch,
            SourceDocument,
        )

        docs = [
            SourceDocument(
                id=f"S{i + 1}",
                url=f"https://e{i}.example.com/p",
                canonical_url=f"https://e{i}.example.com/p",
                title="t",
                domain=f"e{i}.example.com",
                # The quote must appear in the source, because the store
                # re-verifies it there rather than trusting the flag.
                text=f"Preamble. {quote} Trailing text.",
            )
            for i in range(sources)
        ]
        items = [
            EvidenceItem(
                id=f"S{i + 1}-e1",
                source_id=f"S{i + 1}",
                sub_question_id="SQ1",
                claim=claim,
                quote=quote,
                # `quote_verified` is computed from this, not set: only
                # an exact-normalised match counts as verified.
                quote_match=QuoteMatch.EXACT_NORMALIZED,
            )
            for i in range(sources)
        ]
        return EvidenceStore(docs, items)

    def _sq(self, text: str):
        from agentic_research.models import SubQuestion

        return SubQuestion(id="SQ1", text=text, rationale="r")

    def test_on_topic_evidence_from_two_sources_covers(self) -> None:
        coverage = self._store(ON_TOPIC_QUOTE, ON_TOPIC_QUOTE).coverage_for(self._sq(CAPACITY_SQ))
        assert coverage.verdict == "covered"
        assert coverage.gap_cause == ""

    def test_exact_but_off_topic_evidence_is_uncovered_and_says_so(self) -> None:
        """The defect: this was 'covered'."""
        coverage = self._store(DOUBLE_DESCENT_QUOTE, DOUBLE_DESCENT_QUOTE).coverage_for(
            self._sq(CAPACITY_SQ)
        )
        assert coverage.verdict == "uncovered"
        assert coverage.gap_cause == "retrieved source was off topic"
        assert coverage.off_topic_items == 2
        assert "none on topic" in coverage.note

    def test_a_single_on_topic_source_is_weak_not_covered(self) -> None:
        coverage = self._store(ON_TOPIC_QUOTE, ON_TOPIC_QUOTE, sources=1).coverage_for(
            self._sq(CAPACITY_SQ)
        )
        assert coverage.verdict == "weak"
        assert coverage.gap_cause == "evidence insufficient"

    def test_no_evidence_at_all_is_distinguished(self) -> None:
        from agentic_research.evidence.store import EvidenceStore

        coverage = EvidenceStore([], []).coverage_for(self._sq(CAPACITY_SQ))
        assert coverage.gap_cause == "no evidence attributed"


class TestRetrievalSideCausesAreAttributed:
    """Four causes, four different fixes. A gap with no diagnosis is
    indistinguishable from a gap with a different cause."""

    def _coverage(self, **kw):
        from agentic_research.models import SubQuestionCoverage

        base = {
            "sub_question_id": "SQ1",
            "verdict": "uncovered",
            "gap_cause": "no evidence attributed",
        }
        base.update(kw)
        return SubQuestionCoverage(**base)

    def test_a_starved_subquestion_is_named_as_such(self) -> None:
        from agentic_research.graph.nodes.critique import _attribute_gaps

        out = _attribute_gaps(
            [self._coverage()],
            {  # type: ignore[arg-type]
                "retrieval_diagnostics": {
                    "starved_sub_questions": ["SQ1"],
                    "by_sub_question": {"SQ1": []},
                },
                "sources": [],
            },
        )
        assert out[0].gap_cause == "no suitable source found"

    def test_a_covered_subquestion_is_left_alone(self) -> None:
        from agentic_research.graph.nodes.critique import _attribute_gaps

        out = _attribute_gaps(
            [self._coverage(verdict="covered", gap_cause="")],
            {"retrieval_diagnostics": {}, "sources": []},  # type: ignore[arg-type]
        )
        assert out[0].gap_cause == ""

    def test_a_fetch_failure_is_distinguished_from_a_missing_source(self) -> None:
        from agentic_research.graph.nodes.critique import _attribute_gaps
        from agentic_research.models import FetchStatus, SourceDocument

        failed = SourceDocument(
            id="S1",
            url="https://e.example.com/p",
            canonical_url="https://e.example.com/p",
            title="t",
            domain="e.example.com",
            fetch_status=FetchStatus.HTTP_ERROR,
        )
        out = _attribute_gaps(
            [self._coverage(evidence_count=0)],
            {  # type: ignore[arg-type]
                "retrieval_diagnostics": {
                    "starved_sub_questions": [],
                    "by_sub_question": {"SQ1": ["https://e.example.com/p"]},
                },
                "sources": [failed],
            },
        )
        assert out[0].gap_cause == "candidate selected but fetch failed"

    def test_missing_diagnostics_do_not_invent_a_cause(self) -> None:
        """A run without retrieval diagnostics must not be reported as
        having found no suitable source."""
        from agentic_research.graph.nodes.critique import _attribute_gaps

        out = _attribute_gaps(
            [self._coverage(evidence_count=3, gap_cause="evidence insufficient")],
            {"sources": []},  # type: ignore[arg-type]
        )
        assert out[0].gap_cause == "no suitable source found"
