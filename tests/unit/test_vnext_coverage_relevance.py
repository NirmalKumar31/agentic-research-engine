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

from agentic_research.evidence.topicality import (
    lexically_plausible,
    overlap,
    salient_terms,
)

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
    def test_the_capacity_question_admits_its_own_evidence(self) -> None:
        assert lexically_plausible(CAPACITY_SQ, ON_TOPIC_QUOTE)
        assert overlap(CAPACITY_SQ, ON_TOPIC_QUOTE) >= 2

    def test_the_capacity_question_rejects_the_double_descent_paper(self) -> None:
        assert not lexically_plausible(CAPACITY_SQ, DOUBLE_DESCENT_QUOTE)

    def test_the_leakage_question_rejects_both(self) -> None:
        """Neither quote is about data leakage. Both were counted."""
        assert not lexically_plausible(LEAKAGE_SQ, ON_TOPIC_QUOTE)
        assert not lexically_plausible(LEAKAGE_SQ, DOUBLE_DESCENT_QUOTE)


class TestThePrefilterIsDeliberatelyWeak:
    def test_one_coincidental_term_is_not_enough(self) -> None:
        """Almost any machine-learning text shares one term with any
        other, and that coincidence is what let the papers through."""
        assert overlap("How large should the training data be?", "data centres use power") >= 1
        assert not lexically_plausible(
            "How large should the training data be?", "data centres use power"
        )

    def test_a_question_with_no_salient_terms_abstains(self) -> None:
        """It cannot call evidence off-topic without knowing the topic.
        Rejecting here would punish a run for a degenerate question."""
        assert lexically_plausible("q", "anything at all")
        assert lexically_plausible("", "anything at all")

    def test_evidence_with_no_content_words_is_rejected(self) -> None:
        """The asymmetry, on purpose: abstaining here would reinstate
        the defect for the evidence that carries least."""
        assert not lexically_plausible(CAPACITY_SQ, "")
        assert not lexically_plausible(CAPACITY_SQ, "a b c")

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
        assert coverage.gap_cause == "retrieved source did not discuss this sub-question"
        assert coverage.off_topic_items == 2
        assert "none discussing this" in coverage.note

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


class TestTheLadderIsNamedNotCollapsed:
    """Four things get confused if they share one word.

    The earlier name for the lexical check was ``is_topical``, which
    invited exactly the reading it must not have: that two shared words
    show a source answers a question. Each level now has a name, and
    the report states which one it reached.
    """

    def test_every_level_is_named(self) -> None:
        from agentic_research.evidence.topicality import RelevanceLevel

        assert [level.value for level in RelevanceLevel] == [
            "lexically_plausible",
            "evidence_relevant",
            "claim_relevant",
            "slot_satisfied",
        ]

    def test_the_weak_check_is_not_called_topical_any_more(self) -> None:
        """Non-vacuity for the rename: the old name is gone, so nothing
        can import it and read a semantic judgement into it."""
        import agentic_research.evidence.topicality as module

        assert not hasattr(module, "is_topical")
        assert hasattr(module, "lexically_plausible")

    def test_user_facing_gap_causes_do_not_say_topical(self) -> None:
        """A reader must not be told "off topic" as though relevance
        had been assessed. The cause states what was actually
        checked."""
        from agentic_research.evidence.store import EvidenceStore
        from agentic_research.models import EvidenceItem, QuoteMatch, SourceDocument, SubQuestion

        doc = SourceDocument(
            id="S1",
            url="https://e.example.com/p",
            canonical_url="https://e.example.com/p",
            title="t",
            domain="e.example.com",
            text=f"Preamble. {DOUBLE_DESCENT_QUOTE} End.",
        )
        item = EvidenceItem(
            id="S1-e1",
            source_id="S1",
            sub_question_id="SQ1",
            claim=DOUBLE_DESCENT_QUOTE,
            quote=DOUBLE_DESCENT_QUOTE,
            quote_match=QuoteMatch.EXACT_NORMALIZED,
        )
        coverage = EvidenceStore([doc], [item]).coverage_for(
            SubQuestion(id="SQ1", text=CAPACITY_SQ, rationale="r")
        )
        assert "topic" not in coverage.gap_cause
        assert coverage.gap_cause == "retrieved source did not discuss this sub-question"


class TestTerminologyMismatch:
    """Correct evidence using a different name for the same thing.

    A lexical guard that rejects the established synonym of a term is
    worse than no guard: it hides good evidence and reports a gap that
    is not there.
    """

    def test_a_hyphenated_compound_reaches_its_bare_form(self) -> None:
        """ "self-attention" and "scaled dot-product attention" are the
        same mechanism. The compound and the bare word are different
        tokens, so without splitting on the hyphen the correct evidence
        was rejected over punctuation."""
        assert lexically_plausible(
            "How does self-attention work in transformers?",
            "Scaled dot-product attention computes a weighted sum over value vectors.",
        )

    def test_an_acronym_reaches_its_expansion(self) -> None:
        from agentic_research.evidence.topicality import alias_groups

        aliases = alias_groups("", ["retrieval-augmented generation", "RAG"])
        assert lexically_plausible(
            "What is RAG used for?",
            "Retrieval-augmented generation retrieves passages at query time.",
            aliases=aliases,
        )

    def test_an_expansion_reaches_its_acronym(self) -> None:
        from agentic_research.evidence.topicality import alias_groups

        aliases = alias_groups("", ["large language model", "LLM"])
        assert lexically_plausible(
            "What limits an LLM context window?",
            "A large language model has a fixed context window.",
            aliases=aliases,
        )

    def test_a_parenthetical_gloss_declares_its_own_alias(self) -> None:
        from agentic_research.evidence.topicality import alias_groups

        groups = alias_groups("What is retrieval-augmented generation (RAG)?", [])
        assert any("rag" in g and "generation" in g for g in groups)

    def test_aliases_are_bounded_to_what_the_run_names(self) -> None:
        """No global synonym universe. With nothing named, nothing is
        aliased -- so the aliases available to a run are auditable from
        that run's own analysis."""
        from agentic_research.evidence.topicality import alias_groups

        assert alias_groups("", []) == ()
        assert alias_groups("What is overfitting?", ["overfitting"]) == ()

    def test_a_synonym_the_run_never_names_is_not_linked(self) -> None:
        """A recorded limitation, stated rather than papered over.

        "Overfitting" and "poor generalization" are the same phenomenon
        and share no salient term. Nothing in the question or the
        analysis links them, and a hand-written synonym table is
        exactly the unbounded, unauditable thing this avoids -- so the
        prefilter does not link them either.

        What bounds the harm: this decides only whether an item counts
        toward *coverage*. The evidence is still retained, the cause is
        recorded as "retrieved source did not discuss this
        sub-question" rather than as an absence of evidence, and a
        published claim's relevance is decided a level up by the judge,
        which reads meaning rather than tokens.
        """
        assert not lexically_plausible(
            "What causes overfitting?",
            "Models that memorise noise show poor generalization on unseen data.",
        )

    def test_the_same_pair_links_once_the_analysis_names_both(self) -> None:
        """And the bound is not a dead end: naming the second term in
        the question makes the link available, because the context has
        then established it."""
        assert lexically_plausible(
            "What causes overfitting and poor generalization?",
            "Models that memorise noise show poor generalization on unseen data.",
        )


class TestTheTopicTermDoesNotAdmitEverything:
    """Found by the offline evaluation, not by a test.

    For "what are the main causes of overfitting", every sub-question
    and every extracted quote contains "overfitting" -- eleven
    characters, so the distinctive-term shortcut admitted all ten
    excerpts for all five sub-questions, including double-descent
    papers as evidence about data leakage. A term the whole run shares
    carries no information about *which* sub-question a quote bears on.
    """

    QUESTION = "What are the main causes of overfitting in machine learning?"
    LEAKAGE_SQ = (
        "How can data leakage or repeated use of validation/test data create "
        "misleading performance estimates and conceal overfitting?"
    )

    def test_the_topic_word_alone_does_not_admit_a_quote(self) -> None:
        from agentic_research.evidence.topicality import salient_terms

        topic = salient_terms(self.QUESTION)
        # Shares "overfitting" with the sub-question and nothing else.
        quote = "Benign overfitting occurs in overparameterised linear regression."
        assert not lexically_plausible(self.LEAKAGE_SQ, quote, topic_terms=topic)

    def test_without_the_exclusion_it_would_be_admitted(self) -> None:
        """Non-vacuity: the exclusion is doing the work, not the rest of
        the rule."""
        quote = "Benign overfitting occurs in overparameterised linear regression."
        assert lexically_plausible(self.LEAKAGE_SQ, quote)

    def test_a_discriminating_distinctive_term_still_admits(self) -> None:
        from agentic_research.evidence.topicality import salient_terms

        topic = salient_terms(self.QUESTION)
        quote = (
            "Repeated evaluation on the validation split leaks information and "
            "inflates the estimate."
        )
        assert lexically_plausible(self.LEAKAGE_SQ, quote, topic_terms=topic)

    def test_pure_function_words_are_not_salient(self) -> None:
        """ "and" was counting as a shared content word, which the
        offline evaluation surfaced in its shared-term output."""
        from agentic_research.evidence.topicality import salient_terms

        terms = salient_terms("causes and reasons for this but not all of them")
        for word in ("and", "for", "but", "all"):
            assert word not in terms, word
        assert "causes" in terms
        assert "reasons" in terms
