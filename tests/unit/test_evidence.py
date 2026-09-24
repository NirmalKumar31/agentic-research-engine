"""Deduplication, quality scoring, quote verification and packaging."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agentic_research.evidence import (
    EvidenceStore,
    classify_quote,
    classify_source,
    dedupe_by_content,
    dedupe_search_results,
    domain_concentration,
    score_source,
    verify_quote,
)
from agentic_research.models import (
    EvidenceItem,
    QuoteMatch,
    SearchResult,
    SourceDocument,
    SourceType,
    Stance,
    SubQuestion,
)


def result(
    url: str, *, query_id: str = "Q1", score: float = 0.5, title: str = "T", raw: str | None = None
) -> SearchResult:
    return SearchResult(
        url=url, title=title, snippet="s", score=score, query_id=query_id, raw_content=raw
    )


def source(
    sid: str,
    *,
    text: str = "body text here",
    domain: str = "a.com",
    usable: bool = True,
    quality: float = 0.7,
) -> SourceDocument:
    return SourceDocument(
        id=sid,
        url=f"https://{domain}/{sid}",
        canonical_url=f"https://{domain}/{sid}",
        title=f"Title {sid}",
        domain=domain,
        text=text if usable else "",
        content_hash=SourceDocument.hash_text(text) if usable else "",
        quality_score=quality,
    )


def evidence(
    eid: str,
    sid: str,
    sqid: str,
    *,
    verified: bool = True,
    stance: Stance = Stance.SUPPORTS,
    relevance: float = 0.8,
) -> EvidenceItem:
    return EvidenceItem(
        id=eid,
        source_id=sid,
        sub_question_id=sqid,
        claim=f"claim {eid}",
        quote=f"quote for {eid}",
        stance=stance,
        relevance=relevance,
        quote_match=QuoteMatch.EXACT_NORMALIZED if verified else QuoteMatch.NONE,
    )


class TestPreFetchDeduplication:
    """The barrier that makes one unique page cost one fetch."""

    def test_same_page_from_three_queries_becomes_one_candidate(self) -> None:
        results = [
            result("https://www.ex.com/a?utm_source=x", query_id="Q1"),
            result("https://ex.com/a/", query_id="Q2"),
            result("https://ex.com/a#top", query_id="Q3"),
        ]
        mapping = {"Q1": "SQ1", "Q2": "SQ2", "Q3": "SQ3"}
        candidates, stats = dedupe_search_results(results, mapping)

        assert len(candidates) == 1
        assert stats.duplicate_urls == 2
        assert stats.fetches_avoided == 2
        # Provenance is unioned, not discarded: all three queries are recorded.
        assert sorted(candidates[0].found_by_queries) == ["Q1", "Q2", "Q3"]
        assert sorted(candidates[0].sub_question_ids) == ["SQ1", "SQ2", "SQ3"]

    def test_keeps_best_score_and_any_available_content(self) -> None:
        results = [
            result("https://ex.com/a", query_id="Q1", score=0.4),
            result("https://ex.com/a", query_id="Q2", score=0.9, raw="page body"),
        ]
        candidates, _ = dedupe_search_results(results, {"Q1": "SQ1", "Q2": "SQ2"})
        assert candidates[0].best_score == 0.9
        assert candidates[0].provider_content == "page body"

    def test_previously_known_urls_are_not_refetched(self) -> None:
        results = [result("https://ex.com/a"), result("https://ex.com/b")]
        candidates, stats = dedupe_search_results(
            results, {"Q1": "SQ1"}, known_canonical_urls={"https://ex.com/a"}
        )
        assert [c.canonical_url for c in candidates] == ["https://ex.com/b"]
        assert stats.already_known == 1

    def test_near_identical_titles_on_same_domain_merge(self) -> None:
        results = [
            result("https://ex.com/guide", query_id="Q1", title="Fraud Detection Guide"),
            result("https://ex.com/guide-v2", query_id="Q2", title="Fraud Detection Guide"),
        ]
        candidates, stats = dedupe_search_results(results, {"Q1": "SQ1", "Q2": "SQ2"})
        assert len(candidates) == 1
        assert stats.duplicate_titles == 1

    @pytest.mark.parametrize(
        ("title_a", "title_b"),
        [
            ("Deep Learning Tutorial Part 1", "Deep Learning Tutorial Part 2"),
            ("State of AI Report 2024", "State of AI Report 2025"),
            ("Result 0 for imbalanced data", "Result 1 for imbalanced data"),
            ("Migrating to Python 2 Runtime", "Migrating to Python 3 Runtime"),
        ],
    )
    def test_titles_differing_only_by_number_are_not_merged(
        self, title_a: str, title_b: str
    ) -> None:
        """Numbers in a title are usually what distinguishes the documents.
        String similarity alone rates these above any useful threshold."""
        results = [
            result("https://ex.com/a", query_id="Q1", title=title_a),
            result("https://ex.com/b", query_id="Q2", title=title_b),
        ]
        candidates, _ = dedupe_search_results(results, {"Q1": "SQ1", "Q2": "SQ2"})
        assert len(candidates) == 2

    def test_very_short_titles_never_merge(self) -> None:
        results = [
            result("https://ex.com/a", query_id="Q1", title="Docs"),
            result("https://ex.com/b", query_id="Q2", title="Docs"),
        ]
        candidates, _ = dedupe_search_results(results, {"Q1": "SQ1", "Q2": "SQ2"})
        assert len(candidates) == 2

    def test_identical_titles_on_different_domains_do_not_merge(self) -> None:
        """Two sites can both publish 'Getting Started'; merging them would
        also destroy the independent-corroboration signal."""
        results = [
            result("https://a.com/x", query_id="Q1", title="Getting Started"),
            result("https://b.com/y", query_id="Q2", title="Getting Started"),
        ]
        candidates, _ = dedupe_search_results(results, {"Q1": "SQ1", "Q2": "SQ2"})
        assert len(candidates) == 2

    def test_duplicate_rate_metric(self) -> None:
        results = [result("https://ex.com/a", query_id=f"Q{i}") for i in range(4)]
        _, stats = dedupe_search_results(results, {})
        assert stats.results_in == 4
        assert stats.unique_out == 1
        assert stats.duplicate_rate == 0.75

    def test_empty_input(self) -> None:
        candidates, stats = dedupe_search_results([], {})
        assert candidates == [] and stats.duplicate_rate == 0.0


class TestPostFetchDeduplication:
    def test_syndicated_identical_bodies_are_marked(self) -> None:
        body = "the same syndicated article body repeated across two domains"
        a = source("S1", text=body, domain="first.com")
        b = source("S2", text=body, domain="second.com")
        unique, dupes = dedupe_by_content([a, b])
        assert dupes == 1
        assert unique[0].duplicate_of is None
        # Retained rather than deleted, so an existing citation still resolves.
        assert unique[1].duplicate_of == "S1"

    def test_different_bodies_kept(self) -> None:
        unique, dupes = dedupe_by_content(
            [source("S1", text="alpha content"), source("S2", text="beta content")]
        )
        assert dupes == 0
        assert all(s.duplicate_of is None for s in unique)


class TestQuality:
    @pytest.mark.parametrize(
        ("url", "domain", "expected"),
        [
            ("https://arxiv.org/abs/1", "arxiv.org", SourceType.ACADEMIC),
            ("https://nist.gov/x", "nist.gov", SourceType.STANDARDS_BODY),
            ("https://mit.edu/x", "mit.edu", SourceType.ACADEMIC),
            ("https://docs.acme.io/api", "docs.acme.io", SourceType.OFFICIAL_DOCS),
            # A /docs/ path is the publisher documenting its own product,
            # not authority over whatever subject is being researched.
            ("https://acme.io/docs/setup", "acme.io", SourceType.VENDOR),
            ("https://medium.com/@a/b", "medium.com", SourceType.BLOG),
            ("https://reddit.com/r/x", "reddit.com", SourceType.FORUM),
            ("https://reuters.com/a", "reuters.com", SourceType.NEWS),
        ],
    )
    def test_classification(self, url: str, domain: str, expected: SourceType) -> None:
        assert classify_source(url, domain) == expected

    def test_primary_sources_outrank_commentary(self) -> None:
        args = {"search_score": 0.7, "word_count": 800}
        standards, _ = score_source(
            url="u", domain="nist.gov", source_type=SourceType.STANDARDS_BODY, **args
        )
        forum, _ = score_source(url="u", domain="reddit.com", source_type=SourceType.FORUM, **args)
        assert standards > forum

    def test_thin_content_penalised(self) -> None:
        thick, _ = score_source(
            url="u", domain="d", source_type=SourceType.BLOG, search_score=0.5, word_count=900
        )
        thin, _ = score_source(
            url="u", domain="d", source_type=SourceType.BLOG, search_score=0.5, word_count=40
        )
        assert thin < thick

    def test_recency_only_applies_when_question_is_time_sensitive(self) -> None:
        old = datetime.now(UTC) - timedelta(days=1500)
        without, _ = score_source(
            url="u",
            domain="d",
            source_type=SourceType.NEWS,
            search_score=0.5,
            word_count=800,
            published=old,
        )
        with_horizon, reasons = score_source(
            url="u",
            domain="d",
            source_type=SourceType.NEWS,
            search_score=0.5,
            word_count=800,
            published=old,
            recency_horizon_months=12,
        )
        assert with_horizon < without
        assert any("older" in r for r in reasons)

    def test_score_is_explainable(self) -> None:
        _, reasons = score_source(
            url="u", domain="d", source_type=SourceType.ACADEMIC, search_score=0.9, word_count=1000
        )
        assert len(reasons) >= 2 and all(isinstance(r, str) for r in reasons)

    def test_domain_concentration(self) -> None:
        assert domain_concentration(["a.com"] * 3 + ["b.com"]) == 0.75
        assert domain_concentration([]) == 0.0


class TestQuoteVerification:
    """The check that stops a real URL being cited for a sentence it never contained."""

    SRC = (
        "Fraud detection datasets are severely imbalanced, with positive cases often "
        "well under one percent of all recorded transactions in production systems."
    )

    def test_exact_quote_accepted(self) -> None:
        assert verify_quote("severely imbalanced, with positive cases often well under", self.SRC)

    def test_whitespace_is_tolerated(self) -> None:
        assert verify_quote("severely    imbalanced,\nwith positive cases often well", self.SRC)

    def test_smart_punctuation_in_the_source_is_normalised(self) -> None:
        source = "The authors call this the “imbalance problem” and note it harms recall."
        assert verify_quote('call this the "imbalance problem" and note it harms recall', source)

    def test_a_reworded_quote_is_fuzzy_not_verbatim(self) -> None:
        """A 0.88 similarity match was previously reported as verbatim."""
        reworded = "severely imbalanced, with positive examples often well under"
        match, _ = classify_quote(reworded, self.SRC)
        assert match is QuoteMatch.FUZZY
        assert not verify_quote(reworded, self.SRC)

    def test_exact_match_reports_its_offset(self) -> None:
        match, offset = classify_quote("severely imbalanced, with positive cases", self.SRC)
        assert match is QuoteMatch.EXACT_NORMALIZED
        assert offset is not None and self.SRC[offset:].startswith("severely imbalanced")

    def test_paraphrase_rejected(self) -> None:
        assert not verify_quote("these datasets tend to be rather unbalanced in practice", self.SRC)

    def test_fabricated_quote_rejected(self) -> None:
        assert not verify_quote(
            "the model reached 99.7 percent precision on the benchmark", self.SRC
        )

    def test_trivially_short_quote_rejected(self) -> None:
        assert not verify_quote("fraud", self.SRC)

    def test_empty_inputs(self) -> None:
        assert not verify_quote("", self.SRC)
        assert not verify_quote("something", "")


class TestEvidenceStore:
    def test_coverage_requires_two_independent_sources(self) -> None:
        sq = SubQuestion(id="SQ1", text="q", rationale="r")
        one = EvidenceStore([source("S1")], [evidence("S1-e1", "S1", "SQ1")])
        assert one.coverage_for(sq).verdict == "weak"

        two = EvidenceStore(
            [source("S1"), source("S2", domain="b.com")],
            [evidence("S1-e1", "S1", "SQ1"), evidence("S2-e1", "S2", "SQ1")],
        )
        assert two.coverage_for(sq).verdict == "covered"
        assert two.coverage_for(sq).distinct_sources == 2

    def test_unverified_quotes_do_not_count_toward_coverage(self) -> None:
        sq = SubQuestion(id="SQ1", text="q", rationale="r")
        store = EvidenceStore(
            [source("S1"), source("S2")],
            [
                evidence("S1-e1", "S1", "SQ1", verified=False),
                evidence("S2-e1", "S2", "SQ1", verified=False),
            ],
        )
        assert store.coverage_for(sq).verdict == "uncovered"

    def test_no_evidence_is_uncovered(self) -> None:
        sq = SubQuestion(id="SQ9", text="q", rationale="r")
        assert EvidenceStore([], []).coverage_for(sq).verdict == "uncovered"

    def test_contradiction_is_flagged(self) -> None:
        sq = SubQuestion(id="SQ1", text="q", rationale="r")
        store = EvidenceStore(
            [source("S1"), source("S2")],
            [
                evidence("S1-e1", "S1", "SQ1", stance=Stance.SUPPORTS),
                evidence("S2-e1", "S2", "SQ1", stance=Stance.CONTRADICTS),
            ],
        )
        assert store.coverage_for(sq).has_contradiction

    def test_package_groups_by_question_and_lists_sources(self) -> None:
        sqs = [
            SubQuestion(id="SQ1", text="first", rationale="r"),
            SubQuestion(id="SQ2", text="second", rationale="r"),
        ]
        store = EvidenceStore(
            [source("S1"), source("S2", domain="b.com")],
            [evidence("S1-e1", "S1", "SQ1"), evidence("S2-e1", "S2", "SQ2")],
        )
        package = store.build_package(sqs)
        assert "SQ1: first" in package.text and "SQ2: second" in package.text
        # Evidence ids, not source ids: the synthesiser references evidence
        # and the engine resolves the source.
        assert "S1-e1" in package.text and "S2-e1" in package.text
        assert "### Sources" in package.text
        assert package.evidence_count == 2
        assert set(package.source_ids) == {"S1", "S2"}
        assert set(package.evidence_ids) == {"S1-e1", "S2-e1"}

    def test_package_drops_low_confidence_items(self) -> None:
        sqs = [SubQuestion(id="SQ1", text="q", rationale="r")]
        store = EvidenceStore(
            [source("S1")],
            [evidence("S1-e1", "S1", "SQ1", verified=False, relevance=0.2)],
        )
        package = store.build_package(sqs, min_confidence=0.25)
        assert package.evidence_count == 0
        assert package.dropped_low_confidence == 1

    def test_unverified_evidence_never_reaches_synthesis(self) -> None:
        """Isolates the citability gate from the confidence floor.

        A fuzzy quote at relevance 0.9 scores 0.36, comfortably above the 0.25
        floor, so under the old design it reached synthesis and could ground a
        citation. It must now be excluded for being uncitable, while staying
        visible to diagnostic callers.
        """
        sqs = [SubQuestion(id="SQ1", text="q", rationale="r")]
        item = evidence("S1-e1", "S1", "SQ1", relevance=0.9).model_copy(
            update={"quote_match": QuoteMatch.FUZZY}
        )
        assert item.confidence > 0.25, "precondition: clears the confidence floor"

        store = EvidenceStore([source("S1")], [item])
        assert store.build_package(sqs).evidence_count == 0
        assert store.build_package(sqs, citable_only=False).evidence_count == 1

    def test_fuzzy_evidence_is_not_citable(self) -> None:
        sqs = [SubQuestion(id="SQ1", text="q", rationale="r")]
        item = evidence("S1-e1", "S1", "SQ1").model_copy(update={"quote_match": QuoteMatch.FUZZY})
        store = EvidenceStore([source("S1")], [item])
        assert store.build_package(sqs).evidence_count == 0
        assert store.citable_evidence() == []

    def test_contradictions_survive_the_per_question_cap(self) -> None:
        """A cap that silently drops disagreement would defeat the purpose of
        tracking stance at all."""
        sqs = [SubQuestion(id="SQ1", text="q", rationale="r")]
        items = [evidence(f"S1-e{i}", "S1", "SQ1", relevance=0.9) for i in range(8)]
        items.append(evidence("S2-e1", "S2", "SQ1", stance=Stance.CONTRADICTS, relevance=0.3))
        store = EvidenceStore([source("S1"), source("S2")], items)
        package = store.build_package(sqs, max_items_per_question=3)
        assert "contradicts" in package.text


class TestSourceAuthorityNeedsOwnership:
    """A /docs/ path is not evidence of authority over the subject.

    Any URL containing /docs/ used to classify as OFFICIAL_DOCS, so a
    vendor page about a standard outranked the standard itself. Authority
    now requires the publisher to be the first party: a docs.* subdomain
    is its own documentation, a /docs/ path on an arbitrary host is not.
    """

    def test_a_standards_body_publication_stays_authoritative(self) -> None:
        from agentic_research.evidence.quality import classify_source
        from agentic_research.models import SourceType

        assert (
            classify_source(
                "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf", "nvlpubs.nist.gov"
            )
            is SourceType.STANDARDS_BODY
        )

    def test_a_vendor_docs_page_about_a_standard_is_not_official(self) -> None:
        from agentic_research.evidence.quality import classify_source
        from agentic_research.models import SourceType

        result = classify_source(
            "https://www.promptfoo.dev/docs/red-team/nist-ai-rmf/", "promptfoo.dev"
        )
        assert result is not SourceType.OFFICIAL_DOCS
        assert result is SourceType.VENDOR

    def test_an_unrelated_docs_path_is_not_first_party_authority(self) -> None:
        from agentic_research.evidence.quality import classify_source
        from agentic_research.models import SourceType

        assert (
            classify_source("https://example.com/docs/whatever", "example.com")
            is not SourceType.OFFICIAL_DOCS
        )

    def test_a_docs_subdomain_is_still_first_party(self) -> None:
        """docs.stripe.com genuinely is Stripe's documentation."""
        from agentic_research.evidence.quality import classify_source
        from agentic_research.models import SourceType

        assert (
            classify_source("https://docs.stripe.com/api/charges", "docs.stripe.com")
            is SourceType.OFFICIAL_DOCS
        )

    def test_authority_ordering_puts_the_standard_above_commentary(self) -> None:
        """The ranking, not just the label: a standards body must score
        above a vendor page discussing it."""
        from agentic_research.evidence.quality import score_source
        from agentic_research.models import SourceType

        standard, _ = score_source(
            url="https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
            domain="nvlpubs.nist.gov",
            source_type=SourceType.STANDARDS_BODY,
            search_score=0.5,
            word_count=4000,
            published=None,
            recency_horizon_months=None,
        )
        vendor, _ = score_source(
            url="https://www.promptfoo.dev/docs/red-team/nist-ai-rmf/",
            domain="promptfoo.dev",
            source_type=SourceType.VENDOR,
            search_score=0.5,
            word_count=4000,
            published=None,
            recency_horizon_months=None,
        )
        assert standard > vendor
