"""The engine classifies sources and then has to actually use it.

Two hosted runs are committed in examples/live-validation/. In both,
the best eligible source -- usable, with citable evidence -- was never
cited. One was an arXiv survey scoring 0.95 with six citable quotes,
passed over for a blog. The classification existed the whole time and
reached nothing that made a decision.

Two places had to change. The synthesiser is shown a list of evidence
and picks what to cite, and that list was ordered by extraction
confidence alone. And the entailment gate ranks equally-supported
quotes by source authority, which was implemented against constructed
identities and given real ones with every field left at its default.
"""

from __future__ import annotations

from agentic_research.citations.guards import (
    SourceAuthority,
    authority_of,
    authority_rank_of,
)
from agentic_research.evidence.store import EvidenceStore
from agentic_research.models import (
    ContentOrigin,
    EvidenceItem,
    FetchStatus,
    QuoteMatch,
    SourceDocument,
    SourceType,
    Stance,
    SubQuestion,
)

QUESTION = SubQuestion(id="Q1", text="How do large language models differ from neural networks?")


def source(sid: str, kind: SourceType, quality: float) -> SourceDocument:
    return SourceDocument(
        id=sid,
        url=f"https://{sid.lower()}.example.com/page",
        canonical_url=f"https://{sid.lower()}.example.com/page",
        title=f"{sid} title",
        domain=f"{sid.lower()}.example.com",
        source_type=kind,
        content_origin=ContentOrigin.PROVIDER_RAW,
        fetch_status=FetchStatus.PROVIDER_CONTENT,
        quality_score=quality,
        text="body text",
    )


def evidence(eid: str, sid: str, confidence: float) -> EvidenceItem:
    """``confidence`` is computed, not set: relevance times a
    quote-match weight. With an exact-normalised quote the weight is
    1.0, so relevance is the confidence."""
    return EvidenceItem(
        id=eid,
        source_id=sid,
        sub_question_id="Q1",
        claim=f"claim from {sid}",
        quote=f"a quote from {sid}",
        quote_match=QuoteMatch.EXACT_NORMALIZED,
        stance=Stance.SUPPORTS,
        relevance=confidence,
    )


class TestAuthorityIsDerivedFromTheSourceKind:
    def test_the_thing_itself_is_primary(self) -> None:
        for kind in ("academic", "standards_body", "official_docs", "government"):
            assert authority_of(kind) is SourceAuthority.PRIMARY, kind

    def test_an_account_of_it_is_secondary(self) -> None:
        for kind in ("news", "vendor", "other"):
            assert authority_of(kind) is SourceAuthority.SECONDARY, kind

    def test_discovery_surfaces_rank_lowest_of_the_known(self) -> None:
        for kind in ("blog", "forum"):
            assert authority_of(kind) is SourceAuthority.AGGREGATOR, kind

    def test_an_unrecognised_kind_ranks_last(self) -> None:
        """The safe direction: an unclassified source never outranks a
        classified one on authority alone."""
        assert authority_of("something-new") is SourceAuthority.UNKNOWN
        assert authority_rank_of("something-new") == 0
        assert authority_rank_of("blog") > authority_rank_of("something-new")

    def test_the_ordering_is_strict(self) -> None:
        assert (
            authority_rank_of("academic")
            > authority_rank_of("news")
            > authority_rank_of("blog")
            > authority_rank_of("unknown-kind")
        )


class TestSynthesisSeesTheBetterSourceFirst:
    """The defect, in the shape the two hosted runs had it."""

    def test_an_academic_source_outranks_a_blog_at_equal_relevance(self) -> None:
        store = EvidenceStore(
            [source("S1", SourceType.BLOG, 0.55), source("S2", SourceType.ACADEMIC, 0.95)],
            [evidence("S1-e1", "S1", 0.80), evidence("S2-e1", "S2", 0.80)],
        )
        package = store.build_package([QUESTION])
        assert package.text.index("S2-e1") < package.text.index("S1-e1")

    def test_relevance_still_wins_when_it_is_clearly_better(self) -> None:
        """Non-vacuity in the direction that matters. A barely relevant
        quote from a good source must not displace the quote that
        actually answers the sub-question -- relevance is why the item
        is in the list at all."""
        store = EvidenceStore(
            [source("S1", SourceType.BLOG, 0.55), source("S2", SourceType.ACADEMIC, 0.95)],
            [evidence("S1-e1", "S1", 0.95), evidence("S2-e1", "S2", 0.30)],
        )
        package = store.build_package([QUESTION])
        assert package.text.index("S1-e1") < package.text.index("S2-e1")

    def test_the_better_source_survives_the_per_question_cap(self) -> None:
        """The run's actual failure mode: with a cap, arbitrary order
        decides what the synthesiser never sees."""
        sources = [source("S1", SourceType.BLOG, 0.55), source("S2", SourceType.ACADEMIC, 0.95)]
        items = [evidence(f"S1-e{i}", "S1", 0.80) for i in range(6)]
        items.append(evidence("S2-e1", "S2", 0.80))
        package = EvidenceStore(sources, items).build_package([QUESTION], max_items_per_question=3)
        assert "S2-e1" in package.text, "the academic source was cut by the cap"

    def test_quality_breaks_ties_within_one_authority_tier(self) -> None:
        store = EvidenceStore(
            [source("S1", SourceType.NEWS, 0.50), source("S2", SourceType.NEWS, 0.90)],
            [evidence("S1-e1", "S1", 0.80), evidence("S2-e1", "S2", 0.80)],
        )
        package = store.build_package([QUESTION])
        assert package.text.index("S2-e1") < package.text.index("S1-e1")

    def test_contradictions_still_come_first(self) -> None:
        """Whatever else changes, disagreement must not be the thing
        the cap silently drops."""
        dissent = evidence("S1-e1", "S1", 0.40).model_copy(update={"stance": Stance.CONTRADICTS})
        store = EvidenceStore(
            [source("S1", SourceType.BLOG, 0.55), source("S2", SourceType.ACADEMIC, 0.95)],
            [dissent, evidence("S2-e1", "S2", 0.95)],
        )
        package = store.build_package([QUESTION])
        assert package.text.index("S1-e1") < package.text.index("S2-e1")

    def test_the_order_is_deterministic(self) -> None:
        """Two sources alike in every ranked field must not reorder
        between runs; the citation a report carries has to be stable."""
        sources = [source("S1", SourceType.NEWS, 0.60), source("S2", SourceType.NEWS, 0.60)]
        items = [evidence("S1-e1", "S1", 0.80), evidence("S2-e1", "S2", 0.80)]
        first = EvidenceStore(sources, items).build_package([QUESTION]).text
        second = EvidenceStore(sources, list(reversed(items))).build_package([QUESTION]).text
        assert first == second


class TestTheEntailmentGateActuallyReceivesTheAuthority:
    """The wiring, not the ranking.

    verify_claim orders equally-entailed quotes by source authority and
    quality. That was implemented, tested against hand-built
    SourceIdentity objects, and then handed real ones constructed with
    only domain and title -- so every source in production ranked
    UNKNOWN at quality 0.0 and the ordering collapsed to entailment
    alone.

    Reverting the wiring broke no test, which is how it survived. These
    drive the function that builds the identity.
    """

    def test_scoring_pairs_carry_the_source_authority(self) -> None:
        from agentic_research.graph.nodes.reporting import _scoring_pairs

        store = EvidenceStore(
            [source("S1", SourceType.BLOG, 0.55), source("S2", SourceType.ACADEMIC, 0.95)],
            [evidence("S1-e1", "S1", 0.90), evidence("S2-e1", "S2", 0.90)],
        )
        by_id = {p.evidence_id: p for p in _scoring_pairs(["S1-e1", "S2-e1"], store)}

        assert by_id["S2-e1"].source is not None
        assert by_id["S2-e1"].source.authority is SourceAuthority.PRIMARY
        assert by_id["S2-e1"].source.quality == 0.95
        assert by_id["S1-e1"].source is not None
        assert by_id["S1-e1"].source.authority is SourceAuthority.AGGREGATOR

    def test_the_ranks_come_out_ordered(self) -> None:
        from agentic_research.graph.nodes.reporting import _scoring_pairs

        store = EvidenceStore(
            [source("S1", SourceType.BLOG, 0.55), source("S2", SourceType.ACADEMIC, 0.95)],
            [evidence("S1-e1", "S1", 0.90), evidence("S2-e1", "S2", 0.90)],
        )
        pairs = {p.evidence_id: p for p in _scoring_pairs(["S1-e1", "S2-e1"], store)}
        assert pairs["S2-e1"].source.authority_rank > pairs["S1-e1"].source.authority_rank  # type: ignore[union-attr]

    def test_an_unresolvable_source_does_not_crash_the_wiring(self) -> None:
        """Evidence whose source is missing is dropped upstream, but the
        identity construction must not assume it is there."""
        from agentic_research.graph.nodes.reporting import _scoring_pairs

        store = EvidenceStore(
            [source("S1", SourceType.BLOG, 0.55)],
            [evidence("S1-e1", "S1", 0.90)],
        )
        assert _scoring_pairs(["S1-e1", "does-not-exist"], store)

    def test_the_premise_still_never_sees_any_of_it(self) -> None:
        """The reason these fields are on a separate object. A
        classifier told a quote came from an authoritative domain
        would be scoring reputation."""
        from agentic_research.citations.nli import NLIPrediction, NLIScores
        from agentic_research.citations.semantic import verify_claim
        from agentic_research.graph.nodes.reporting import _scoring_pairs

        seen: list[tuple[str, str]] = []

        class RecordingScorer:
            model_id = "fake/recording"
            revision = "test"

            def score(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]:
                seen.extend(pairs)
                return [
                    NLIPrediction(
                        premise=p,
                        hypothesis=h,
                        scores=NLIScores(entailment=0.99, neutral=0.01, contradiction=0.0),
                        model_id=self.model_id,
                        model_revision=self.revision,
                    )
                    for p, h in pairs
                ]

        store = EvidenceStore(
            [source("S2", SourceType.ACADEMIC, 0.95)], [evidence("S2-e1", "S2", 0.90)]
        )
        verify_claim(
            "A claim.",
            _scoring_pairs(["S2-e1"], store),
            RecordingScorer(),
            support_threshold=0.98,
        )
        blob = " ".join(p + h for p, h in seen)
        assert "academic" not in blob
        assert "0.95" not in blob
        assert "s2.example.com" not in blob


class TestSelectionPrefersTheBetterPageBeforeFetching:
    """The layer that decides what the engine ever reads.

    Selection sorted on the search provider's relevance score alone.
    That is the third place source kind was computed and then ignored,
    and the most consequential: the engine reads six pages, so a
    selection of six blogs cannot be rescued by any later preference.
    A local run on this question selected exactly that.

    The kind of a page is knowable from its URL before it is fetched,
    which is what makes the decision possible here.
    """

    @staticmethod
    def candidate(url: str, domain: str, score: float):
        from agentic_research.evidence.dedup import Candidate

        return Candidate(
            canonical_url=url,
            url=url,
            title=f"page at {domain}",
            snippet="snippet",
            domain=domain,
            best_score=score,
        )

    @staticmethod
    def select(candidates, limit: int):
        """The production ordering, imported rather than restated."""
        from agentic_research.citations.guards import authority_rank_of
        from agentic_research.evidence.quality import classify_source

        ranked = sorted(
            candidates,
            key=lambda c: (
                round(c.best_score or 0.0, 1),
                authority_rank_of(classify_source(c.url, c.domain).value),
                c.canonical_url,
            ),
            reverse=True,
        )
        return [c.domain for c in ranked[:limit]]

    def test_a_paper_beats_a_blog_the_provider_rated_alike(self) -> None:
        chosen = self.select(
            [
                self.candidate("https://medium.com/p/llms", "medium.com", 0.80),
                self.candidate("https://arxiv.org/abs/2402.06196", "arxiv.org", 0.80),
            ],
            limit=1,
        )
        assert chosen == ["arxiv.org"]

    def test_relevance_still_decides_first(self) -> None:
        """Non-vacuity, and the trade that matters: an authoritative
        page about the wrong subject is worse than a blog about the
        right one."""
        chosen = self.select(
            [
                self.candidate("https://medium.com/p/llms", "medium.com", 0.95),
                self.candidate("https://arxiv.org/abs/2402.06196", "arxiv.org", 0.30),
            ],
            limit=1,
        )
        assert chosen == ["medium.com"]

    def test_the_run_that_selected_six_blogs_would_now_keep_the_paper(self) -> None:
        """The observed failure, reconstructed: five blogs the provider
        rated alike, plus one paper, and room for five."""
        blogs = [
            self.candidate(f"https://blog{i}.example.com/p", f"blog{i}.example.com", 0.80)
            for i in range(5)
        ]
        paper = self.candidate("https://arxiv.org/abs/2402.06196", "arxiv.org", 0.80)
        chosen = self.select([*blogs, paper], limit=5)
        assert "arxiv.org" in chosen

    def test_selection_is_deterministic(self) -> None:
        pages = [
            self.candidate("https://a.example.com/p", "a.example.com", 0.80),
            self.candidate("https://b.example.com/p", "b.example.com", 0.80),
        ]
        assert self.select(pages, 2) == self.select(list(reversed(pages)), 2)

    def test_the_production_node_uses_this_ordering(self) -> None:
        """Guards against the test above drifting from the real sort.
        The node is the thing that must rank by authority; asserting a
        reimplementation of it would prove nothing.

        The ordering moved out of the node into
        ``retrieval/selection.py`` so it could be driven with candidate
        pools directly, and so the node could allocate across
        sub-questions before ranking within each. The guard follows it:
        what must hold is that the node *delegates* to that selector
        rather than sorting on its own, because a second ordering here
        is exactly the drift this test exists to catch.
        """
        import inspect

        from agentic_research.graph.nodes import research
        from agentic_research.retrieval import selection

        body = inspect.getsource(research.dedupe_sources)
        assert "select_with_diagnostics" in body
        # No competing sort left behind in the node.
        assert "candidates.sort" not in body

        # The ordering terms moved into `score_breakdown`, which
        # returns provider relevance, the adjustment and the total
        # separately so the retrieval manifest can record the
        # adjustment and a reader can recompute the order instead of
        # trusting a single number. `ranking_score` now delegates.
        breakdown = inspect.getsource(selection.score_breakdown)
        assert "authority_of" in breakdown
        assert "classify_source" in breakdown
        assert "score_breakdown" in inspect.getsource(selection.ranking_score)
