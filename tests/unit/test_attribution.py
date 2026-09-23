"""The cross-attribution experiment.

78% of evidence in the measured run is cross-attributed. The proposed fix --
show each source only the sub-questions it was retrieved for -- drives that
to zero *by construction*, so the experiment is not testing whether it works.
It is measuring what it costs.

These tests pin the parts that must be right for that measurement to mean
anything: that every arm reads the same sources, that only the sub-question
set differs, that the harness refuses a corpus or a provider that would
produce a misleading table, and that the one structural invariant --
``retrieved_only`` cannot produce a cross-attributed item -- is checked rather
than assumed.

Nothing here contacts a model or the network.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentic_research.config import ModelRole, ModelSpec, Provider, Settings
from agentic_research.evaluation.ab import EvidenceCorpus
from agentic_research.evaluation.attribution import (
    CloudSpendRefused,
    Experiment,
    PassResult,
    Strategy,
    UnusableCorpusError,
    eligible_sources,
    invariant_breaches,
    measure,
    run_experiment,
    sub_questions_for,
    validate_corpus_for_extraction,
)
from agentic_research.models import (
    DiscoveryRef,
    EvidenceItem,
    FetchStatus,
    QuoteMatch,
    SearchQuery,
    SourceDocument,
    SubQuestion,
)
from fakes import PAGE_TEXT, FakeRouter

# SQ1 and SQ2 are deliberately near-identical in wording and SQ3/SQ4 share no
# content words with them, so the adjacency scoring has an unambiguous answer
# and a test can assert which question it picks rather than only how many.
_QUESTIONS = [
    ("SQ1", "Which resampling methods handle imbalanced fraud datasets?"),
    ("SQ2", "Which resampling methods handle imbalanced transaction datasets?"),
    ("SQ3", "How should detection models be evaluated?"),
    ("SQ4", "What are the deployment costs of streaming infrastructure?"),
]

# Source -> the (query, sub-question) paths that retrieved it. S4 has none,
# which is what makes it ineligible for every arm.
_DISCOVERY = {
    "S1": [("Q1", "SQ1")],
    "S2": [("Q2", "SQ2"), ("Q3", "SQ3")],
    "S3": [("Q4", "SQ4")],
    "S4": [],
}


def _source(source_id: str, *, text: str = PAGE_TEXT) -> SourceDocument:
    return SourceDocument(
        id=source_id,
        url=f"https://example.com/{source_id}",
        canonical_url=f"https://example.com/{source_id}",
        title=f"Page {source_id}",
        domain="example.com",
        text=text,
        fetch_status=FetchStatus.OK,
        discovered_by=[
            DiscoveryRef(query_id=q, sub_question_id=sq) for q, sq in _DISCOVERY[source_id]
        ],
    )


@pytest.fixture
def corpus() -> EvidenceCorpus:
    return EvidenceCorpus(
        question="Compare approaches for fraud detection on imbalanced data",
        sub_questions=[SubQuestion(id=qid, text=text, rationale="r") for qid, text in _QUESTIONS],
        sources=[_source(sid) for sid in _DISCOVERY],
        evidence=[],
        completed_queries=[
            SearchQuery(id=q, sub_question_id=sq, text=f"query {q}", round_number=1)
            for refs in _DISCOVERY.values()
            for q, sq in refs
        ],
    )


@pytest.fixture(autouse=True)
def _fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the model boundary, leaving the graph and the worker real."""
    import agentic_research.evaluation.attribution as attribution

    monkeypatch.setattr(attribution, "ModelRouter", lambda settings, tracker=None: FakeRouter())


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


class TestStrategySelection:
    def test_all_open_shows_every_sub_question(self, corpus: EvidenceCorpus) -> None:
        shown = sub_questions_for(Strategy.ALL_OPEN, _source("S1"), corpus.sub_questions)
        assert [q.id for q in shown] == ["SQ1", "SQ2", "SQ3", "SQ4"]

    def test_retrieved_only_shows_the_discovered_sub_questions(
        self, corpus: EvidenceCorpus
    ) -> None:
        shown = sub_questions_for(Strategy.RETRIEVED_ONLY, _source("S2"), corpus.sub_questions)
        assert [q.id for q in shown] == ["SQ2", "SQ3"]

    def test_adjacent_adds_the_nearest_question_not_an_arbitrary_one(
        self, corpus: EvidenceCorpus
    ) -> None:
        shown = sub_questions_for(
            Strategy.ADJACENT, _source("S1"), corpus.sub_questions, adjacent_k=2
        )
        # SQ2 shares almost every content word with SQ1; SQ3 and SQ4 share
        # none, so k=2 adds one question rather than filling its quota.
        assert [q.id for q in shown] == ["SQ1", "SQ2"]

    def test_adjacent_k_is_a_ceiling_not_a_quota(self, corpus: EvidenceCorpus) -> None:
        """Padding the set with unrelated questions to reach k would rebuild
        the problem the strategy exists to reduce."""
        shown = sub_questions_for(
            Strategy.ADJACENT, _source("S1"), corpus.sub_questions, adjacent_k=3
        )
        assert len(shown) == 2

    def test_adjacent_with_k_zero_is_retrieved_only(self, corpus: EvidenceCorpus) -> None:
        source = _source("S1")
        assert sub_questions_for(
            Strategy.ADJACENT, source, corpus.sub_questions, adjacent_k=0
        ) == sub_questions_for(Strategy.RETRIEVED_ONLY, source, corpus.sub_questions)

    def test_corpus_order_is_preserved_in_every_strategy(self, corpus: EvidenceCorpus) -> None:
        """The worker attributes an out-of-range id to the *first* entry, so a
        reordered list would introduce a difference the strategy did not
        cause."""
        for strategy in Strategy:
            shown = sub_questions_for(strategy, _source("S2"), corpus.sub_questions)
            ids = [q.id for q in shown]
            assert ids == sorted(ids), f"{strategy.value} reordered the sub-questions"

    def test_selection_is_deterministic(self, corpus: EvidenceCorpus) -> None:
        source = _source("S1")
        first = sub_questions_for(Strategy.ADJACENT, source, corpus.sub_questions)
        second = sub_questions_for(Strategy.ADJACENT, source, corpus.sub_questions)
        assert [q.id for q in first] == [q.id for q in second]

    def test_every_strategy_has_a_letter_and_a_description(self) -> None:
        assert {s.letter for s in Strategy} == {"A", "B", "C"}
        assert all(s.description for s in Strategy)


# ---------------------------------------------------------------------------
# Eligibility and corpus validation
# ---------------------------------------------------------------------------


class TestEligibility:
    def test_a_source_with_no_discovery_path_is_excluded_from_every_arm(
        self, corpus: EvidenceCorpus
    ) -> None:
        """Letting B fall back to A for that one source would make the arms
        incomparable, which is worse than dropping it from all of them."""
        assert [s.id for s in eligible_sources(corpus)] == ["S1", "S2", "S3"]

    def test_unusable_and_duplicate_sources_are_excluded(self, corpus: EvidenceCorpus) -> None:
        empty = _source("S1", text="")
        duplicate = _source("S2").model_copy(update={"duplicate_of": "S1"})
        corpus.sources = [empty, duplicate, _source("S3")]
        assert [s.id for s in eligible_sources(corpus)] == ["S3"]


class TestCorpusValidation:
    def test_a_healthy_corpus_passes(self, corpus: EvidenceCorpus) -> None:
        assert validate_corpus_for_extraction(corpus) == []

    def test_stripped_source_text_is_refused(self, corpus: EvidenceCorpus) -> None:
        corpus.sources = [_source(sid, text="") for sid in _DISCOVERY]
        problems = validate_corpus_for_extraction(corpus)
        assert any("no text" in p for p in problems)
        assert any("freeze" in p for p in problems), "the fix should be named"

    def test_a_single_sub_question_is_refused(self, corpus: EvidenceCorpus) -> None:
        corpus.sub_questions = corpus.sub_questions[:1]
        assert any("nothing to measure" in p for p in validate_corpus_for_extraction(corpus))

    def test_a_corpus_where_b_equals_a_is_refused(self, corpus: EvidenceCorpus) -> None:
        """Every source retrieved for every sub-question makes narrowing a
        no-op; the resulting table would show three identical arms and look
        like a finding."""
        everything = [
            DiscoveryRef(query_id=f"Q{i}", sub_question_id=qid)
            for i, (qid, _) in enumerate(_QUESTIONS, start=1)
        ]
        corpus.sources = [_source("S1").model_copy(update={"discovered_by": everything})]
        assert any("identical to strategy A" in p for p in validate_corpus_for_extraction(corpus))


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    async def test_a_cloud_researcher_is_refused_without_explicit_consent(
        self, corpus: EvidenceCorpus, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agentic_research.evaluation.attribution as attribution

        class _CloudRouter(FakeRouter):
            def assignments(self) -> dict[ModelRole, ModelSpec]:
                return dict.fromkeys(
                    ModelRole, ModelSpec(provider=Provider.OPENAI, model="gpt-6-luna")
                )

        monkeypatch.setattr(
            attribution, "ModelRouter", lambda settings, tracker=None: _CloudRouter()
        )
        with pytest.raises(CloudSpendRefused) as caught:
            await run_experiment(corpus, settings, repeats=2)
        # The refusal has to state the size of the bill it just prevented,
        # or it is only an obstacle.
        assert "18" in str(caught.value), "should name 3 sources x 3 strategies x 2 repeats"

    async def test_cloud_runs_when_the_spend_is_accepted(
        self, corpus: EvidenceCorpus, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agentic_research.evaluation.attribution as attribution

        class _CloudRouter(FakeRouter):
            def assignments(self) -> dict[ModelRole, ModelSpec]:
                return dict.fromkeys(
                    ModelRole, ModelSpec(provider=Provider.OPENAI, model="gpt-6-luna")
                )

        monkeypatch.setattr(
            attribution, "ModelRouter", lambda settings, tracker=None: _CloudRouter()
        )
        experiment = await run_experiment(
            corpus, settings, strategies=[Strategy.ALL_OPEN], allow_cloud=True
        )
        assert experiment.passes and experiment.passes[0].ok

    async def test_a_degraded_corpus_is_refused_before_any_model_work(
        self, corpus: EvidenceCorpus, settings: Settings
    ) -> None:
        corpus.sources = [_source(sid, text="") for sid in _DISCOVERY]
        with pytest.raises(UnusableCorpusError):
            await run_experiment(corpus, settings)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _item(item_id: str, source: str, question: str, *, citable: bool, cross: bool) -> EvidenceItem:
    return EvidenceItem(
        id=item_id,
        source_id=source,
        sub_question_id=question,
        claim="a finding",
        quote="a span",
        quote_match=QuoteMatch.EXACT_NORMALIZED if citable else QuoteMatch.FUZZY,
        cross_attributed=cross,
    )


class TestMeasurement:
    def test_cross_attribution_is_reported_over_all_items_and_over_citable_ones(
        self, corpus: EvidenceCorpus
    ) -> None:
        """The published 78% is over all evidence, but only citable evidence
        can ground a citation, so the two are not interchangeable."""
        result = PassResult(
            strategy=Strategy.ALL_OPEN,
            repeat=1,
            ok=True,
            evidence=[
                _item("S1-e1", "S1", "SQ1", citable=True, cross=False),
                _item("S1-e2", "S1", "SQ2", citable=True, cross=True),
                _item("S2-e1", "S2", "SQ3", citable=False, cross=True),
                _item("S2-e2", "S2", "SQ4", citable=False, cross=True),
            ],
            shown_per_source={"S1": 4, "S2": 4, "S3": 4},
        )
        row = measure(result, corpus)
        assert row["evidence_items"] == 4
        assert row["citable_items"] == 2
        assert row["cross_attribution_rate"] == 0.75
        assert row["citable_cross_attribution_rate"] == 0.5

    def test_an_empty_pass_reports_none_rather_than_zero(self, corpus: EvidenceCorpus) -> None:
        """A rate of 0.0 over no evidence reads as a result. It is not one."""
        row = measure(
            PassResult(strategy=Strategy.RETRIEVED_ONLY, repeat=1, ok=True),
            corpus,
        )
        assert row["evidence_items"] == 0
        assert row["quote_fidelity"] is None
        assert row["cross_attribution_rate"] is None
        assert row["citable_cross_attribution_rate"] is None

    def test_source_utilisation_counts_only_sources_that_earned_a_citable_item(
        self, corpus: EvidenceCorpus
    ) -> None:
        result = PassResult(
            strategy=Strategy.ALL_OPEN,
            repeat=1,
            ok=True,
            evidence=[
                _item("S1-e1", "S1", "SQ1", citable=True, cross=False),
                _item("S2-e1", "S2", "SQ2", citable=False, cross=False),
            ],
            shown_per_source={"S1": 4, "S2": 4, "S3": 4},
        )
        # One of three eligible sources produced something citable.
        assert measure(result, corpus)["source_utilisation"] == pytest.approx(1 / 3, abs=1e-4)

    def test_mean_sub_questions_shown_records_the_mechanism_being_varied(
        self, corpus: EvidenceCorpus
    ) -> None:
        result = PassResult(
            strategy=Strategy.RETRIEVED_ONLY,
            repeat=1,
            ok=True,
            shown_per_source={"S1": 1, "S2": 2, "S3": 1},
        )
        row = measure(result, corpus)
        assert row["mean_sub_questions_shown"] == pytest.approx(4 / 3, abs=0.01)


class TestInvariants:
    def test_a_cross_attributed_item_under_retrieved_only_is_flagged(self) -> None:
        """Impossible by construction, so if it happens the harness is
        measuring something other than the strategy."""
        result = PassResult(
            strategy=Strategy.RETRIEVED_ONLY,
            repeat=1,
            ok=True,
            evidence=[_item("S1-e1", "S1", "SQ2", citable=True, cross=True)],
        )
        breaches = invariant_breaches(result)
        assert breaches and "impossible by construction" in breaches[0]

    def test_all_open_is_allowed_to_cross_attribute(self) -> None:
        result = PassResult(
            strategy=Strategy.ALL_OPEN,
            repeat=1,
            ok=True,
            evidence=[_item("S1-e1", "S1", "SQ2", citable=True, cross=True)],
        )
        assert invariant_breaches(result) == []


# ---------------------------------------------------------------------------
# Aggregation across repeats
# ---------------------------------------------------------------------------


def _experiment(rows: list[dict[str, Any]]) -> Experiment:
    return Experiment(
        question="q",
        corpus_summary="",
        environment={},
        strategies=[Strategy.ALL_OPEN],
        repeats=len(rows),
        adjacent_k=2,
        eligible_sources=3,
        measurements={Strategy.ALL_OPEN.value: rows},
    )


class TestAggregation:
    def test_reports_spread_not_just_a_mean(self) -> None:
        summary = _experiment(
            [
                {"repeat": 1, "citable_items": 10},
                {"repeat": 2, "citable_items": 20},
                {"repeat": 3, "citable_items": 30},
            ]
        ).aggregate(Strategy.ALL_OPEN)
        assert summary["repeats"] == 3
        assert summary["citable_items"] == {"mean": 20.0, "min": 10, "max": 30, "n": 3}

    def test_n_records_how_many_repeats_actually_produced_a_number(self) -> None:
        """A metric is None when its denominator was empty. A mean over two of
        three repeats is a different claim from a mean over three."""
        summary = _experiment(
            [
                {"repeat": 1, "quote_fidelity": None},
                {"repeat": 2, "quote_fidelity": 0.8},
                {"repeat": 3, "quote_fidelity": 0.6},
            ]
        ).aggregate(Strategy.ALL_OPEN)
        assert summary["quote_fidelity"]["n"] == 2
        assert summary["quote_fidelity"]["mean"] == pytest.approx(0.7)

    def test_a_failed_pass_does_not_poison_the_aggregate(self) -> None:
        summary = _experiment(
            [
                {"repeat": 1, "failed": True, "error": "boom"},
                {"repeat": 2, "citable_items": 12},
            ]
        ).aggregate(Strategy.ALL_OPEN)
        assert summary["repeats"] == 1
        assert summary["citable_items"]["mean"] == 12

    def test_all_passes_failing_is_stated_rather_than_shown_as_empty(self) -> None:
        summary = _experiment([{"repeat": 1, "failed": True, "error": "boom"}]).aggregate(
            Strategy.ALL_OPEN
        )
        assert summary["all_passes_failed"] is True


# ---------------------------------------------------------------------------
# End to end, through the real extract_worker
# ---------------------------------------------------------------------------


class TestExperimentEndToEnd:
    async def test_every_arm_reads_the_same_sources(
        self, corpus: EvidenceCorpus, settings: Settings
    ) -> None:
        experiment = await run_experiment(corpus, settings)

        assert experiment.eligible_sources == 3
        assert experiment.excluded_sources == ["S4"]
        assert all(p.ok for p in experiment.passes), [p.error for p in experiment.passes]
        for result in experiment.passes:
            assert set(result.shown_per_source) == {"S1", "S2", "S3"}
            # One extraction call per source, in every arm. If an arm made a
            # different number of calls it is not the same experiment.
            assert result.llm_calls == 3

    async def test_only_the_sub_question_set_differs_between_arms(
        self, corpus: EvidenceCorpus, settings: Settings
    ) -> None:
        experiment = await run_experiment(corpus, settings)
        by_strategy = {p.strategy: p for p in experiment.passes}

        assert by_strategy[Strategy.ALL_OPEN].shown_per_source == {"S1": 4, "S2": 4, "S3": 4}
        assert by_strategy[Strategy.RETRIEVED_ONLY].shown_per_source == {"S1": 1, "S2": 2, "S3": 1}
        adjacent = by_strategy[Strategy.ADJACENT].shown_per_source
        for source_id, count in by_strategy[Strategy.RETRIEVED_ONLY].shown_per_source.items():
            assert count <= adjacent[source_id] <= 4

    async def test_retrieved_only_removes_cross_attribution_entirely(
        self, corpus: EvidenceCorpus, settings: Settings
    ) -> None:
        arms = [Strategy.ALL_OPEN, Strategy.RETRIEVED_ONLY]
        experiment = await run_experiment(corpus, settings, strategies=arms)
        narrowed = experiment.aggregate(Strategy.RETRIEVED_ONLY)
        everything = experiment.aggregate(Strategy.ALL_OPEN)

        assert experiment.breaches == []
        assert narrowed["cross_attribution_rate"]["mean"] == 0.0
        # Showing a source a question nothing retrieved it for is the entire
        # mechanism; if A did not cross-attribute there would be nothing to
        # trade away and the experiment would be pointless.
        assert everything["cross_attribution_rate"]["mean"] > 0.0

    async def test_every_reported_rate_is_a_rate(
        self, corpus: EvidenceCorpus, settings: Settings
    ) -> None:
        experiment = await run_experiment(corpus, settings)
        rates = (
            "quote_fidelity",
            "quote_drift",
            "cross_attribution_rate",
            "citable_cross_attribution_rate",
            "evidence_coverage",
            "sub_questions_with_citable_evidence",
            "source_utilisation",
        )
        for rows in experiment.measurements.values():
            for row in rows:
                for name in rates:
                    value = row[name]
                    assert value is None or 0.0 <= value <= 1.0, f"{name} = {value}"
                assert row["citable_items"] <= row["evidence_items"]

    async def test_repeats_are_recorded_individually_as_well_as_aggregated(
        self, corpus: EvidenceCorpus, settings: Settings
    ) -> None:
        """One pass is one draw from a sampling model. Publishing only the
        mean would hide that."""
        experiment = await run_experiment(
            corpus, settings, strategies=[Strategy.ALL_OPEN], repeats=2
        )
        rows = experiment.measurements[Strategy.ALL_OPEN.value]
        assert [r["repeat"] for r in rows] == [1, 2]
        assert experiment.aggregate(Strategy.ALL_OPEN)["repeats"] == 2

    async def test_the_report_states_what_the_numbers_do_and_do_not_show(
        self, corpus: EvidenceCorpus, settings: Settings
    ) -> None:
        payload = (await run_experiment(corpus, settings)).to_dict()
        assert payload["experiment"] == "cross_attribution"
        assert "by construction" in payload["note"]
        assert payload["design"]["excluded_sources"] == ["S4"]
        assert set(payload["design"]["strategies"]) == {s.value for s in Strategy}
        assert payload["environment"], "a measurement without its environment is not reproducible"
        assert "invariant_breaches" in payload

    async def test_production_extraction_is_unchanged(self) -> None:
        """The experiment exists to inform a decision, not to make it. Until
        there is a measurement, every source still sees every open
        sub-question."""
        from agentic_research.graph.routing import dispatch_extraction

        questions = [SubQuestion(id=qid, text=text, rationale="r") for qid, text in _QUESTIONS]
        state = {
            "pending_extraction": ["S1"],
            "sources": [_source("S1")],
            "sub_questions": questions,
        }
        sends = dispatch_extraction(state)  # type: ignore[arg-type]
        assert isinstance(sends, list) and len(sends) == 1
        assert [q.id for q in sends[0].arg["sub_questions"]] == ["SQ1", "SQ2", "SQ3", "SQ4"]
