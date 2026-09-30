"""A comparison has to be answerable by claims that can be verified.

Nine hosted runs asked how a large language model differs from a
neural network. Between them they produced exactly one
`direct_contrast` claim, and the atomicity guard refused it -- rightly,
because a contrast asserts two things and every other guard reasons
about "the sentence that supports this claim".

So the synthesiser did the only thing left: it wrote claims about each
subject and declared `dimension`. The relevance judge then refused
each one for "describing neural networks, not how LLMs differ" --
applying the *report's* question to a single claim, which no single
atomic claim can answer. Comparisons published nothing.

Two changes, at the two levels the question actually lives at.

At the **claim** level the judge no longer vetoes a claim that
structurally fills an optional part of the answer. A prompt change
asking it to judge against the listed parts was tried first and
measurably failed -- the run after it refused three `dimension` claims
with the same reasoning -- so the authority is narrowed in code.

At the **report** level `assess_coverage` will not call a comparison
answered unless the published claims between them speak about every
subject the question named. That is what stops the original defect,
and it always was a report-level problem: five supported claims about
one of two subjects, published as an answer to how they differ.
"""

from __future__ import annotations

from agentic_research.answer_contract import QuestionType, build_contract
from agentic_research.answer_coverage import assess_coverage
from agentic_research.comparison import SideClaim
from agentic_research.graph.nodes.reporting import _discharges_a_core_slot

COMPARISON = build_contract(
    "How does a large language model differ from a neural network?",
    QuestionType.COMPARISON,
    entities=("large language model", "neural network"),
)
DEFINITION = build_contract(
    "What is a vector database?", QuestionType.DEFINITION, entities=("vector database",)
)


class TestWhoTheJudgeMayVeto:
    def test_it_may_veto_the_core_slot(self) -> None:
        assert _discharges_a_core_slot(COMPARISON, "direct_contrast")

    def test_it_may_veto_a_slot_that_stands_in_for_the_core(self) -> None:
        """`relationship` discharges `direct_contrast`, so a claim
        declaring it is answering, and the judge decides."""
        assert _discharges_a_core_slot(COMPARISON, "relationship")

    def test_it_may_not_veto_an_optional_slot(self) -> None:
        """The change. A `dimension` claim contributes to a comparison
        and cannot answer one on its own, so the judge's opinion about
        whether it answers is not the deciding one."""
        assert not _discharges_a_core_slot(COMPARISON, "dimension")

    def test_an_undeclared_slot_stays_with_the_judge(self) -> None:
        """Conservative: nothing is known about what the claim was
        for, so the verdict stands."""
        assert _discharges_a_core_slot(COMPARISON, None)
        assert _discharges_a_core_slot(COMPARISON, "")

    def test_a_slot_this_contract_never_asked_for_stays_with_the_judge(self) -> None:
        """The structural check refuses these upstream, but narrowing
        an authority must not depend on another gate having caught the
        case first."""
        assert _discharges_a_core_slot(COMPARISON, "invented_slot")

    def test_it_generalises_past_comparisons(self) -> None:
        assert _discharges_a_core_slot(DEFINITION, "definition")
        assert not _discharges_a_core_slot(DEFINITION, "distinguishing_property")


class TestTheReportLevelGateIsWhatStopsTheOriginalDefect:
    def test_one_claim_per_subject_answers_a_comparison(self) -> None:
        """Slot and text together now: a contrast is assembled from one
        verified claim per subject *within a dimension*, so which
        dimension each claim was declared against is part of the
        question being asked."""
        coverage = assess_coverage(
            COMPARISON,
            ["dimension", "dimension"],
            claims=[
                SideClaim(
                    subject="",
                    text="Large language models predict token sequences from text.",
                    answer_slot="dimension",
                ),
                SideClaim(
                    subject="",
                    text="Neural networks classify images into categories.",
                    answer_slot="dimension",
                ),
            ],
        )
        assert coverage.answered
        assert coverage.comparison_pairs

    def test_two_claims_on_different_slots_are_not_a_contrast(self) -> None:
        """Mentioning both subjects is not comparing them. One claim
        about what a thing is and another about how it relates are both
        true and address no common axis."""
        coverage = assess_coverage(
            COMPARISON,
            ["dimension", "relationship"],
            claims=[
                SideClaim(
                    subject="",
                    text="Large language models predict token sequences from text.",
                    answer_slot="dimension",
                ),
                SideClaim(
                    subject="",
                    text="Neural networks are a broader family than language models.",
                    answer_slot="relationship",
                ),
            ],
        )
        assert not coverage.comparison_pairs

    def test_five_claims_about_one_subject_still_do_not(self) -> None:
        """The v1.1.x failure verbatim: supported, cited claims about
        one of two subjects, published as an answer to how they
        differ. It must still fail, and now it fails here rather than
        at the claim gate."""
        coverage = assess_coverage(
            COMPARISON,
            ["dimension"] * 5,
            claim_texts=[
                "Large language models predict tokens.",
                "Large language models use transformer architectures.",
                "Large language models are trained on text corpora.",
                "Large language models have billions of parameters.",
                "Large language models can summarise documents.",
            ],
        )
        assert not coverage.answered
        assert "direct_contrast" in coverage.missing_core

    def test_a_real_contrast_still_answers(self) -> None:
        assert assess_coverage(
            COMPARISON, ["direct_contrast"], claim_texts=["LLMs differ from neural networks in X."]
        ).answered

    def test_publishing_nothing_does_not(self) -> None:
        assert not assess_coverage(COMPARISON, [], claim_texts=[]).answered

    def test_claims_with_no_slot_do_not_discharge_it(self) -> None:
        """Spanning the entities is not enough on its own. Something
        has to have declared a part of the answer."""
        assert not assess_coverage(
            COMPARISON,
            [],
            claim_texts=["LLMs predict tokens.", "Neural networks classify images."],
        ).answered

    def test_the_route_is_comparison_only(self) -> None:
        """Non-vacuity in the other direction: a definition is not
        answered by mentioning its subject in some optional claim."""
        coverage = assess_coverage(
            DEFINITION,
            ["distinguishing_property"],
            claim_texts=["A vector database stores embeddings."],
        )
        assert not coverage.answered
        assert "definition" in coverage.missing_core


class TestTheNarrowingIsOnTheRealPath:
    """The helper above is not the behaviour. This drives
    `_check_entailment` with a judge that says no.

    Written because removing the narrowing from the production path
    broke none of the tests above -- they exercised the predicate in
    isolation while the demotion lived somewhere else. That is the
    defect this project has now found seven times: a value or a rule
    tested where it is defined rather than where it is used.
    """

    @staticmethod
    def _ctx(monkeypatch, judge_says: bool):
        from types import SimpleNamespace

        from agentic_research.citations.nli import NLIPrediction, NLIScores
        from agentic_research.config import Settings
        from agentic_research.graph.nodes import reporting
        from agentic_research.schemas import RelevanceOut, RelevanceVerdictOut
        from fakes import FakeRouter

        class Entails:
            model_id = "fake/entails"
            revision = "test"

            def score(self, pairs):
                return [
                    NLIPrediction(
                        premise=p,
                        hypothesis=h,
                        scores=NLIScores(entailment=0.999, neutral=0.001, contradiction=0.0),
                        model_id=self.model_id,
                        model_revision=self.revision,
                    )
                    for p, h in pairs
                ]

        router = FakeRouter()
        router.responses["RelevanceOut"] = lambda user: RelevanceOut(
            verdicts=[
                RelevanceVerdictOut(claim_index=i, answers_question=judge_says, reason="scripted")
                for i, _ in enumerate([ln for ln in user.splitlines() if ln[:1].isdigit()])
            ]
        )
        settings = Settings(_env_file=None, nli_support_threshold=0.98)
        monkeypatch.setattr(
            reporting,
            "ctx",
            lambda: SimpleNamespace(settings=settings, router=router, nli_scorer=Entails()),
        )

    # A `direct_contrast` claim has to actually contrast, or the
    # structural check refuses it before the judge ever sees it --
    # correctly. The first version of this fixture used a one-sided
    # claim for both slots and failed for that reason, not because of
    # the narrowing.
    TEXTS = {
        "direct_contrast": "Large language models differ from neural networks in scale.",
        "other": "Neural networks classify images into categories.",
    }

    @staticmethod
    def _report_and_store(slot: str):
        from agentic_research.evidence.store import EvidenceStore
        from agentic_research.models import (
            Claim,
            ClaimKind,
            ContentOrigin,
            EvidenceItem,
            FetchStatus,
            QuoteMatch,
            ResearchReport,
            SourceDocument,
            Stance,
        )

        quote = TestTheNarrowingIsOnTheRealPath.TEXTS.get(
            slot, TestTheNarrowingIsOnTheRealPath.TEXTS["other"]
        )
        source = SourceDocument(
            id="S1",
            url="https://e.example.com/p",
            canonical_url="https://e.example.com/p",
            title="t",
            domain="e.example.com",
            source_type="other",
            content_origin=ContentOrigin.PROVIDER_RAW,
            fetch_status=FetchStatus.PROVIDER_CONTENT,
            quality_score=0.6,
            text=quote,
        )
        item = EvidenceItem(
            id="S1-e1",
            source_id="S1",
            sub_question_id="Q1",
            claim="c",
            quote=quote,
            quote_match=QuoteMatch.EXACT_NORMALIZED,
            stance=Stance.SUPPORTS,
            relevance=0.9,
        )
        claim = Claim(
            text=quote,
            evidence_ids=["S1-e1"],
            citation_ids=["S1"],
            kind=ClaimKind.FACTUAL,
            answer_slot=slot,
        )
        return ResearchReport(title="T", summary_claims=[claim]), EvidenceStore([source], [item])

    async def _run(self, monkeypatch, slot: str, judge_says: bool) -> bool:
        from agentic_research.graph.nodes import reporting
        from agentic_research.models import CitationVerification

        self._ctx(monkeypatch, judge_says)
        report, store = self._report_and_store(slot)
        result = CitationVerification()
        await reporting._check_entailment(report, store, result, {"contract": COMPARISON})
        return bool(result.judgments) and result.judgments[0].publishable

    async def test_a_dimension_claim_publishes_despite_a_negative_judgement(
        self, monkeypatch
    ) -> None:
        """The whole point. The judge says no; the claim fills an
        optional part; it publishes, and the report's coverage decides
        whether the parts added up."""
        assert await self._run(monkeypatch, "dimension", judge_says=False)

    async def test_a_core_slot_claim_is_still_withheld_by_the_judge(self, monkeypatch) -> None:
        """Non-vacuity, and the safety property. The judge keeps its
        authority over the slot the answer turns on."""
        assert not await self._run(monkeypatch, "direct_contrast", judge_says=False)

    async def test_a_slotless_claim_is_still_withheld_by_the_judge(self, monkeypatch) -> None:
        assert not await self._run(monkeypatch, "", judge_says=False)

    async def test_a_positive_judgement_publishes_either_way(self, monkeypatch) -> None:
        assert await self._run(monkeypatch, "dimension", judge_says=True)
        assert await self._run(monkeypatch, "direct_contrast", judge_says=True)
