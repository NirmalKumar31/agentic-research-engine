"""Two losses the v1.13 run showed, neither of them an evidence failure.

That run finally answered the question -- two complete pairs, 6 claims,
20.7% yield -- and in doing so made two different defects legible.

**The slot label.** Two verified claims, one defining each subject, were
deleted for declaring `direct_contrast` while naming a single subject.
That slot is filled by the per-subject claims the engine pairs; it is
never aimed at. Labelled with the axis they addressed they were a third
complete pair -- and the report went on to say "The evidence did not
establish how the subjects differ on purpose and abstraction level" in
the same Limitations list.

**The uncited source.** AWS Prescriptive Guidance on LangChain *and*
LangGraph -- 0.97, official docs, discussing both subjects where
first-party documentation structurally cannot -- was retrieved and never
cited. "retrieved, not cited" covered both "nothing citable came out of
it" and "plenty did and none was used", which need opposite fixes.
"""

from __future__ import annotations

from agentic_research.citations.publication import filter_report_by_verification
from agentic_research.evidence.store import EvidenceStore
from agentic_research.models import (
    Claim,
    ContentOrigin,
    EvidenceItem,
    FetchStatus,
    QuoteMatch,
    ReportSection,
    ResearchReport,
    SourceDocument,
    SourceType,
    Stance,
)
from agentic_research.report import render_markdown

CLAIM = "LangChain is a modular framework for building LLM-powered applications."


def report_of(text: str = CLAIM) -> ResearchReport:
    return ResearchReport(
        title="T",
        sections=[ReportSection(heading="H", claims=[Claim(text=text, evidence_ids=("S1-e1",))])],
    )


class TestTheSlotLabelLossIsReported:
    def test_the_count_reaches_the_reader(self) -> None:
        kept, removed = filter_report_by_verification(report_of(), {}, mislabelled_contrast=2)
        assert removed == 1
        joined = " ".join(kept.limitations)
        assert "A further 2 were verified and then deleted" in joined
        assert "declaring the contrast slot while describing a single subject" in joined

    def test_it_says_the_claims_could_have_been_paired(self) -> None:
        """The actionable half: the fix is the label, not the evidence."""
        kept, _ = filter_report_by_verification(report_of(), {}, mislabelled_contrast=1)
        assert "could have been paired" in " ".join(kept.limitations)

    def test_nothing_is_said_when_none_were_lost_that_way(self) -> None:
        kept, _ = filter_report_by_verification(report_of(), {}, mislabelled_contrast=0)
        assert "declaring the contrast slot" not in " ".join(kept.limitations)

    def test_it_sits_alongside_the_off_subject_cause(self) -> None:
        """Both are causes of the same exclusion count, not alternatives."""
        kept, _ = filter_report_by_verification(
            report_of(), {}, off_subject=3, mislabelled_contrast=2
        )
        joined = " ".join(kept.limitations)
        assert "3 cited a quote that was not about" in joined
        assert "A further 2 were verified" in joined

    def test_the_count_never_removes_a_claim(self) -> None:
        from agentic_research.citations.publication import claim_key

        verdicts = {claim_key(CLAIM, ("S1-e1",)): "supported"}
        kept, removed = filter_report_by_verification(
            report_of(), verdicts, mislabelled_contrast=99
        )
        assert removed == 0
        assert kept.sections[0].claims[0].text == CLAIM


class TestTheSynthesiserIsToldNotToAimAtTheContrastSlot:
    def test_the_rule_is_in_the_prompt(self) -> None:
        from agentic_research.answer_contract import QuestionType, build_contract
        from agentic_research.graph.prompts import synthesizer_user

        contract = build_contract(
            "langchain vs langgraph differences?",
            QuestionType.COMPARISON,
            entities=["LangChain", "LangGraph"],
            comparison_subjects=["LangChain", "LangGraph"],
            dimensions=["purpose_and_abstraction_level"],
        )
        text = synthesizer_user(
            "q",
            "comparison",
            "EV",
            "",
            answer_slots=contract.required_slots,
            comparison_subjects=contract.subjects_to_span,
        )
        assert "Never give a single-subject claim the answer_slot" in text
        assert "fills itself from the per-subject claims" in text

    def test_it_is_only_said_for_a_comparison(self) -> None:
        from agentic_research.graph.prompts import synthesizer_user

        text = synthesizer_user("q", "overview", "EV", "")
        assert "direct_contrast" not in text


def source(sid: str) -> SourceDocument:
    return SourceDocument(
        id=sid,
        url=f"https://{sid.lower()}.example.com/p",
        canonical_url=f"https://{sid.lower()}.example.com/p",
        title=f"{sid} title",
        domain=f"{sid.lower()}.example.com",
        source_type=SourceType.OFFICIAL_DOCS,
        content_origin=ContentOrigin.PROVIDER_RAW,
        fetch_status=FetchStatus.PROVIDER_CONTENT,
        quality_score=0.97,
        text="body",
    )


def citable(eid: str, sid: str) -> EvidenceItem:
    return EvidenceItem(
        id=eid,
        source_id=sid,
        sub_question_id="Q1",
        claim="c",
        quote="a located quote",
        quote_match=QuoteMatch.EXACT_NORMALIZED,
        stance=Stance.SUPPORTS,
        relevance=0.9,
    )


def uncitable(eid: str, sid: str) -> EvidenceItem:
    """A quote that could not be found in the source text."""
    return EvidenceItem(
        id=eid,
        source_id=sid,
        sub_question_id="Q1",
        claim="c",
        quote="a quote nobody could locate",
        quote_match=QuoteMatch.NONE,
        stance=Stance.SUPPORTS,
        relevance=0.9,
    )


class TestAnUncitedSourceSaysWhichKindItWas:
    """Two different failures, opposite fixes, one sentence covering both."""

    def test_evidence_extracted_but_unused_says_how_much(self) -> None:
        sources = [source("S1")]
        items = [citable("S1-e1", "S1"), citable("S1-e2", "S1")]
        body = render_markdown(ResearchReport(title="T"), sources, None, evidence=items)
        assert "retrieved, not cited — 2 citable quotes extracted" in body

    def test_one_quote_is_singular(self) -> None:
        body = render_markdown(
            ResearchReport(title="T"), [source("S1")], None, evidence=[citable("S1-e1", "S1")]
        )
        assert "1 citable quote extracted" in body
        assert "quotes extracted" not in body

    def test_nothing_extractable_says_so_instead(self) -> None:
        body = render_markdown(
            ResearchReport(title="T"), [source("S1")], None, evidence=[uncitable("S1-e1", "S1")]
        )
        assert "no citable quote could be extracted" in body

    def test_a_source_with_no_evidence_at_all_says_so(self) -> None:
        body = render_markdown(ResearchReport(title="T"), [source("S1")], None, evidence=[])
        assert "no citable quote could be extracted" in body

    def test_a_cited_source_carries_no_note(self) -> None:
        report = ResearchReport(
            title="T",
            sections=[
                ReportSection(
                    heading="H",
                    claims=[Claim(text="c", evidence_ids=("S1-e1",), citation_ids=("S1",))],
                )
            ],
        )
        body = render_markdown(report, [source("S1")], None, evidence=[citable("S1-e1", "S1")])
        assert "retrieved, not cited" not in body


class TestTheCounterIsOnTheRealPath:
    """The counter lives in `_check_entailment`, so it is tested there.

    Written because two mutations survived the first version of this
    file: removing the increment entirely, and making it fire for every
    slot. Both survived because every test above drove
    `filter_report_by_verification` with the count handed to it --
    exercising where the value is *consumed*, never where it is
    *produced*. That is the defect this repo has now produced a dozen
    times, and I reproduced it in the tests for a fix about exactly this
    kind of mistake.
    """

    COMPARISON = None  # built in `_contract` to keep imports local

    @staticmethod
    def _contract():
        from agentic_research.answer_contract import QuestionType, build_contract

        return build_contract(
            "How does a large language model differ from a neural network?",
            QuestionType.COMPARISON,
            entities=("large language model", "neural network"),
            comparison_subjects=("large language model", "neural network"),
            dimensions=("input_domain",),
        )

    @staticmethod
    def _ctx(monkeypatch):
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
                RelevanceVerdictOut(claim_index=i, answers_question=True, reason="scripted")
                for i, _ in enumerate([ln for ln in user.splitlines() if ln[:1].isdigit()])
            ]
        )
        settings = Settings(_env_file=None, nli_support_threshold=0.98)
        monkeypatch.setattr(
            reporting,
            "ctx",
            lambda: SimpleNamespace(settings=settings, router=router, nli_scorer=Entails()),
        )

    # Names one subject only, so the structural check refuses it for the
    # contrast slot -- which is the live shape: a definition of one side.
    ONE_SIDED = "A large language model is trained on text."

    # Mentions neither subject, so relevance refuses it whatever slot it
    # declares. Needed to reach the counter's branch with a slot that is
    # *not* `direct_contrast` -- a one-sided claim on a named axis is
    # publishable, so it never enters the branch and cannot show that the
    # slot condition is doing any work.
    OFF_TOPIC = "Gradient boosting builds an ensemble of shallow trees."

    @staticmethod
    def _report_and_store(slot: str, text: str | None = None):
        from agentic_research.models import (
            ClaimKind,
            ContentOrigin,
            FetchStatus,
            SourceDocument,
        )

        text = text or TestTheCounterIsOnTheRealPath.ONE_SIDED
        src = SourceDocument(
            id="S1",
            url="https://e.example.com/p",
            canonical_url="https://e.example.com/p",
            title="t",
            domain="e.example.com",
            source_type="other",
            content_origin=ContentOrigin.PROVIDER_RAW,
            fetch_status=FetchStatus.PROVIDER_CONTENT,
            quality_score=0.6,
            text=text,
        )
        item = EvidenceItem(
            id="S1-e1",
            source_id="S1",
            sub_question_id="Q1",
            claim="c",
            quote=text,
            quote_match=QuoteMatch.EXACT_NORMALIZED,
            stance=Stance.SUPPORTS,
            relevance=0.9,
        )
        claim = Claim(
            text=text,
            evidence_ids=["S1-e1"],
            citation_ids=["S1"],
            kind=ClaimKind.FACTUAL,
            answer_slot=slot,
        )
        return ResearchReport(title="T", summary_claims=[claim]), EvidenceStore([src], [item])

    async def _run(self, monkeypatch, slot: str, text: str | None = None) -> int:
        from agentic_research.graph.nodes import reporting
        from agentic_research.models import CitationVerification

        self._ctx(monkeypatch)
        report, store = self._report_and_store(slot, text)
        result = CitationVerification()
        await reporting._check_entailment(report, store, result, {"contract": self._contract()})
        return result.mislabelled_contrast_claims

    async def test_a_one_sided_contrast_claim_is_counted(self, monkeypatch) -> None:
        assert await self._run(monkeypatch, "direct_contrast") == 1

    async def test_a_named_axis_claim_is_not_counted(self, monkeypatch) -> None:
        """Non-vacuity. The counter is about the label, not about every
        refusal: a one-sided claim on a named axis is perfectly legal and
        is exactly what the engine pairs."""
        assert await self._run(monkeypatch, "input_domain") == 0

    async def test_a_slotless_claim_is_not_counted(self, monkeypatch) -> None:
        assert await self._run(monkeypatch, "") == 0

    async def test_a_refused_claim_on_another_slot_is_not_counted(self, monkeypatch) -> None:
        """The condition is the *label*, and this is what proves it.

        A claim naming neither subject is refused whatever slot it
        declares, so this reaches the counter's branch with a slot that
        is not `direct_contrast`. Without it, replacing the slot check
        with `if True` changed nothing any test could see -- the other
        controls are publishable claims that never enter the branch.
        """
        assert await self._run(monkeypatch, "input_domain", self.OFF_TOPIC) == 0

    async def test_and_that_claim_really_was_refused(self, monkeypatch) -> None:
        """Non-vacuity for the test above: if the claim published, it
        would prove nothing about the counter."""
        from agentic_research.graph.nodes import reporting
        from agentic_research.models import CitationVerification

        self._ctx(monkeypatch)
        report, store = self._report_and_store("input_domain", self.OFF_TOPIC)
        result = CitationVerification()
        await reporting._check_entailment(report, store, result, {"contract": self._contract()})
        assert result.judgments
        assert not result.judgments[0].publishable
