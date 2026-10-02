"""The analyst wrote sentences first and attached quotes afterwards.

From the hosted v1.12 run on "langchain vs langgraph differences?", the
first in which retrieval worked: four first-party documentation pages at
0.90-0.98, and **five of eight claims refused**. Four of those five cited
S1 -- GeeksforGeeks, quality 0.64, the worst source in the set -- at
entailment 0.0013-0.0064. Those are not near misses; the quote does not
carry the claim at all. The three official docs pages were never quoted.

The failed claims were all contrast-shaped, and S1 was the only page
whose *title* was a comparison. So the analyst reached for the source
that looked like the answer and retrofitted ids to sentences it had
already decided to write.

Two changes, tested separately because they carry different risk:

* the refusal now names *which* kind of failure it was -- diagnosis only,
  computed after the verdict, incapable of withholding anything;
* the evidence block states which subjects each axis covers and tags
  every item with the subject it names, so a contrast is written where
  the evidence is rather than where the analyst hoped it would be.
"""

from __future__ import annotations

from agentic_research.citations.publication import filter_report_by_verification
from agentic_research.citations.semantic import CitedEvidence
from agentic_research.comparison import subjects_mentioned
from agentic_research.evidence.store import EvidenceStore
from agentic_research.evidence.topicality import alias_groups
from agentic_research.graph.nodes.reporting import _cites_off_subject
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
    SubQuestion,
)

SUBJECTS = ("LangChain", "LangGraph")
ALIASES = alias_groups("langchain vs langgraph differences?", list(SUBJECTS))

LIVE_CLAIM = (
    "LangChain represents workflows as sequences of operations in which "
    "each step's output becomes the next step's input."
)


def quote(text: str) -> list[CitedEvidence]:
    return [CitedEvidence(evidence_id="E1", quote=text)]


class TestTheRefusalNamesWhichFailureItWas:
    def test_a_quote_about_something_else_is_flagged(self) -> None:
        assert _cites_off_subject(
            LIVE_CLAIM,
            quote("Checkpointers let a graph resume from the last completed step."),
        )

    def test_a_quote_about_the_other_subject_is_flagged(self) -> None:
        """The live shape: a LangChain claim resting on LangGraph evidence."""
        assert _cites_off_subject(
            LIVE_CLAIM, quote("LangGraph exposes nodes and edges for branching control flow.")
        )

    def test_a_quote_on_the_claims_subject_is_not(self) -> None:
        assert not _cites_off_subject(
            LIVE_CLAIM,
            quote("LangChain chains compose operations so each step's output feeds the next."),
        )

    def test_a_claim_citing_nothing_is_not_flagged(self) -> None:
        """There is no binding to diagnose, and no quote to blame."""
        assert not _cites_off_subject(LIVE_CLAIM, [])

    def test_one_good_quote_among_several_clears_it(self) -> None:
        pairs = [
            CitedEvidence(evidence_id="E1", quote="Checkpointers resume an interrupted graph."),
            CitedEvidence(
                evidence_id="E2",
                quote="LangChain chains compose operations so each step's output feeds the next.",
            ),
        ]
        assert not _cites_off_subject(LIVE_CLAIM, pairs)

    def test_an_acronym_subject_is_not_called_off_subject(self) -> None:
        """Alias-aware, like the coverage prefilter it reuses."""
        assert not _cites_off_subject(
            "It persists state through checkpointers.",
            quote("LangGraph persists state through checkpointers."),
            ALIASES,
        )


class TestItIsADiagnosisAndNeverAGate:
    """The design promise, and the reason it is computed after the verdict.

    A lexical check that could withhold a claim would be a sixth
    deterministic gate, added in the release that fixed two of them for
    refusing true claims. So this must be unable to remove anything.
    """

    def test_an_off_subject_count_never_removes_a_claim(self) -> None:
        report = ResearchReport(
            title="T",
            sections=[
                ReportSection(heading="H", claims=[Claim(text=LIVE_CLAIM, evidence_ids=("S1-e1",))])
            ],
        )
        from agentic_research.citations.publication import claim_key

        verdicts = {claim_key(LIVE_CLAIM, ("S1-e1",)): "supported"}
        kept, removed = filter_report_by_verification(report, verdicts, off_subject=99)
        assert removed == 0
        assert kept.sections[0].claims[0].text == LIVE_CLAIM

    def test_the_cause_is_named_when_there_is_one(self) -> None:
        report = ResearchReport(
            title="T",
            sections=[
                ReportSection(heading="H", claims=[Claim(text=LIVE_CLAIM, evidence_ids=("S1-e1",))])
            ],
        )
        kept, removed = filter_report_by_verification(report, {}, off_subject=4)
        assert removed == 1
        joined = " ".join(kept.limitations)
        assert "4 cited a quote that was not about the" in joined
        assert "how the claim was assembled" in joined

    def test_no_cause_is_invented_when_there_is_none(self) -> None:
        report = ResearchReport(
            title="T",
            sections=[
                ReportSection(heading="H", claims=[Claim(text=LIVE_CLAIM, evidence_ids=("S1-e1",))])
            ],
        )
        kept, _ = filter_report_by_verification(report, {}, off_subject=0)
        joined = " ".join(kept.limitations)
        assert "not about the" not in joined
        assert "were excluded because the cited evidence" in joined


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
        quality_score=0.95,
        text="body",
    )


def item(eid: str, sid: str, claim: str, quote_text: str) -> EvidenceItem:
    return EvidenceItem(
        id=eid,
        source_id=sid,
        sub_question_id="Q1",
        claim=claim,
        quote=quote_text,
        quote_match=QuoteMatch.EXACT_NORMALIZED,
        stance=Stance.SUPPORTS,
        relevance=0.9,
    )


AXIS = SubQuestion(id="Q1", text="How do they differ in workflow control?")


def package_text(items: list[EvidenceItem], subjects: tuple[str, ...] = SUBJECTS) -> str:
    store = EvidenceStore([source("S1")], items)
    return store.build_package([AXIS], comparison_subjects=subjects, aliases=ALIASES).text


class TestTheBlockSaysWhichAxisCanCarryAContrast:
    """Official docs are single-subject -- a vendor does not document its
    competitor -- so a contrast has to be assembled per subject. The
    synthesiser was left to infer which axes had both sides."""

    BOTH = [
        item("S1-e1", "S1", "LangChain uses an implicit loop.", "LangChain uses an implicit loop."),
        item(
            "S1-e2",
            "S1",
            "LangGraph makes the graph explicit.",
            "LangGraph makes the execution graph explicit.",
        ),
    ]
    ONE = [
        item(
            "S1-e1",
            "S1",
            "LangGraph persists state.",
            "LangGraph persists state through checkpointers.",
        )
    ]

    def test_both_sides_present_invites_a_contrast(self) -> None:
        text = package_text(self.BOTH)
        assert "A contrast on this axis is possible" in text
        assert "LangChain, LangGraph" in text

    def test_one_side_only_forbids_a_contrast_and_says_which_is_missing(self) -> None:
        text = package_text(self.ONE)
        assert "cannot carry a contrast -- do not write one" in text
        assert "Nothing below speaks about LangChain" in text

    def test_neither_subject_named_is_stated_plainly(self) -> None:
        text = package_text(
            [item("S1-e1", "S1", "Checkpointers resume graphs.", "Checkpointers resume a graph.")]
        )
        assert "no evidence here speaks about either subject by name" in text

    def test_each_item_carries_its_own_subject_tag(self) -> None:
        """The heading is not enough: choosing an id happens per item."""
        text = package_text(self.BOTH)
        assert "[LangChain]" in text
        assert "[LangGraph]" in text

    def test_an_item_naming_both_is_tagged_with_both(self) -> None:
        text = package_text(
            [
                item(
                    "S1-e1",
                    "S1",
                    "LangGraph builds on LangChain.",
                    "LangGraph is built on LangChain components.",
                )
            ]
        )
        assert "[LangChain + LangGraph]" in text

    def test_a_non_comparison_block_is_untagged(self) -> None:
        """No subjects declared, so nothing to say and nothing added."""
        text = package_text(self.BOTH, subjects=())
        assert "[" not in text.split("### Sources")[0].replace("[S1]", "")
        assert "contrast" not in text


class TestTheSubjectHelper:
    def test_it_returns_subjects_in_the_declared_order(self) -> None:
        assert subjects_mentioned(
            "LangGraph is built on LangChain components.", SUBJECTS, ALIASES
        ) == ("LangChain", "LangGraph")

    def test_it_returns_nothing_when_neither_is_named(self) -> None:
        assert subjects_mentioned("Checkpointers resume a graph.", SUBJECTS, ALIASES) == ()
