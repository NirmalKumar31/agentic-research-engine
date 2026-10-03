"""Two defects from the v1.14 hosted run — the best run measured.

That run answered the question with two complete pairs, 7 claims and 35%
evidence yield, and the new source accounting finally named the next
bottleneck instead of leaving it to be guessed:

    S3 reference.langchain.com 0.96 — not cited, 2 citable quotes extracted
    S4 docs.langchain.com      0.96 — not cited, 1 citable quote extracted
    S6 docs.langchain.com      0.91 — not cited, 2 citable quotes extracted

Five usable quotes from first-party documentation, none cited, while
**six of seven citations went to the two lowest-quality sources in the
set** (0.62 and 0.64). Extraction was fine. Selection was not.

The second defect was in the same report: two published claims asserting
the same thing with the words rearranged.
"""

from __future__ import annotations

import pytest

from agentic_research.citations.publication import (
    _content_words,
    _restates,
    deduplicate_claims,
)
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
    SubQuestion,
)

AXIS = SubQuestion(id="Q1", text="How do they differ?")


def source(sid: str, kind: SourceType, quality: float) -> SourceDocument:
    return SourceDocument(
        id=sid,
        url=f"https://{sid.lower()}.example.com/p",
        canonical_url=f"https://{sid.lower()}.example.com/p",
        title=sid,
        domain=f"{sid.lower()}.example.com",
        source_type=kind,
        content_origin=ContentOrigin.PROVIDER_RAW,
        fetch_status=FetchStatus.PROVIDER_CONTENT,
        quality_score=quality,
        text="body",
    )


def item(eid: str, sid: str, text: str) -> EvidenceItem:
    return EvidenceItem(
        id=eid,
        source_id=sid,
        sub_question_id="Q1",
        claim=text,
        quote=text,
        quote_match=QuoteMatch.EXACT_NORMALIZED,
        stance=Stance.SUPPORTS,
        relevance=0.9,
    )


def package(sources, items) -> str:
    return EvidenceStore(sources, items).build_package([AXIS]).text


class TestTheSourceKindIsOnTheLineWhereTheIdIsChosen:
    """The sort already preferred authority. A sort is invisible.

    `build_package` has ranked by source authority for several releases,
    and the model reading the list has no way to know the order means
    anything. The classification existed the whole time and reached
    nothing that made a decision.
    """

    SOURCES = [source("S1", SourceType.OTHER, 0.64), source("S2", SourceType.OFFICIAL_DOCS, 0.96)]
    ITEMS = [item("S1-e1", "S1", "A listicle sentence."), item("S2-e1", "S2", "A docs sentence.")]

    def test_the_kind_and_score_appear_on_the_item_line(self) -> None:
        text = package(self.SOURCES, self.ITEMS)
        assert "(supports, official docs 0.96)" in text
        assert "(supports, other 0.64)" in text

    def test_it_is_on_the_evidence_line_not_only_in_the_source_list(self) -> None:
        """Choosing an id happens per item, so the fact has to be there."""
        body = package(self.SOURCES, self.ITEMS).split("### Sources")[0]
        assert "official docs 0.96" in body

    def test_the_source_id_is_still_withheld_from_the_item_line(self) -> None:
        """Showing it would invite citing sources directly, which this
        design removes on purpose. The *kind* is not the id."""
        body = package(self.SOURCES, self.ITEMS).split("### Sources")[0]
        for line in body.splitlines():
            if line.startswith("- S"):
                assert "[S1]" not in line and "[S2]" not in line

    def test_underscores_are_not_shown_to_the_model(self) -> None:
        assert "official_docs" not in package(self.SOURCES, self.ITEMS).split("### Sources")[0]

    def test_the_prompt_tells_it_what_to_do_with_that(self) -> None:
        """Showing the score is useless if nothing says to act on it."""
        from agentic_research.graph.prompts import SYNTHESIZER_SYSTEM

        assert "more authoritative one" in SYNTHESIZER_SYSTEM

    def test_the_prompt_keeps_relevance_in_front_of_authority(self) -> None:
        """The failure this rule could cause if stated alone: citing a
        good source for a point its quote does not carry."""
        from agentic_research.graph.prompts import SYNTHESIZER_SYSTEM

        assert "does not override relevance" in SYNTHESIZER_SYSTEM


# The two claims the v1.14 run published, from one source, in one slot.
RESTATEMENT_A = (
    "LangChain components are the components on which LangGraph's orchestration layer is built."
)
RESTATEMENT_B = "LangGraph is an orchestration layer built on LangChain components."

PAIR_LC = "LangChain is intended to connect large language models into structured workflows."
PAIR_LG = (
    "LangGraph models agent workflows as looping, stateful graphs built from "
    "state, nodes, and edges."
)


def claim(text: str, slot: str = "relationship", evidence: str = "S5-e1") -> Claim:
    return Claim(text=text, evidence_ids=(evidence,), answer_slot=slot)


def dedupe(claims: list[Claim]) -> tuple[int, list[str]]:
    report = ResearchReport(title="T", sections=[ReportSection(heading="H", claims=claims)])
    out, removed = deduplicate_claims(report)
    return removed, [c.text for s in out.sections for c in s.claims]


class TestARestatementIsADuplicate:
    def test_the_live_pair_collapses_to_one(self) -> None:
        removed, kept = dedupe([claim(RESTATEMENT_A), claim(RESTATEMENT_B, evidence="S5-e4")])
        assert removed == 1
        assert kept == [RESTATEMENT_A]

    def test_the_earliest_survives(self) -> None:
        """Same precedence as exact duplicates: the synthesiser led with it."""
        removed, kept = dedupe([claim(RESTATEMENT_B), claim(RESTATEMENT_A, evidence="S5-e4")])
        assert removed == 1
        assert kept == [RESTATEMENT_B]

    def test_a_different_evidence_id_does_not_save_it(self) -> None:
        """A restatement may quote a different sentence from one page."""
        removed, _ = dedupe(
            [claim(RESTATEMENT_A, evidence="S5-e2"), claim(RESTATEMENT_B, evidence="S9-e9")]
        )
        assert removed == 1


class TestItDoesNotCollapseWhatItMustNot:
    """This overrides a documented refusal to be fuzzy, so the
    non-firing cases matter more than the firing one.

    The rule is set equality on content words within one slot — not a
    similarity threshold. Two claims that merely resemble each other are
    two claims.
    """

    def test_the_two_sides_of_a_comparison_pair_both_survive(self) -> None:
        """The case that would have made this change unshippable.

        Each side names a different subject, so the word sets differ.
        Safe by construction rather than by a tuned threshold.
        """
        removed, kept = dedupe(
            [claim(PAIR_LC, slot="purpose"), claim(PAIR_LG, slot="purpose", evidence="S2-e1")]
        )
        assert removed == 0
        assert len(kept) == 2

    def test_the_same_words_in_different_slots_both_survive(self) -> None:
        removed, _ = dedupe([claim(RESTATEMENT_A), claim(RESTATEMENT_B, slot="dimension")])
        assert removed == 0

    def test_a_claim_with_no_slot_is_never_collapsed(self) -> None:
        """Without a slot nothing says the two address the same part."""
        removed, _ = dedupe([claim(RESTATEMENT_A, slot=""), claim(RESTATEMENT_B, slot="")])
        assert removed == 0

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            # One extra content word is one extra assertion.
            ("LangGraph persists state.", "LangGraph persists state durably."),
            # Similar, not identical — a threshold would collapse these.
            ("LangGraph uses nodes and edges.", "LangGraph uses typed nodes and edges."),
        ],
    )
    def test_near_misses_are_kept(self, a: str, b: str) -> None:
        removed, _ = dedupe([claim(a), claim(b, evidence="S1-e2")])
        assert removed == 0


class TestContentWords:
    def test_a_possessive_yields_the_bare_word(self) -> None:
        """Not special-cased — the word pattern and length filter do it."""
        assert _content_words("LangGraph's orchestration layer") == _content_words(
            "LangGraph orchestration layer"
        )

    def test_stopwords_are_dropped(self) -> None:
        assert _content_words("it is on the layer") == frozenset({"layer"})

    def test_an_empty_claim_restates_nothing(self) -> None:
        assert not _restates(claim(""), claim(""))
