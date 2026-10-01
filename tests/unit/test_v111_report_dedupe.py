"""Each published claim is rendered once, not two or three times.

Rebuilt from the "langchain vs langgraph" live run, whose report printed
four claims eight times: the Summary repeated the first comparison table
verbatim, and a section repeated the second. The coverage assessment
noticed and filed it as a *limitation* -- "More than one published claim
fills the X slot" -- rather than suppressing it, so the document both
repeated itself and apologised for doing so.

Tested through ``render_markdown``, the function a reader's output
actually comes from, rather than against the dedupe helper.
"""

from __future__ import annotations

from agentic_research.comparison import ComparisonPair, SideClaim
from agentic_research.models import Claim, ReportSection, ResearchReport
from agentic_research.report import render_markdown

PURPOSE_LC = (
    "LangChain is designed to connect LLMs into structured workflows for "
    "multi-step reasoning and automation."
)
PURPOSE_LG = (
    "LangGraph is designed for graph-based workflows supporting flexible "
    "and parallel multi-step reasoning."
)
STATE_LC = "LangChain's building blocks include memory."
STATE_LG = "LangGraph's persistence layer gives agents short-term memory through checkpointers."


def live_report() -> ResearchReport:
    return ResearchReport(
        title="LangChain and LangGraph: Key Differences",
        summary_claims=[
            Claim(text=PURPOSE_LC, evidence_ids=("S1-e1",)),
            Claim(text=PURPOSE_LG, evidence_ids=("S1-e2",)),
        ],
        sections=[
            ReportSection(
                heading="State management and persistence",
                claims=[
                    Claim(text=STATE_LC, evidence_ids=("S5-e1",)),
                    Claim(text=STATE_LG, evidence_ids=("S3-e1",)),
                ],
            )
        ],
    )


def live_pairs() -> tuple[ComparisonPair, ...]:
    return (
        ComparisonPair(
            dimension="Purpose and abstraction level",
            sides=(
                SideClaim("LangChain", PURPOSE_LC, "purpose", ("S1-e1",)),
                SideClaim("LangGraph", PURPOSE_LG, "purpose", ("S1-e2",)),
            ),
        ),
        ComparisonPair(
            dimension="State management and persistence",
            sides=(
                SideClaim("LangChain", STATE_LC, "state", ("S5-e1",)),
                SideClaim("LangGraph", STATE_LG, "state", ("S3-e1",)),
            ),
        ),
    )


class TestClaimsAreRenderedOnce:
    def body(self) -> str:
        return render_markdown(live_report(), [], None, comparison_pairs=live_pairs())

    def test_every_claim_appears_exactly_once(self) -> None:
        body = self.body()
        for claim in (PURPOSE_LC, PURPOSE_LG, STATE_LC, STATE_LG):
            assert body.count(claim) == 1, f"rendered {body.count(claim)}x: {claim[:50]}"

    def test_the_table_is_the_copy_that_survives(self) -> None:
        """It carries the axis label and attributes each side to a subject.

        A pair that lost a cell to an earlier heading would stop being a
        contrast, so the table has to win the duplicate.
        """
        body = self.body()
        assert "## How they differ" in body
        for pair in live_pairs():
            for side in pair.sides:
                assert f"| {side.subject} |" in body

    def test_headings_left_with_nothing_are_dropped(self) -> None:
        """An empty heading is worse than no heading: it reads as a gap.

        Asserted on the ``##`` form specifically. The comparison table
        prints the same words as a bold axis label, so a substring test
        without the heading marker would pass either way -- it did, and
        let a mutation that printed the empty heading survive.
        """
        body = self.body()
        assert "## Summary" not in body
        assert "## State management and persistence" not in body
        assert "**State management and persistence**" in body

    def test_a_claim_not_in_the_table_still_renders(self) -> None:
        """The dedupe must only bite where there is real duplication."""
        extra = "LangGraph supports human-in-the-loop interrupts."
        report = live_report()
        report.sections[0].claims.append(Claim(text=extra, evidence_ids=("S3-e9",)))
        body = render_markdown(report, [], None, comparison_pairs=live_pairs())
        assert body.count(extra) == 1
        assert "## State management and persistence" in body


class TestReportsWithoutATableAreUnchanged:
    """The speculative-decoding run had no comparison table at all."""

    def test_summary_survives_when_there_is_no_table(self) -> None:
        body = render_markdown(live_report(), [], None)
        assert "## Summary" in body
        assert body.count(PURPOSE_LC) == 1
        assert body.count(STATE_LC) == 1

    def test_a_claim_repeated_within_one_section_is_still_collapsed(self) -> None:
        report = ResearchReport(
            title="T",
            sections=[
                ReportSection(
                    heading="H",
                    claims=[
                        Claim(text=STATE_LC, evidence_ids=("S5-e1",)),
                        Claim(text=STATE_LC + ".", evidence_ids=("S5-e2",)),
                    ],
                )
            ],
        )
        body = render_markdown(report, [], None)
        assert body.count(STATE_LC) == 1
