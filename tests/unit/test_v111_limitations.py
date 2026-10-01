"""The Limitations section must not contradict the answer above it.

From the "langchain vs langgraph" live run, which published a table of
two named dimensions, each carrying both subjects, and then said:

    The evidence did not establish a named dimension along which they
    differ.
    More than one published claim fills the Purpose and abstraction
    level slot; they may repeat each other.
    More than one published claim fills the State management and
    persistence slot; they may repeat each other.

All three are false, and a report arguing with itself costs more trust
than the gaps it was describing. The first mistook the contract's
*fallback* axis for a requirement; the other two described the two sides
of a working contrast as a possible defect.
"""

from __future__ import annotations

from agentic_research.answer_contract import QuestionType, build_contract
from agentic_research.answer_coverage import AnswerCoverage
from agentic_research.comparison import ComparisonPair, SideClaim

QUESTION = "langchain vs langgraph differences?"
AXIS = "purpose_and_abstraction_level"


def comparison_contract(*dimensions: str):
    return build_contract(
        QUESTION,
        QuestionType.COMPARISON,
        entities=["LangChain", "LangGraph"],
        comparison_subjects=["LangChain", "LangGraph"],
        dimensions=list(dimensions),
    )


def named_pair(dimension: str = AXIS) -> ComparisonPair:
    return ComparisonPair(
        dimension=dimension,
        sides=(
            SideClaim("LangChain", "LangChain connects LLMs into workflows.", dimension, ("S1-e1",)),
            SideClaim("LangGraph", "LangGraph runs graph-based workflows.", dimension, ("S1-e2",)),
        ),
    )


class TestTheFallbackAxisIsNotAMissingRequirement:
    def test_generic_dimension_is_silent_when_a_named_axis_carried_a_pair(self) -> None:
        contract = comparison_contract(AXIS)
        coverage = AnswerCoverage(
            contract=contract,
            satisfied=(AXIS,),
            duplicates=(),
            quality_by_slot={},
            comparison_pairs=(named_pair(),),
        )
        joined = " ".join(coverage.limitations())
        assert "named dimension along which they differ" not in joined

    def test_but_it_is_reported_when_no_pair_was_formed(self) -> None:
        """The suppression is conditional, not a deletion of the gap."""
        contract = comparison_contract(AXIS)
        coverage = AnswerCoverage(
            contract=contract,
            satisfied=(),
            duplicates=(),
            quality_by_slot={},
            comparison_pairs=(),
        )
        joined = " ".join(coverage.limitations())
        assert "named dimension along which they differ" in joined

    def test_a_pair_on_the_generic_axis_alone_does_not_suppress_it(self) -> None:
        """Only a *named* axis counts. A contrast on "dimension" is the
        fallback itself, so it cannot be the evidence that the fallback
        was unnecessary."""
        contract = comparison_contract()
        coverage = AnswerCoverage(
            contract=contract,
            satisfied=(),
            duplicates=(),
            quality_by_slot={},
            comparison_pairs=(named_pair("dimension"),),
        )
        joined = " ".join(coverage.limitations())
        assert "named dimension along which they differ" in joined


class TestTwoClaimsInAPairedSlotIsNotADefect:
    def test_no_duplicate_warning_for_a_slot_carrying_a_pair(self) -> None:
        """One claim per subject is what makes the pair a contrast."""
        contract = comparison_contract(AXIS)
        coverage = AnswerCoverage(
            contract=contract,
            satisfied=(AXIS,),
            duplicates=(AXIS,),
            quality_by_slot={},
            comparison_pairs=(named_pair(),),
        )
        joined = " ".join(coverage.limitations())
        assert "may repeat each other" not in joined

    def test_but_an_unpaired_slot_still_warns(self) -> None:
        """Outside a contrast, two claims in one slot may genuinely repeat."""
        contract = comparison_contract(AXIS)
        coverage = AnswerCoverage(
            contract=contract,
            satisfied=(AXIS,),
            duplicates=("relationship",),
            quality_by_slot={},
            comparison_pairs=(named_pair(),),
        )
        joined = " ".join(coverage.limitations())
        assert "More than one published claim fills the relationship slot" in joined


class TestRealGapsSurvive:
    """The point was to remove false statements, not to shorten the list."""

    def test_an_unsatisfied_named_axis_is_still_reported(self) -> None:
        contract = comparison_contract(AXIS, "workflow_and_control_model")
        coverage = AnswerCoverage(
            contract=contract,
            satisfied=(AXIS,),
            duplicates=(),
            quality_by_slot={},
            comparison_pairs=(named_pair(),),
        )
        joined = " ".join(coverage.limitations())
        assert "workflow and control model" in joined.lower()
