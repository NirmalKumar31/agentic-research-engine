"""What an answer to this question has to contain.

The engine could publish a claim that was well cited, entailed by its
quote, and no answer to what was asked. A visitor asked how a large
language model differs from a neural network and received five
supported claims describing what a large language model *is*. Every
gate passed. Nothing in the pipeline had ever been told what the
question required.

The contract is that missing statement. It is built before retrieval,
from the question alone, and everything downstream refers to it:
retrieval ranks evidence against its slots, generation targets them,
publication refuses a claim that fills none, and the report reports
which are still empty.

Slots are canonical per question type rather than invented per run.
A model asked to decide both the shape of an answer and its content
will happily produce a shape its content already fits, which is the
failure this exists to stop. The model supplies the subject matter --
entities, dimensions, constraints; the shape is fixed here and can be
checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class QuestionType(StrEnum):
    DEFINITION = "definition"
    COMPARISON = "comparison"
    CAUSAL = "causal"
    NUMERIC = "numeric"
    TEMPORAL = "temporal"
    PROCEDURAL = "procedural"
    RECOMMENDATION = "recommendation"
    LIST = "list"
    SYNTHESIS = "synthesis"


@dataclass(frozen=True)
class AnswerSlot:
    """One thing the answer must contain."""

    name: str
    description: str
    core: bool = True
    """A core slot carries the answer. A report satisfying none of them
    has not answered the question, whatever else it contains."""
    satisfied_by: tuple[str, ...] = ()
    """Other slots that also discharge this one, when the question
    admits more than one shape of answer.

    Not a general escape hatch -- an empty tuple is the default and
    every slot but one has it. It exists because "how does X differ
    from Y" has two honest answers and only one of them is a contrast:
    when Y is a category containing X, no contrast exists to find, and
    the answer is where X sits inside Y."""


def _slot(
    name: str,
    description: str,
    core: bool = True,
    satisfied_by: tuple[str, ...] = (),
) -> AnswerSlot:
    return AnswerSlot(name=name, description=description, core=core, satisfied_by=satisfied_by)


# The shape of each kind of answer, fixed rather than generated.
CANONICAL_SLOTS: dict[QuestionType, tuple[AnswerSlot, ...]] = {
    QuestionType.DEFINITION: (
        _slot("definition", "What the subject is"),
        _slot("distinguishing_property", "What separates it from adjacent things", core=False),
    ),
    QuestionType.COMPARISON: (
        # The slot the live failure was missing. Definitions of each
        # side, however well supported, do not fill it.
        #
        # `relationship` discharges it because a comparison has two
        # honest answers. Hosted acceptance asked how a large language
        # model differs from a neural network, found and published that
        # one is a subset of the other, and then reported that it had
        # not answered -- because a subset is not a contrast. It was
        # the answer. When one subject is a category containing the
        # other there is no contrast to find, and demanding one makes
        # the engine wrong about itself.
        #
        # The cost, stated rather than hidden: for a genuine comparison
        # of two unrelated subjects, a vague relationship claim now
        # discharges the core slot too. The relevance judgement is the
        # backstop there -- a claim that does not help answer how they
        # differ is withheld before coverage ever sees its slot.
        _slot(
            "direct_contrast",
            "An explicit statement of how the subjects differ",
            satisfied_by=("relationship",),
        ),
        _slot("dimension", "A named dimension along which they differ", core=False),
        _slot(
            "relationship",
            "How the subjects relate, such as one being a kind of the other",
            core=False,
        ),
    ),
    QuestionType.CAUSAL: (
        _slot("causal_evidence", "Evidence for the causal link, not merely association"),
        _slot("candidate_drivers", "Factors the evidence supports", core=False),
        _slot("limitations", "What the evidence cannot establish about cause", core=False),
    ),
    QuestionType.NUMERIC: (
        _slot("measured_value", "The figure asked for, with its units"),
        _slot(
            "measurement_conditions", "How and under what conditions it was measured", core=False
        ),
        _slot("limitations", "What the figure does not establish", core=False),
    ),
    QuestionType.TEMPORAL: (
        _slot("figure_for_period", "The value or event for the period asked about"),
        _slot("period_confirmation", "Confirmation the evidence covers that period", core=False),
    ),
    QuestionType.PROCEDURAL: (
        _slot("steps", "What to do, in order"),
        _slot("preconditions", "What must be true before starting", core=False),
    ),
    QuestionType.RECOMMENDATION: (
        _slot("recommendation", "What the evidence supports doing"),
        _slot("tradeoffs", "What is given up", core=False),
    ),
    QuestionType.LIST: (
        _slot("items", "The members asked for"),
        _slot("completeness", "Whether the list is exhaustive", core=False),
    ),
    QuestionType.SYNTHESIS: (_slot("part_answer", "An answer to one named part of the question"),),
}

# Aliases the analysis stage already emits, mapped rather than
# re-derived, so the two classifications cannot disagree.
_OUTPUT_FORMAT_TO_TYPE: dict[str, QuestionType] = {
    "comparison": QuestionType.COMPARISON,
    "overview": QuestionType.DEFINITION,
    "howto": QuestionType.PROCEDURAL,
    "timeline": QuestionType.TEMPORAL,
    "decision_support": QuestionType.RECOMMENDATION,
}


@dataclass(frozen=True)
class AnswerContract:
    """What this question requires of an answer.

    ``usable`` is false when the question could not be turned into a
    contract with enough shape to check against. That is a refusal, not
    a default: guessing a shape would reintroduce exactly the problem
    of a pipeline that does not know what it is being asked.
    """

    question: str
    question_type: QuestionType
    entities: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    ambiguities: tuple[str, ...] = ()
    usable: bool = True
    unusable_reason: str = ""
    required_slots: tuple[AnswerSlot, ...] = field(default_factory=tuple)

    @property
    def core_slots(self) -> tuple[AnswerSlot, ...]:
        return tuple(s for s in self.required_slots if s.core)

    @property
    def slot_names(self) -> frozenset[str]:
        return frozenset(s.name for s in self.required_slots)

    def has_slot(self, name: str) -> bool:
        return name in self.slot_names

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "question_type": str(self.question_type),
            "entities": list(self.entities),
            "dimensions": list(self.dimensions),
            "constraints": list(self.constraints),
            "ambiguities": list(self.ambiguities),
            "usable": self.usable,
            "unusable_reason": self.unusable_reason,
            # satisfied_by travels with the slot. Without it a client
            # cannot tell that a core requirement was discharged by an
            # alternative, so it computes "unanswered" and renders that
            # directly beneath a report whose own limitations say the
            # opposite. The hosted v1.2.1 run produced exactly that
            # contradiction before this line existed.
            "required_slots": [
                {
                    "name": s.name,
                    "description": s.description,
                    "core": s.core,
                    "satisfied_by": list(s.satisfied_by),
                }
                for s in self.required_slots
            ],
        }


def unusable(question: str, reason: str) -> AnswerContract:
    """A refusal to guess at what the question requires."""
    return AnswerContract(
        question=question,
        question_type=QuestionType.SYNTHESIS,
        usable=False,
        unusable_reason=reason,
        required_slots=(),
    )


def build_contract(
    question: str,
    question_type: QuestionType | str,
    *,
    entities: list[str] | tuple[str, ...] = (),
    dimensions: list[str] | tuple[str, ...] = (),
    constraints: list[str] | tuple[str, ...] = (),
    ambiguities: list[str] | tuple[str, ...] = (),
    parts: list[str] | tuple[str, ...] = (),
) -> AnswerContract:
    """Assemble a contract, refusing rather than guessing.

    ``parts`` names the components of a multi-part question; each
    becomes its own core slot, so a report answering one of three parts
    cannot present itself as complete.
    """
    if not question.strip():
        return unusable(question, "the question is empty")

    try:
        qtype = QuestionType(str(question_type))
    except ValueError:
        return unusable(question, f"unrecognised question type {question_type!r}")

    slots = list(CANONICAL_SLOTS[qtype])

    # A comparison needs two sides. With fewer, there is nothing to
    # contrast and the contract would be satisfiable by a definition --
    # which is the defect being fixed.
    if qtype is QuestionType.COMPARISON and len(tuple(entities)) < 2:
        return unusable(
            question,
            f"a comparison needs at least two subjects; only {len(tuple(entities))} was identified",
        )

    # Named dimensions become their own slots. The canonical set fixes
    # that a comparison needs a contrast; which dimensions matter is
    # subject matter, and comes from the question.
    if qtype is QuestionType.COMPARISON:
        named = [d for d in dimensions if d.strip()]
        if named:
            # The generic placeholder only earns its place when the
            # question named no dimensions of its own.
            slots = [s for s in slots if s.name != "dimension"]
        slots += [
            _slot(d, f"How the subjects differ on {d.replace('_', ' ')}", core=False)
            for d in dimensions
            if d.strip()
        ]

    if qtype is QuestionType.SYNTHESIS:
        named_parts = tuple(p for p in parts if p.strip())
        if not named_parts:
            return unusable(
                question,
                "a multi-part question needs its parts named, or there is no way "
                "to report which were answered",
            )
        slots = [
            _slot(f"part_{i + 1}", f"An answer to: {part}") for i, part in enumerate(named_parts)
        ]

    return AnswerContract(
        question=question,
        question_type=qtype,
        entities=tuple(entities),
        dimensions=tuple(dimensions),
        constraints=tuple(constraints),
        ambiguities=tuple(ambiguities),
        usable=True,
        required_slots=tuple(slots),
    )


def type_from_output_format(output_format: str) -> QuestionType | None:
    """Map the analysis stage's existing label, without re-deriving it."""
    return _OUTPUT_FORMAT_TO_TYPE.get(output_format)
