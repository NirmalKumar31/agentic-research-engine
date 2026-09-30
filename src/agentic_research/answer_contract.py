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
    CAUSAL_DRIVERS = "causal_drivers"
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
        # `relationship` used to be a declared alternative here, and the
        # comment justifying it admitted the cost: for a genuine
        # comparison of two unrelated subjects, a vague relationship
        # claim discharged the core slot too. That is now decided at
        # coverage time by the *kind* of relationship rather than by its
        # slot name -- see `comparison.discharges_contrast`. Only a
        # relationship that explains why a contrast is inappropriate
        # (one subject is a subtype of, equivalent to, or a component
        # of the other) may stand in for one. "Both are used with
        # language models" may not.
        #
        # There is deliberately no static `satisfied_by` on this slot.
        # A name cannot express "only when the claim says a particular
        # kind of thing", and encoding it as one let every relationship
        # claim through.
        _slot("direct_contrast", "An explicit statement of how the subjects differ"),
        _slot("dimension", "A named dimension along which they differ", core=False),
        _slot(
            "relationship",
            "How the subjects relate, such as one being a kind of the other",
            core=False,
        ),
    ),
    QuestionType.CAUSAL: (
        # A yes/no causal test: "does X cause Y?"
        #
        # `candidate_drivers` used to discharge this slot, and the
        # comment justifying it admitted the cost: for a genuine causal
        # test, naming a plausible driver discharged the core slot
        # without establishing causation, with the relevance judgement
        # as the only backstop. That is the association-for-causation
        # substitution this project's entire verification layer exists
        # to refuse, and leaving a model as its only guard was not
        # defensible. The alternative is removed.
        #
        # The question it was protecting -- "what causes X?" -- is a
        # different shape and now has its own contract below, where the
        # drivers are the answer rather than a substitute for one.
        _slot(
            "causal_evidence",
            "Evidence capable of supporting the causal relationship, not association",
        ),
        _slot("effect_direction", "Which way the relationship runs", core=False),
        _slot("limitations", "What the evidence cannot establish about cause", core=False),
    ),
    QuestionType.CAUSAL_DRIVERS: (
        # "What causes X?", "why does X happen?", "what factors
        # contribute to X?" -- the drivers are the answer.
        #
        # `causation_limits` is deliberately non-core, which is a
        # stated deviation from "limitations about whether causation was
        # established are required". Making it core would mean a run
        # publishes nothing unless it also publishes a claim about the
        # limits of its own evidence, and no measured run has ever done
        # that -- the requirement would make every driver question
        # unanswerable, which is worse than the gap it closes. What
        # actually stops a driver being reported as a cause is
        # `causal_guard`, which refuses a claim asserting causation from
        # associational evidence, and the coverage narrative, which
        # names this slot as unestablished when it is empty.
        _slot("candidate_drivers", "Factors the evidence supports as contributing"),
        _slot("mechanism", "How a driver produces the effect", core=False),
        _slot(
            "causation_limits",
            "Whether the evidence establishes causation or only association",
            core=False,
        ),
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
    "causal_analysis": QuestionType.CAUSAL,
    "causal_drivers": QuestionType.CAUSAL_DRIVERS,
    "metric": QuestionType.NUMERIC,
    "list": QuestionType.LIST,
    "synthesis": QuestionType.SYNTHESIS,
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
    """Every concept the question names, including the setting.

    Useful for retrieval and for reporting a subject no source covers.
    Deliberately *not* the sides of a comparison: the analyst's entity
    list is described to it as technologies, organisations or concepts,
    and treating each one as a thing to be contrasted is what made a
    correctly answered comparison report itself unanswered."""
    comparison_subjects: tuple[str, ...] = ()
    """The explicit sides of a comparison; empty for every other shape.

    Read from the question's wording rather than accepted from the
    model, so a setting ("for language models") cannot become a side."""
    shape_source: str = "model"
    """How ``question_type`` was decided: ``model``, ``wording`` or
    ``corrected-from-wording``. Recorded because an override that
    leaves no trace cannot be audited."""
    dimensions: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    ambiguities: tuple[str, ...] = ()
    usable: bool = True
    unusable_reason: str = ""
    required_slots: tuple[AnswerSlot, ...] = field(default_factory=tuple)

    @property
    def context_entities(self) -> tuple[str, ...]:
        """Entities that are not sides of the comparison.

        The separation the coverage rule needs: these may steer
        retrieval and must never create an unanswered slot."""
        sides = {s.lower() for s in self.comparison_subjects}
        return tuple(e for e in self.entities if e.lower() not in sides)

    @property
    def subjects_to_span(self) -> tuple[str, ...]:
        """The subjects a published answer must speak about.

        For a comparison that is its declared sides and nothing else.
        For every other shape there is no spanning requirement, so this
        is empty rather than "all the entities" -- a definition question
        does not become unanswered because a context noun went
        unmentioned."""
        if self.question_type is QuestionType.COMPARISON:
            return self.comparison_subjects or self.entities
        return ()

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
            "comparison_subjects": list(self.comparison_subjects),
            "shape_source": self.shape_source,
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


def _validated_sides(
    comparison_subjects: list[str] | tuple[str, ...],
    entities: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """The comparison's sides: distinct, non-empty, order preserved.

    Falls back to ``entities`` only when no sides were supplied, which
    is what a caller predating the split does. Once sides are supplied
    they are authoritative -- an entity that is not a side must not be
    able to re-enter through the back door and make the comparison
    unanswerable again.
    """
    source = comparison_subjects if any(str(s).strip() for s in comparison_subjects) else entities
    out: list[str] = []
    for raw in source:
        subject = str(raw).strip()
        if not subject:
            continue
        if subject.lower() in {s.lower() for s in out}:
            continue
        out.append(subject)
    return tuple(out)


def build_contract(
    question: str,
    question_type: QuestionType | str,
    *,
    entities: list[str] | tuple[str, ...] = (),
    dimensions: list[str] | tuple[str, ...] = (),
    constraints: list[str] | tuple[str, ...] = (),
    ambiguities: list[str] | tuple[str, ...] = (),
    parts: list[str] | tuple[str, ...] = (),
    comparison_subjects: list[str] | tuple[str, ...] = (),
    shape_source: str = "model",
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
    # The sides, not every concept named. `comparison_subjects` is read
    # from the question's wording; falling back to `entities` keeps a
    # caller that predates the split working, and is the only reason
    # the fallback exists.
    sides = _validated_sides(comparison_subjects, entities)
    if qtype is QuestionType.COMPARISON and len(sides) < 2:
        return unusable(
            question,
            f"a comparison needs at least two subjects; only {len(sides)} was identified",
        )

    # Named dimensions become their own slots. The canonical set fixes
    # that a comparison needs a contrast; which dimensions matter is
    # subject matter, and comes from the question.
    if qtype is QuestionType.COMPARISON:
        # The generic `dimension` slot is kept even when named axes
        # exist, which reverses an earlier decision to drop it.
        #
        # It was dropped because a placeholder alongside real axes
        # looked redundant. It is not: a claim that contributes to the
        # comparison without sitting on a named axis needs somewhere to
        # declare itself, and with no such slot the structural
        # relevance check refuses it as an unknown slot -- so naming
        # axes would *reduce* what a comparison can publish.
        #
        # It cannot complete a contrast on its own; only a named axis
        # can, because two claims declaring the generic slot may
        # address entirely different properties. Keeping it costs
        # nothing and losing it costs claims.
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
        comparison_subjects=sides if qtype is QuestionType.COMPARISON else (),
        shape_source=shape_source,
        dimensions=tuple(dimensions),
        constraints=tuple(constraints),
        ambiguities=tuple(ambiguities),
        usable=True,
        required_slots=tuple(slots),
    )


def type_from_output_format(output_format: str) -> QuestionType | None:
    """Map the analysis stage's existing label, without re-deriving it."""
    return _OUTPUT_FORMAT_TO_TYPE.get(output_format)
