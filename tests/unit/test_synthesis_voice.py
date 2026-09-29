"""A claim asserts something about the subject, not about the evidence.

Three hosted runs produced claims of the form "The evidence describes
X" and "The architectures discussed in the study are based on Y".
Every one was refused, at entailment 0.007, 0.031 and 0.115, because
the premise is the quote and the quote does not say what the evidence
describes -- it just says the thing.

The synthesiser prompt was teaching it. Two of its three worked
examples for splitting a compound claim began "The source reports",
so the model was following the instruction it was given, and the
verifier was correctly refusing the result. One run spent two of its
three claims this way.

Measured on the pinned checkpoint against the quote "Large language
models are built on artificial neural network architectures":

    plain claim              0.998   publishes
    "The source reports X"   0.856   withheld
    "The evidence describes" 0.519   withheld

There is a real exception and it is load-bearing. Audit 2 of this
project found "We demonstrate that X" published as bare "X", which
presents one paper's result as the field's agreement, and the framing
guard exists to catch it. So the rule is not "never attribute" -- it
is carry the frame the quote has, never add one it does not.
"""

from __future__ import annotations

import re

from agentic_research.graph.prompts import SYNTHESIZER_SYSTEM


class TestThePromptDoesNotTeachTheDefect:
    @staticmethod
    def positive_examples() -> list[str]:
        """The quoted lines under each `This:`, and only those.

        Parsed rather than grepped, because the prompt deliberately
        contains meta-claims under `Not this:` and in the measurement
        table. A test that flagged those would be flagging the lesson.
        """
        out: list[str] = []
        collecting = False
        for line in SYNTHESIZER_SYSTEM.splitlines():
            stripped = line.strip()
            if stripped == "This:":
                collecting = True
                continue
            if collecting:
                if stripped.startswith('"') and line.startswith("    "):
                    out.append(stripped.strip('"'))
                else:
                    collecting = False
        return out

    def test_the_parser_finds_the_examples(self) -> None:
        """Guards the test above from passing because it read nothing."""
        examples = self.positive_examples()
        assert len(examples) >= 5, examples
        assert any("Quantization reduced memory usage" in e for e in examples)

    def test_no_worked_example_adds_a_frame(self) -> None:
        """The examples are what the model copies. A `This:` line
        beginning with a corpus reference instructs the model to
        produce a claim the verifier will refuse."""
        offenders = [
            e
            for e in self.positive_examples()
            if re.match(r"^(The source|The evidence|The study|The paper)\b", e)
        ]
        assert offenders == [], f"prompt teaches meta-claims: {offenders}"

    def test_it_states_the_rule(self) -> None:
        assert "about the SUBJECT, never about the evidence" in SYNTHESIZER_SYSTEM

    def test_it_shows_the_measured_cost(self) -> None:
        """A rule with a number behind it is one a model can weigh.
        These are real scores from the pinned checkpoint."""
        for score in ("0.998", "0.856", "0.519"):
            assert score in SYNTHESIZER_SYSTEM

    def test_the_framing_exception_survives(self) -> None:
        """The rule must not become "never attribute". Audit 2 found a
        source's own finding published as settled fact, and the framing
        guard still requires the frame when the quote has one."""
        assert "if the QUOTE itself is framed" in SYNTHESIZER_SYSTEM
        assert "keep that frame" in SYNTHESIZER_SYSTEM
        assert "never" in SYNTHESIZER_SYSTEM and "add one it does not" in SYNTHESIZER_SYSTEM

    def test_atomicity_guidance_is_untouched(self) -> None:
        """The examples were edited; the rule they illustrate was not."""
        assert "Write ATOMIC claims" in SYNTHESIZER_SYSTEM
        assert "One subject, one assertion about it" in SYNTHESIZER_SYSTEM
        assert "Two results are two claims" in SYNTHESIZER_SYSTEM


class TestTheGuardsStillDisagreeWithEachOtherNowhere:
    def test_the_framing_guard_accepts_an_unframed_claim(self) -> None:
        """The two rules have to be satisfiable at once. A claim with
        no added frame, against a quote with no frame, must pass the
        framing guard -- otherwise the prompt above would be telling
        the model to write something a guard rejects."""
        from agentic_research.citations.guards import framing_guard

        quote = "Large language models are built on artificial neural network architectures."
        claim = "Large language models are built on artificial neural network architectures."
        assert framing_guard(claim, quote).passed

    def test_the_framing_guard_still_catches_a_deleted_frame(self) -> None:
        """Non-vacuity, and the defect audit 2 found."""
        from agentic_research.citations.guards import framing_guard

        quote = "We demonstrate that quantization reduces memory usage by 75%."
        assert not framing_guard("Quantization reduces memory usage by 75%.", quote).passed


class TestTheSynthesiserIsToldWhichSlotIsRequired:
    """Three slots rendered alike read as three equally good options.

    A hosted run on v1.5.0 wrote three `dimension` claims and no
    contrast. The relevance gate refused two of them for describing
    one subject instead of contrasting them, and the report published
    nothing. The contract knew `direct_contrast` was the core slot;
    the call site flattened the slots to (name, description) pairs and
    dropped `core` on the way to the prompt.

    Sixth instance of the same shape in this project: a value computed
    and then not passed to the thing that needed it.
    """

    @staticmethod
    def prompt(contract) -> str:
        from agentic_research.graph.prompts import synthesizer_user

        return synthesizer_user(
            contract.question,
            "comparison",
            "(evidence)",
            "",
            claim_budget=8,
            answer_slots=contract.required_slots,
        )

    @staticmethod
    def comparison():
        from agentic_research.answer_contract import QuestionType, build_contract

        return build_contract(
            "How does a large language model differ from a neural network?",
            QuestionType.COMPARISON,
            entities=("large language model", "neural network"),
        )

    def test_the_core_slot_is_marked_required(self) -> None:
        assert "direct_contrast (REQUIRED)" in self.prompt(self.comparison())

    def test_the_optional_ones_are_marked_optional(self) -> None:
        """Non-vacuity: marking everything required would be the same
        failure with louder words."""
        text = self.prompt(self.comparison())
        assert "dimension (optional)" in text
        assert "relationship (optional)" in text
        assert "dimension (REQUIRED)" not in text

    def test_it_says_the_report_fails_without_it(self) -> None:
        """The run that motivated this filled optional parts three
        times over. Naming the slot is not enough; the consequence of
        omitting it has to be stated."""
        text = self.prompt(self.comparison())
        assert "has not answered the question unless a claim fills direct_contrast" in text
        assert "answers nothing" in text

    def test_the_descriptions_survive(self) -> None:
        for slot in self.comparison().required_slots:
            assert slot.description in self.prompt(self.comparison())

    def test_a_contract_with_no_core_slot_makes_no_demand(self) -> None:
        """Guards against emitting "unless a claim fills " with an
        empty name, which would read as a broken instruction."""
        from agentic_research.answer_contract import AnswerContract, AnswerSlot, QuestionType
        from agentic_research.graph.prompts import synthesizer_user

        slots = (AnswerSlot(name="only_optional", description="d", core=False),)
        contract = AnswerContract(
            question="q", question_type=QuestionType.DEFINITION, required_slots=slots
        )
        text = synthesizer_user(
            contract.question, "overview", "(e)", "", answer_slots=contract.required_slots
        )
        assert "has not answered the question unless" not in text
        assert "only_optional (optional)" in text

    def test_no_slots_at_all_renders_nothing(self) -> None:
        from agentic_research.graph.prompts import synthesizer_user

        text = synthesizer_user("q", "overview", "(e)", "", answer_slots=None)
        assert "REQUIRED" not in text


class TestTheCoreFlagSurvivesTheCallSite:
    def test_reporting_passes_slots_not_flattened_pairs(self) -> None:
        """The bug was here, not in the prompt. Flattening to
        (name, description) silently discarded `core`."""
        import inspect

        from agentic_research.graph.nodes import reporting

        body = inspect.getsource(reporting.synthesize_report)
        assert "answer_slots=(" in body
        assert "(s.name, s.description) for s in" not in body, (
            "the call site flattens the slots again and drops `core`"
        )
