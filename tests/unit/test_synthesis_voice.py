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
