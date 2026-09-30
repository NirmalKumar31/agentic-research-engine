"""Two functions with no production call site, and why each is kept.

A checkpoint reported four functions as "dead code with ~19 tests
pointing at them". Tracing every call site showed that was wrong about
half of them: two have real consumers in committed tooling, and three
one-line wrappers had none at all.

| function | verdict | evidence |
| --- | --- | --- |
| `relevance.assess_relevance` | retained | `examples/quality-eval/run_eval.py:137` |
| `atomicity.compound_markers` | retained | `examples/release-audit/build_audit.py:97` |
| `atomicity.looks_compound` | removed | no consumer; `bool(compound_markers(x))` |
| `atomicity.is_atomic` | removed | no consumer; `not compound_propositions(x)` |
| `propositions.is_atomic` | removed | no consumer; `len(decompose(x)) <= 1` |

The removals' tests were repointed at the functions they were proxying
for -- `compound_propositions` is what production calls, and
`compound_markers` is what the audit tool calls -- so the suite now
exercises the real dependencies rather than a wrapper around them. That
is strictly more coverage of what matters, not less.

These tests run at the integration boundary: they assert the tools
import and call these functions, and that the functions behave as those
tools rely on. A retained API with no test through its consumer is a
promise nobody checks.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


class TestAssessRelevanceServesTheEvaluationHarness:
    def test_the_harness_imports_and_calls_it(self) -> None:
        source = (ROOT / "examples/quality-eval/run_eval.py").read_text()
        assert "from agentic_research.citations.relevance import assess_relevance" in source
        assert "assess_relevance(" in source

    def test_it_composes_structure_and_judgement_as_the_harness_needs(self) -> None:
        """The harness supplies a scripted judgement and wants one
        verdict back. Production cannot use this because it narrows the
        judge's authority by slot, which this signature cannot say."""
        from agentic_research.answer_contract import QuestionType, build_contract
        from agentic_research.citations.relevance import assess_relevance

        contract = build_contract(
            "What is retrieval-augmented generation?",
            QuestionType.DEFINITION,
            entities=("retrieval-augmented generation",),
        )
        claim = "Retrieval-augmented generation retrieves passages at query time."

        yes = assess_relevance(claim, "definition", contract, model_says_relevant=True)
        assert yes.publishable

        no = assess_relevance(claim, "definition", contract, model_says_relevant=False)
        assert not no.publishable

    def test_an_unobtained_judgement_withholds(self) -> None:
        """The property the harness depends on: `None` is not a yes.
        Defaulting it to one would make the parameter decorative."""
        from agentic_research.answer_contract import QuestionType, build_contract
        from agentic_research.citations.relevance import assess_relevance

        contract = build_contract(
            "What is retrieval-augmented generation?",
            QuestionType.DEFINITION,
            entities=("retrieval-augmented generation",),
        )
        verdict = assess_relevance(
            "Retrieval-augmented generation retrieves passages at query time.",
            "definition",
            contract,
            model_says_relevant=None,
        )
        assert not verdict.publishable
        assert not verdict.checked


class TestCompoundMarkersServesTheReleaseAudit:
    def test_the_audit_imports_and_calls_it(self) -> None:
        source = (ROOT / "examples/release-audit/build_audit.py").read_text()
        assert "from agentic_research.citations.atomicity import compound_markers" in source
        assert "compound_markers(" in source

    def test_it_returns_the_markers_rather_than_a_boolean(self) -> None:
        """Why the wrapper was not worth keeping: the audit shows a
        reader *why* a claim reads as compound, and a boolean cannot."""
        from agentic_research.citations.atomicity import compound_markers

        markers = compound_markers("Pruning cut index memory by 60% while maintaining high recall.")
        assert markers
        assert all(isinstance(m, str) for m in markers)
        assert compound_markers("Pruning cut index memory by 60%.") == []

    def test_it_is_not_the_production_guard(self) -> None:
        """The distinction the docstring states, measured in both
        directions. The two disagree, so neither can stand in for the
        other -- which is why the guard is `compound_propositions` and
        this is only an explanation for a reader.
        """
        from agentic_research.citations.atomicity import (
            compound_markers,
            compound_propositions,
        )

        # The conjunction scan MISSES a compound the production guard
        # catches: no joiner from its list appears, but there are two
        # subjects with finite verbs. Using this as the guard would
        # publish a claim asserting two things against one quote.
        missed = "The index is small but the recall is high."
        assert compound_markers(missed) == []
        assert compound_propositions(missed)

        # And it flags more joiners than there are propositions, so it
        # is not merely a weaker version of the same test.
        noisier = "RAG retrieves passages, whereas fine-tuning updates weights."
        assert len(compound_markers(noisier)) > len(compound_propositions(noisier))


class TestTheRemovedWrappersAreGone:
    def test_no_module_exposes_them(self) -> None:
        """Non-vacuity for the removal: an import that still resolves
        would mean the deletion did not happen."""
        from agentic_research.citations import atomicity, propositions

        assert not hasattr(atomicity, "looks_compound")
        assert not hasattr(atomicity, "is_atomic")
        assert not hasattr(propositions, "is_atomic")

    def test_nothing_in_the_repository_calls_or_imports_them(self) -> None:
        """Calls and imports, not every mention. The docstring on
        `compound_markers` names `looks_compound` to record why it was
        removed, and a prose reference is the opposite of a live
        dependency."""
        import re

        pattern = re.compile(
            r"(?:\b(?:looks_compound|is_atomic)\s*\()"
            r"|(?:^\s*(?:from|import)\b.*\b(?:looks_compound|is_atomic)\b)",
            re.M,
        )
        offenders: list[str] = []
        for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "examples").rglob("*.py")):
            if pattern.search(path.read_text()):
                offenders.append(str(path.relative_to(ROOT)))
        assert offenders == []
