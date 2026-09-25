"""There is no generative path to a support verdict, and there must not be.

Three attempts at using the 4B instruction model as an entailment
classifier were measured against human labels and all three failed:
never supported, then 16 false positives, then 11 malformed audits out
of 30. The conclusion was architectural, not a tuning problem.

The risk now is quiet regression -- someone adding a "if the classifier
is unavailable, ask the model" branch, which would look like resilience
and would publish unverified claims. These tests make that change fail
loudly instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agentic_research.citations import nli
from agentic_research.citations.fake_nli import FakeScorer
from agentic_research.citations.semantic import verify_claim

SRC = Path(__file__).resolve().parents[2] / "src" / "agentic_research"


def test_the_generative_verifier_prompt_no_longer_exists() -> None:
    from agentic_research.graph import prompts

    assert not hasattr(prompts, "VERIFIER_SYSTEM")
    assert not hasattr(prompts, "verifier_user")


def test_the_generative_verdict_schema_no_longer_exists() -> None:
    from agentic_research import schemas

    assert not hasattr(schemas, "EntailmentOut")


def test_no_module_mentions_the_removed_verifier() -> None:
    """A grep, deliberately. Re-adding the prompt anywhere in the
    package should fail a test rather than pass review unnoticed."""
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if any(name in path.read_text() for name in ("VERIFIER_SYSTEM", "verifier_user"))
    ]
    assert not offenders, f"the generative verifier reappeared in: {offenders}"


def test_the_verification_node_calls_no_language_model() -> None:
    """``reporting.py`` still calls a model for synthesis. It must not
    call one for verification, so the verification functions are
    checked specifically rather than the file as a whole."""
    tree = ast.parse((SRC / "graph" / "nodes" / "reporting.py").read_text())
    verifying = {"_check_entailment", "_check_contradictions", "verify_citations"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef) or node.name not in verifying:
            continue
        calls = {
            ast.unparse(child.func)
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        }
        assert not [c for c in calls if "structured" in c or "router.get" in c], (
            f"{node.name} reaches a generative model: {calls}"
        )


def test_an_unavailable_classifier_withholds_rather_than_degrading() -> None:
    verdict = verify_claim(
        "Throughput rose 20%.",
        [("E1", "Throughput rose 20%.")],
        FakeScorer(fail_with="checkpoint missing"),
        support_threshold=0.98,
    )
    assert not verdict.publishable
    assert not verdict.checked


def test_the_adapter_raises_instead_of_returning_a_neutral_score() -> None:
    """A neutral score is indistinguishable from a real "no support"
    finding, so the unavailable case must be a different type of event."""
    with pytest.raises(nli.NLIUnavailable):
        nli.NLIVerifier("definitely/not-a-real-model", "main").score([("a", "b")])
