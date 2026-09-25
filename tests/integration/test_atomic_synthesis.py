"""Does the synthesiser actually emit atomic claims?

Marked ``ollama``: this runs the real local synthesiser against the real
prompt. A prompt change is only a hypothesis until a model has been
asked, so this is the check that the Phase C wording did anything.

The assertions are structural and deliberately loose. A 4B model will
word things unpredictably, and this must fail on fused propositions and
invented relationships, not on phrasing. Generic synthetic evidence
throughout; no development calibration text appears here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_research.citations.atomicity import compound_markers
from agentic_research.config import LLMMode, Settings
from agentic_research.graph.prompts import SYNTHESIZER_SYSTEM, synthesizer_user
from agentic_research.llm.base import UsageTracker
from agentic_research.llm.router import ModelRole, ModelRouter
from agentic_research.schemas import ReportOut

pytestmark = pytest.mark.ollama

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "atomic_synthesis.json"


def _cases() -> list[dict]:
    return json.loads(FIXTURE.read_text())["cases"]


@pytest.fixture
def synthesiser():
    settings = Settings(llm_mode=LLMMode.LOCAL, persist_runs=False)
    router = ModelRouter(settings, UsageTracker(50, 0.0, max_provider_requests=50))
    return router.get(ModelRole.SYNTHESIZER)


def _all_claims(report: ReportOut) -> list[str]:
    texts = [c.text for c in report.summary_claims] + [c.text for c in report.key_findings]
    for section in report.sections:
        texts.extend(c.text for c in section.claims)
    return texts


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["id"])
async def test_synthesiser_keeps_propositions_separate(case: dict, synthesiser) -> None:
    block = "\n".join(f"[{e['id']}] {e['quote']}" for e in case["evidence"])
    report = await synthesiser.structured(
        ReportOut,
        SYNTHESIZER_SYSTEM,
        synthesizer_user(case["question"], "overview", block, "", claim_budget=6),
    )
    claims = _all_claims(report)
    assert claims, "the synthesiser produced no claims"
    joined = " ".join(claims).lower()

    banned = [p for p in case.get("forbidden_phrases", []) if p.lower() in joined]
    assert not banned, f"asserted more than the evidence stated: {banned}\nclaims: {claims}"

    for required in case.get("must_preserve", []):
        assert required.lower() in joined, f"dropped the qualifier {required!r}\nclaims: {claims}"

    fuse = case.get("must_not_fuse")
    if fuse:
        together = [c for c in claims if all(part.lower() in c.lower() for part in fuse)]
        assert not together, f"fused {fuse} into one claim: {together}"

    fused = [(c, compound_markers(c)) for c in claims if compound_markers(c)]
    assert not fused, f"compound claims emitted: {fused}"
