"""Deterministic overclaim guards, applied before anything may publish.

An NLI model scores whether a hypothesis follows from a premise, and it
is good at that. It is measurably not reliable on a narrow set of
transformations that matter enormously here: "may need" becoming
"typically requires", one product's measurement becoming a category
claim, a value becoming a ranking. Those are exactly the substitutions
the calibration found being approved, in every verifier design tried.

So they are checked in code instead. Each guard answers one question
with string evidence, fails the claim when it fires, and abstains
otherwise. A guard failure withholds regardless of entailment score.

These are deliberately narrow. They will miss overclaims -- that is
acceptable, because a missed guard leaves the NLI threshold as the
remaining defence, while an over-eager guard withholds true claims for
no reason. This is not a linguistic reasoning engine and must not grow
into one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agentic_research.citations.numerics import literals

# Ordered weakest to strongest. A claim may not sit in a higher band
# than its evidence.
_MODALITY_BANDS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (1, ("may", "might", "could", "can", "possibly", "sometimes")),
    (2, ("often", "typically", "generally", "usually", "commonly", "frequently")),
    (3, ("must", "required", "requires", "require", "always", "necessary",
         "mandates", "mandatory", "will", "only", "unless", "guarantees")),
)

_RANKING = (
    "highest", "lowest", "best", "worst", "most effective", "least effective",
    "optimal", "leading", "fastest", "slowest", "superior", "top", "outperforms",
    "unmatched", "premier",
)

_CAUSAL = (
    "caused", "causes", "causing", "drove", "drives", "led to", "leads to",
    "resulted in", "results in", "because of", "due to", "owing to",
)

# Association language that does not establish cause.
_ASSOCIATIVE = (
    "associated", "correlated", "linked", "related", "accompanied", "alongside",
)


@dataclass(frozen=True)
class GuardResult:
    name: str
    passed: bool
    detail: str = ""


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def _phrases_present(text: str, phrases: tuple[str, ...]) -> list[str]:
    lowered = (text or "").lower()
    return [p for p in phrases if p in lowered]


def _modality_band(text: str) -> int:
    """Strongest modality band present, 0 when none."""
    lowered = (text or "").lower()
    words = _words(lowered)
    band = 0
    for level, terms in _MODALITY_BANDS:
        for term in terms:
            hit = term in words if " " not in term else term in lowered
            if hit:
                band = max(band, level)
    return band


def numeric_guard(claim: str, evidence: str) -> GuardResult:
    """Every numeric literal in the claim must appear in the evidence.

    Literal comparison only. Equivalent values written differently
    ("100M" and "100 million") will fail this, which withholds a true
    claim -- the safe direction.
    """
    claim_values = literals(claim)
    if not claim_values:
        return GuardResult("numeric", True, "no numeric literals in claim")

    present = {v.replace(" ", "").lower() for v in literals(evidence)}
    missing = [v for v in claim_values if v.replace(" ", "").lower() not in present]
    if missing:
        return GuardResult(
            "numeric", False, f"claim states {', '.join(missing)}; not in the cited evidence"
        )
    return GuardResult("numeric", True, f"all of {', '.join(claim_values)} present")


def modality_guard(claim: str, evidence: str) -> GuardResult:
    """A claim may not be more certain than its evidence."""
    claim_band = _modality_band(claim)
    evidence_band = _modality_band(evidence)
    if claim_band == 0:
        return GuardResult("modality", True, "claim states no modality")
    if claim_band > evidence_band:
        return GuardResult(
            "modality",
            False,
            f"claim modality band {claim_band} exceeds evidence band {evidence_band}",
        )
    return GuardResult("modality", True, f"claim band {claim_band} within evidence {evidence_band}")


def ranking_guard(claim: str, evidence: str) -> GuardResult:
    """A superlative needs the evidence to make the comparison.

    A reported value is not a ranking: "Redis: 5ms" does not establish
    "Redis had the lowest latency", however low 5ms happens to be.
    """
    claimed = _phrases_present(claim, _RANKING)
    if not claimed:
        return GuardResult("ranking", True, "claim asserts no ranking")
    supported = _phrases_present(evidence, _RANKING)
    unmatched = [r for r in claimed if r not in supported]
    if unmatched:
        return GuardResult(
            "ranking", False, f"claim asserts {', '.join(unmatched)}; evidence states no ranking"
        )
    return GuardResult("ranking", True, f"evidence states {', '.join(supported)}")


def causal_guard(claim: str, evidence: str) -> GuardResult:
    """Association in the evidence must not become causation in the claim."""
    claimed = _phrases_present(claim, _CAUSAL)
    if not claimed:
        return GuardResult("causal", True, "claim asserts no causation")
    if _phrases_present(evidence, _CAUSAL):
        return GuardResult("causal", True, "evidence states causation")
    if _phrases_present(evidence, _ASSOCIATIVE):
        return GuardResult(
            "causal", False, f"claim asserts {', '.join(claimed)} from associational evidence"
        )
    return GuardResult(
        "causal", False, f"claim asserts {', '.join(claimed)}; evidence states no causal link"
    )


ALL_GUARDS = (numeric_guard, modality_guard, ranking_guard, causal_guard)


def run_guards(claim: str, evidence: str) -> list[GuardResult]:
    return [guard(claim, evidence) for guard in ALL_GUARDS]


def guards_pass(results: list[GuardResult]) -> bool:
    return all(r.passed for r in results)
