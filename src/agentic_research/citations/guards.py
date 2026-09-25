"""Deterministic overclaim guards, applied before anything may publish.

An NLI model scores whether a hypothesis follows from a premise, and it
is good at that. It is measurably unreliable on a narrow set of
transformations that matter enormously here: "may need" becoming
"typically requires", one product's measurement becoming a category
claim, a reported value becoming a ranking. Those are exactly the
substitutions the calibration found being approved, by every generative
verifier design tried.

So they are checked in code instead. Each guard answers one question
from string evidence, fails the claim when it fires, and abstains
otherwise. A guard failure withholds regardless of entailment score.

These are deliberately narrow. They will miss overclaims -- acceptable,
because a missed guard leaves the NLI threshold as the remaining
defence, while an over-eager guard withholds true claims for no reason.
This is not a linguistic reasoning engine and must not grow into one.

One known weakness, stated rather than engineered around: the evidence
side is scanned as a bag of words, so a long premise mentioning "must"
about some unrelated matter satisfies the modality guard for a claim
about something else. Premises here are single evidence quotes rather
than whole pages, which bounds the dilution but does not remove it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agentic_research.citations.numerics import compare_key, literals

# Modality bands, ordered weakest to strongest. A claim may not sit in a
# higher band than its evidence.
_WEAK = ("may", "might", "could", "can", "possibly", "potentially", "sometimes")
_TENDENCY = ("often", "typically", "generally", "usually", "commonly", "frequently", "tends")
_STRONG = (
    "must",
    "required",
    "require",
    "requires",
    "always",
    "necessary",
    "mandates",
    "mandate",
    "mandatory",
    "will",
    "guarantees",
    "ensures",
    "essential",
    "never",
)
_BANDS: tuple[tuple[int, tuple[str, ...]], ...] = ((1, _WEAK), (2, _TENDENCY), (3, _STRONG))

# "may require" is a hedged necessity, not a necessity: the hedge scopes
# over the strong verb. Without this the guard fails a claim that copies
# its evidence exactly.
_HEDGED = re.compile(
    r"\b(" + "|".join(_WEAK) + r")\b((?:\s+\w+){0,2}?)\s+(?:be\s+)?\b(" + "|".join(_STRONG) + r")\b"
)

# Ranking language. Superlatives are single words; ranking verbs carry a
# preposition or object, because the bare verbs are polysemous -- "leads
# to" is causal, not comparative, and matching "leads" alone would read
# a ranking into evidence that states none, which is the unsafe
# direction for this guard.
_RANKING = (
    "highest",
    "lowest",
    "best",
    "worst",
    "most effective",
    "least effective",
    "optimal",
    "leading",
    "fastest",
    "slowest",
    "superior",
    "outperforms",
    "outranks",
    "unmatched",
    "premier",
    "the most",
    "the least",
    "wins on",
    "wins in",
    "leads for",
    "leads in",
    "leads on",
    "leads the",
    "ahead of",
    "top choice",
    "best choice",
    "number one",
    "first place",
    "ranks first",
    "beats",
)

_CAUSAL = (
    "caused",
    "causes",
    "causing",
    "drove",
    "drives",
    "led to",
    "leads to",
    "resulted in",
    "results in",
    "because of",
    "due to",
    "owing to",
    "responsible for",
    "triggered",
)

# Association language: real evidence, but not of cause.
_ASSOCIATIVE = (
    "associated",
    "correlated",
    "linked",
    "related",
    "accompanied",
    "alongside",
    "observed with",
    "coincided",
)

_EXCLUSIVE = ("only", "exclusively", "solely", "unless", "no other", "nothing else")


@dataclass(frozen=True)
class GuardResult:
    name: str
    passed: bool
    detail: str = ""


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def _phrases(text: str, phrases: tuple[str, ...]) -> list[str]:
    lowered = (text or "").lower()
    words = _words(lowered)
    return [p for p in phrases if (p in lowered if " " in p else p in words)]


def modality_band(text: str) -> int:
    """Strongest unhedged modality band in the text, 0 when none.

    Hedged necessities are collapsed to their hedge first, so "may
    require" reports band 1 and "typically requires" reports band 3.
    """
    lowered = _HEDGED.sub(r"\1\2", (text or "").lower())
    band = 0
    for level, terms in _BANDS:
        if _phrases(lowered, terms):
            band = max(band, level)
    return band


def numeric_guard(claim: str, evidence: str) -> GuardResult:
    """Every numeric literal in the claim must appear in the evidence.

    Literal comparison, with comma grouping folded. Values written
    differently ("100M" against "100 million") fail this and withhold a
    true claim -- the safe direction.
    """
    claimed = literals(claim)
    if not claimed:
        return GuardResult("numeric", True, "claim states no numeric literal")

    present = {compare_key(t) for t in literals(evidence)}
    missing = [t for t in claimed if compare_key(t) not in present]
    if missing:
        return GuardResult(
            "numeric", False, f"claim states {', '.join(missing)}, absent from the cited evidence"
        )
    return GuardResult("numeric", True, f"evidence states {', '.join(claimed)}")


def modality_guard(claim: str, evidence: str) -> GuardResult:
    """A claim may not be more certain than its evidence."""
    claim_band = modality_band(claim)
    evidence_band = modality_band(evidence)
    if claim_band == 0:
        return GuardResult("modality", True, "claim asserts no modality")
    if claim_band > evidence_band:
        return GuardResult(
            "modality",
            False,
            f"claim modality band {claim_band} exceeds evidence band {evidence_band}",
        )
    return GuardResult(
        "modality", True, f"claim band {claim_band} within evidence band {evidence_band}"
    )


def ranking_guard(claim: str, evidence: str) -> GuardResult:
    """A superlative needs the evidence to make some comparison.

    A reported value is not a ranking: "Redis: 5ms" does not establish
    "Redis had the lowest latency", however low 5ms happens to be.

    This asks only whether the evidence ranks anything at all, not
    whether it ranks the same thing the claim does. An earlier version
    required the claim's exact ranking word to reappear in the quote,
    which blocked "achieves the highest throughput" against evidence
    reading "wins on raw throughput" -- the same ranking in different
    words. Lexical identity is the wrong test for a semantic question,
    so the guard supplies the cheap necessary condition and leaves "is
    it the same ranking?" to the classifier, which is what it is for.
    """
    claimed = _phrases(claim, _RANKING)
    if not claimed:
        return GuardResult("ranking", True, "claim asserts no ranking")
    supported = _phrases(evidence, _RANKING)
    if not supported:
        return GuardResult(
            "ranking", False, f"claim asserts '{', '.join(claimed)}'; evidence ranks nothing"
        )
    return GuardResult("ranking", True, f"evidence ranks: '{', '.join(supported)}'")


def causal_guard(claim: str, evidence: str) -> GuardResult:
    """Association in the evidence must not become causation in the claim."""
    claimed = _phrases(claim, _CAUSAL)
    if not claimed:
        return GuardResult("causal", True, "claim asserts no causation")
    if _phrases(evidence, _CAUSAL):
        return GuardResult("causal", True, "evidence states causation")
    if _phrases(evidence, _ASSOCIATIVE):
        return GuardResult(
            "causal", False, f"claim asserts '{', '.join(claimed)}' from associational evidence"
        )
    return GuardResult(
        "causal", False, f"claim asserts '{', '.join(claimed)}'; evidence states no causal link"
    )


def exclusivity_guard(claim: str, evidence: str) -> GuardResult:
    """Evidence that something helps does not establish that only it helps."""
    claimed = _phrases(claim, _EXCLUSIVE)
    if not claimed:
        return GuardResult("exclusivity", True, "claim asserts no exclusivity")
    supported = _phrases(evidence, _EXCLUSIVE)
    unmatched = [t for t in claimed if t not in supported]
    if unmatched:
        return GuardResult(
            "exclusivity", False, f"claim asserts '{', '.join(unmatched)}'; evidence does not"
        )
    return GuardResult("exclusivity", True, f"evidence states '{', '.join(supported)}'")


ALL_GUARDS = (numeric_guard, modality_guard, ranking_guard, causal_guard, exclusivity_guard)
GUARD_NAMES = tuple(g(" ", " ").name for g in ALL_GUARDS)


def run_guards(claim: str, evidence: str) -> list[GuardResult]:
    """Every guard, always, so the audit record shows what each decided."""
    return [guard(claim, evidence) for guard in ALL_GUARDS]


def guards_pass(results: list[GuardResult]) -> bool:
    return all(r.passed for r in results)


def failed_guards(results: list[GuardResult]) -> list[GuardResult]:
    return [r for r in results if not r.passed]
