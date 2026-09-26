"""Deterministic numeric-literal anchoring for the verifier.

The verifier receives intact evidence and still invents numeric
mismatches in its own reasoning: "5x" reported as ".5x", "1.0" as "1:0",
"32-bit" as "3:2-bit". One calibration case was rejected on a
discrepancy that did not exist in the input.

Telling the model to compare literals carefully did not fully stop it,
so the comparison is done here instead and handed over as a fact. This
is deliberately not a quantity-reasoning engine: it extracts literal
tokens and reports which of the claim's appear verbatim in the evidence.

It decides nothing. A matching literal does not make a claim supported,
and a missing one does not make it unsupported -- equivalent values are
routinely written differently ("100M" and "100 million"). Its only job
is to stop the model asserting a mismatch that the text does not
contain.
"""

from __future__ import annotations

import re

# One pattern per shape, ordered longest-match-first within the
# alternation so "32-bit" is not split into "32" and "bit", and "40-60%"
# stays one range rather than two numbers.
_NUMERIC = re.compile(
    r"""
    (?<![\w.])                       # not mid-token
    (?:
        \d[\d,]*\.?\d*\s*[-\u2013]\s*\d[\d,]*\.?\d*\s*%  # 40-60%, en dash too
      | \d[\d,]*\.?\d*\s*-\s*bit                      # 32-bit, 8-bit
      | \d[\d,]*\.?\d*\s*%                            # 80%
      | \d[\d,]*\.?\d*\s*x\b                          # 5x
      | \d[\d,]*\.?\d*\s*(?:ms|GB|MB|TB|KB)\+?        # 4ms, 64GB+
      | \d[\d,]*\.?\d*\s*[MBK]\b                      # 100M
      | \d[\d,]*\.\d+                                 # 1.0
      | \d[\d,]*\+                                    # 20,000+
      | \d[\d,]*                                      # 50
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _normalise(token: str) -> str:
    """Collapse internal spacing so "5 x" and "5x" compare equal.

    Case is folded for units, but nothing else is rewritten -- the point
    is to report the literal, so digits and separators are untouched.
    """
    return re.sub(r"\s+", "", token).lower()


def compare_key(token: str) -> str:
    """Key under which two written forms count as the same quantity.

    Only comma grouping is folded, because "20,000" and "20000" are the
    same number written two ways and the difference carries no meaning.
    Nothing else is equated: "100M" and "100 million" stay distinct, so
    a claim using the second form against evidence using the first is
    withheld rather than silently accepted.
    """
    return _normalise(token).replace(",", "")


def literals(text: str) -> list[str]:
    """Numeric literals in the order they appear, de-duplicated."""
    seen: dict[str, str] = {}
    for match in _NUMERIC.finditer(text or ""):
        token = match.group().strip()
        seen.setdefault(_normalise(token), token)
    return list(seen.values())


def anchor(claim: str, evidence_text: str) -> str:
    """Render the anchor block, or "" when the claim has no numerics.

    Absent from the prompt entirely for non-numeric claims, so nothing
    invites the model to look for numbers that were never at issue.
    """
    claim_literals = literals(claim)
    if not claim_literals:
        return ""

    present = {_normalise(t) for t in literals(evidence_text)}
    found = [t for t in claim_literals if _normalise(t) in present]
    missing = [t for t in claim_literals if _normalise(t) not in present]

    lines = [
        "Numeric literals in the claim: " + ", ".join(claim_literals),
        "Found verbatim in the cited evidence: " + (", ".join(found) if found else "none"),
        "Not found verbatim: " + (", ".join(missing) if missing else "none"),
    ]
    return "\n".join(lines)
