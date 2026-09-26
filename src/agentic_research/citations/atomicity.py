"""A conservative check for compound claims. Diagnostic only.

A compound claim is the input shape that defeats the semantic verifier.
Three propositions fused into one sentence cannot be carried by any
single quote, so either the claim is withheld despite its parts being
evidenced, or -- the dangerous case -- one quote scores high enough on
the sentence as a whole and an unevidenced clause rides along with it.
The single development-set false positive is exactly that: "reduces
memory by 75%" and "high accuracy" are both in the quote, and "high
recall accuracy" is not.

The real fix is in the synthesiser prompt, which asks for one
proposition per claim. This module only measures whether that worked.
It never gates publication: a heuristic that withheld claims would be
deciding publication questions on sentence shape, which is precisely
the reasoning this project replaced with a classifier.

Deliberately shallow. It looks for the joiners the prompt names and
does not parse anything. False positives here cost a line in an audit
report and nothing else.
"""

from __future__ import annotations

import re

# Joiners that fuse propositions needing different evidence. "and" is
# absent: it joins noun phrases far more often than clauses, and
# flagging every "Govern, Map, Measure and Manage" would make the
# signal useless.
_MARKERS = (
    "while maintaining",
    "while preserving",
    "while retaining",
    "while achieving",
    "without sacrificing",
    "without compromising",
    "thereby",
    "as well as",
    "which also",
    "whereas",
    "while also",
)

# "X, while Y" with a comma is nearly always two clauses.
_CLAUSAL = re.compile(r",\s+(?:while|whereas|but|although|though)\s+", re.IGNORECASE)


def compound_markers(text: str) -> list[str]:
    """Joiners suggesting more than one proposition, in order of appearance."""
    lowered = (text or "").lower()
    found = [m for m in _MARKERS if m in lowered]
    if _CLAUSAL.search(text or ""):
        found.append("clausal 'while'/'but'")
    return sorted(set(found), key=lowered.find)


def looks_compound(text: str) -> bool:
    return bool(compound_markers(text))


# Abbreviations whose full stop does not end a sentence. Without these a
# claim citing "e.g." or "U.S." would read as two.
_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "cf.",
    "approx.",
    "Fig.",
    "No.",
    "Dr.",
    "Mr.",
    "Ms.",
    "Inc.",
    "Ltd.",
    "U.S.",
    "U.K.",
    "Q1.",
    "Q2.",
    "Q3.",
    "Q4.",
)

_SENTENCE_BREAK = re.compile(r"[.!?]['\")\]]*\s+(?=[A-Z0-9])")


def sentence_count(text: str) -> int:
    """Sentences in a claim, counted conservatively.

    Known abbreviations are masked first so "e.g." and "U.S." do not
    read as sentence ends. Under-counting is the safe direction here:
    it lets a claim through to the other guards rather than withholding
    it on a punctuation artefact.
    """
    masked = text or ""
    for abbreviation in _ABBREVIATIONS:
        masked = masked.replace(abbreviation, abbreviation.replace(".", "\u2024"))
    masked = masked.strip()
    if not masked:
        return 0
    return len(_SENTENCE_BREAK.split(masked))


def is_atomic(text: str) -> bool:
    """One sentence, one proposition -- the shape the gate can verify.

    A claim spanning two sentences cannot be carried by a single quote
    and cannot be scoped to the sentence that supports it, so every
    guard that reasons about "the supporting sentence" silently picks
    one half and ignores the other. The release audit published exactly
    that: two propositions fused, where the guards examined the half
    that was clean and the other half had deleted the source's own
    voice.

    Enforced structurally rather than by listing conjunctions, because
    the failure is the shape, not the vocabulary.
    """
    return sentence_count(text) <= 1
