"""One bounded rewrite, for wording and nothing else.

Three of six claims in the frozen set were refused for how they were
phrased rather than for lacking support: attribution naming the wrong
actor, a figure stated without its units, a hedge dropped. Those are
real refusals -- the guards exist because a generative verifier
approved exactly those substitutions -- and they are also a waste,
because the evidence was there and only the sentence was wrong.

So a supported claim that fails a wording guard gets one rewrite.

The danger is obvious and the whole module is shaped around it. A
rewrite that may change anything can turn "teams that reviewed more
code reported fewer defects" into "code review reduces defects", and
a rewrite that may delete anything can drop the unsupported half of a
bundled claim and publish the rest. Either would be laundering.

So:

* only claims whose every proposition is already supported are
  eligible -- repair is not a second chance at evidence;
* the rewrite sees the contract, the claim, its quotes and the exact
  rule it broke, and nothing else;
* the result is checked deterministically before it is re-verified:
  no new number, no changed number, no new named subject, no
  strengthened modality, no causal language that was not there;
* every gate then runs again from the beginning;
* one attempt, and a second failure withholds with both reasons kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from agentic_research.citations.numerics import literals

# Guards whose failure is about wording. A claim failing one of these
# said something the evidence supports, badly.
REPAIRABLE_GUARDS = frozenset({"attribution", "atomicity", "numeric", "modality", "ranking"})

# Guards that are never about wording. Causation and exclusivity
# failures mean the claim asserts something the evidence does not, and
# rephrasing that is the laundering this module refuses.
UNREPAIRABLE_GUARDS = frozenset({"causal", "exclusivity", "framing", "hedge"})

_CAUSAL_WORDS = re.compile(
    r"\b(causes?|caused|causing|because of|leads? to|results? in|due to|"
    r"responsible for|drives?|produces?)\b",
    re.I,
)

# Ordered weakest to strongest. A rewrite may soften and may not harden.
_MODALITY_LADDER = (
    (re.compile(r"\b(may|might|could|can|suggests?|indicates?)\b", re.I), 1),
    (re.compile(r"\b(often|typically|generally|usually|tends? to)\b", re.I), 2),
    (re.compile(r"\b(is|are|was|were|does|do|will|shows?|demonstrates?)\b", re.I), 3),
    (re.compile(r"\b(always|never|all|every|must|proves?|guarantees?)\b", re.I), 4),
)


class Repairer(Protocol):
    """Rewrites a claim's wording, or declines."""

    def rewrite(
        self, claim_text: str, rejection_reason: str, quotes: list[str], contract_summary: str
    ) -> str | None: ...


@dataclass(frozen=True)
class RepairOutcome:
    attempted: bool
    accepted: bool
    text: str | None
    reason: str
    original_reason: str = ""


# A sentence with no modal marker asserts its claim flatly, which is
# strong. Scoring that as zero made adding a hedge look like
# strengthening, so the one rewrite that is always safe -- softening
# -- was refused.
_BARE_ASSERTION_LEVEL = 3


def _modality_level(text: str) -> int:
    return max(
        (level for pattern, level in _MODALITY_LADDER if pattern.search(text)),
        default=_BARE_ASSERTION_LEVEL,
    )


def _named_subjects(text: str) -> set[str]:
    """Capitalised or internally-capitalised words, minus sentence openers."""
    found: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        for position, token in enumerate(sentence.split()):
            word = token.strip(".,;:()[]\"'")
            if len(word) < 3:
                continue
            if any(c.isupper() for c in word[1:]) or (position > 0 and word[:1].isupper()):
                found.add(word.lower())
    return found


def validate_rewrite(original: str, rewritten: str) -> tuple[bool, str]:
    """Whether a rewrite stayed inside what a rewrite may do.

    Deterministic and checked before the claim is re-verified, so a
    rewrite that smuggled something in never reaches the gates that
    would have to notice.
    """
    if not rewritten.strip():
        return False, "the rewrite is empty"
    if rewritten.strip() == original.strip():
        return False, "the rewrite changed nothing"

    before = set(literals(original))
    after = set(literals(rewritten))
    added = after - before
    if added:
        return False, f"the rewrite introduced numbers not in the original: {sorted(added)}"

    new_subjects = _named_subjects(rewritten) - _named_subjects(original)
    if new_subjects:
        return False, f"the rewrite introduced subjects not in the original: {sorted(new_subjects)}"

    if _modality_level(rewritten) > _modality_level(original):
        return False, "the rewrite states the claim more strongly than the original"

    if _CAUSAL_WORDS.search(rewritten) and not _CAUSAL_WORDS.search(original):
        return False, "the rewrite turned an association into a cause"

    return True, "the rewrite stayed within wording"


def failed_guard_names(reason: str) -> frozenset[str]:
    """Guard names quoted in a withholding reason."""
    match = re.search(r"deterministic guard[s]?: ([a-z, ]+)", reason)
    if not match:
        return frozenset()
    return frozenset(name.strip() for name in match.group(1).split(",") if name.strip())


def is_repairable(reason: str, *, every_proposition_supported: bool) -> tuple[bool, str]:
    """Whether this refusal is about wording and nothing else.

    A claim whose evidence does not carry it is not eligible, whatever
    guard also fired. Repair is not a second chance at evidence.
    """
    if not every_proposition_supported:
        return False, "the evidence does not support every proposition"
    names = failed_guard_names(reason)
    if not names:
        return False, "the refusal was not a guard failure"
    if names & UNREPAIRABLE_GUARDS:
        blocking = sorted(names & UNREPAIRABLE_GUARDS)
        return False, f"{', '.join(blocking)} is about what the claim asserts, not how"
    if not names <= REPAIRABLE_GUARDS:
        return False, f"unknown guards: {sorted(names - REPAIRABLE_GUARDS)}"
    return True, f"wording only: {', '.join(sorted(names))}"
