"""Splitting a claim into the things it independently asserts.

A claim bundling a measured figure with an assertion its quote never
contained was published, at entailment 0.983, because the sentence as
a whole was close enough to the quote as a whole. Verifying the
sentence verified the average of its parts, and the unsupported half
rode in on the supported one.

So support is checked per proposition, and one unsupported
proposition withholds the whole claim.

**This is not a conjunction splitter.** "A and B", a comma, or several
verbs do not mean several assertions, and treating them that way would
shred a procedure into steps and a definition into fragments -- then
fail each fragment for lacking evidence the original never needed.
Three fixtures exist to catch exactly that.

The signal used instead is a *new subject*. A clause that introduces
its own subject and its own finite verb is making its own claim:

    developers took 19% longer, and the authors note it contradicts
    ^^^^^^^^^^                      ^^^^^^^^^^^

A clause that continues the same subject is not, however many verbs
it has:

    a vector database stores embeddings and retrieves them by search
    ^^^^^^^^^^^^^^^^^

Sequential steps of one process share an actor and stay together.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Coordinators that can join two independent clauses. A split is only
# considered at one of these, and only then if what follows has its
# own subject.
_COORDINATORS = (
    ", and ",
    ", but ",
    ", while ",
    ", whereas ",
    "; and ",
    "; but ",
    "; ",
    # An explanation is its own assertion, and the causal link is a
    # third. A supported figure must not carry an unsupported reason
    # into print on its back.
    ", because ",
    ", since ",
    ", which means ",
    ", which shows ",
    ", so ",
)

# Openers that mark a following clause as a separate assertion about
# someone else, rather than a continuation.
_ATTRIBUTION_OPENERS = re.compile(
    r"^(the |a |an )?(authors?|researchers?|study|studies|paper|report|"
    r"analysts?|company|team|spokesperson)\b",
    re.I,
)

# Words that introduce a *reason* or *consequence* asserted separately
# from the fact before them.
_EXPLANATORY = re.compile(
    r"^(because|since|which (means|shows|suggests|implies)|so that|therefore|"
    r"as a result)\b",
    re.I,
)

# A pronoun subject continues the previous subject rather than
# introducing a new one.
_PRONOUNS = {"it", "they", "this", "that", "these", "those", "which", "who"}

# A clause needs a verb. Without one it is a list item -- "scale,
# decoder-only architecture, and a self-supervised next-token
# objective" is three things one claim says, not three claims.
_FINITE_VERBS = frozenset(
    [
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
        "take",
        "takes",
        "took",
        "taken",
        "make",
        "makes",
        "made",
        "show",
        "shows",
        "showed",
        "shown",
        "find",
        "finds",
        "found",
        "note",
        "notes",
        "noted",
        "report",
        "reports",
        "reported",
        "reveal",
        "reveals",
        "revealed",
        "reduce",
        "reduces",
        "reduced",
        "increase",
        "increases",
        "increased",
        "differ",
        "differs",
        "differed",
        "use",
        "uses",
        "used",
        "require",
        "requires",
        "required",
        "contain",
        "contains",
        "contained",
        "state",
        "states",
        "stated",
        "say",
        "says",
        "said",
        "mean",
        "means",
        "meant",
        "compute",
        "computes",
        "computed",
        "store",
        "stores",
        "stored",
        "retrieve",
        "retrieves",
        "retrieved",
        "publish",
        "publishes",
        "published",
        "sustain",
        "sustains",
        "sustained",
        "complete",
        "completes",
        "completed",
        "contradict",
        "contradicts",
        "contradicted",
        "grow",
        "grows",
        "grew",
        "depend",
        "depends",
        "depended",
    ]
)


def _has_finite_verb(clause: str) -> bool:
    """Whether the clause predicates anything.

    Hyphenated words are skipped before the suffix test: a compound
    adjective like "self-supervised" ends in -ed and is not a verb.
    """
    tokens = [t.strip(".,;:()").lower() for t in clause.split()]
    words = [t for t in tokens if t and "-" not in t]
    if any(w in _FINITE_VERBS for w in words):
        return True
    # A past-tense or third-person form anywhere but the first slot.
    return any(len(w) > 3 and (w.endswith("ed") or w.endswith("es")) for w in words[1:])


@dataclass(frozen=True)
class Proposition:
    """One independently falsifiable assertion."""

    text: str
    index: int


def _has_own_subject(clause: str) -> bool:
    """Whether a clause introduces a subject of its own.

    A pronoun does not: it points back at the subject already there.
    A gerund does not: "updating clients" is a step, not an actor.
    """
    words = clause.strip().split()
    if not words:
        return False
    if _EXPLANATORY.match(clause):
        return True
    if _ATTRIBUTION_OPENERS.match(clause):
        return True
    if not _has_finite_verb(clause):
        # No recognised verb. A short fragment is a list item -- "a
        # self-supervised next-token objective" is one of three things
        # a claim says, not a claim. A long one is more likely a
        # clause whose verb this lexicon does not know, and the safe
        # reading there is to split: a missed split lets an
        # unsupported half ride in on a supported one, which is the
        # defect this module exists for.
        return len(words) >= 6
    first = words[0].lower().strip(",;")
    if first in _PRONOUNS:
        return False
    if first.endswith("ing"):
        # "updating clients to use it" -- a step in a process.
        return False
    # A determiner or a capitalised noun followed by a finite verb
    # looks like a fresh subject.
    if first in {"the", "a", "an"} and len(words) > 2:
        return True
    return bool(re.match(r"^[A-Z][a-z]", words[0])) and len(words) > 2


def decompose(claim_text: str) -> list[Proposition]:
    """The independently falsifiable assertions in a claim.

    Returns one proposition for a claim that makes one, which is the
    common case and the one that must not be broken.
    """
    text = claim_text.strip()
    if not text:
        return []

    parts = [text]
    for coordinator in _COORDINATORS:
        expanded: list[str] = []
        for part in parts:
            pieces = part.split(coordinator)
            if len(pieces) == 1:
                expanded.append(part)
                continue
            current = pieces[0]
            for piece in pieces[1:]:
                if _has_own_subject(piece):
                    expanded.append(current.strip(" ,;"))
                    current = piece
                else:
                    # Same subject continuing: keep them together.
                    current = f"{current}{coordinator}{piece}"
            expanded.append(current.strip(" ,;"))
        parts = expanded

    cleaned = [p.strip(" ,;.") for p in parts if p.strip(" ,;.")]
    return [Proposition(text=p, index=i) for i, p in enumerate(cleaned)]


def is_atomic(claim_text: str) -> bool:
    """Whether the claim asserts exactly one thing."""
    return len(decompose(claim_text)) <= 1
