"""Whether a claim asserts one proposition or several.

This gates publication. An earlier version said it was diagnostic only
and never gated anything, and for one release that was accidentally
true -- the guard called sentence_count and this module's clause check
was reachable only from its own tests. The wiring is the property that
matters, not the helper.

Two layers defend atomicity. The synthesiser is asked for one
proposition per claim and is the first line; this is the backstop for
when it does not comply. A compound claim is the input shape that
defeats every other guard, because each of them reasons about "the
sentence that supports this claim" and a fused claim gives them two.

Conservative on purpose. It is a clause and predicate heuristic, not a
semantic parser: it will call some single-proposition sentences
compound and withhold them, and a genuinely fused single-predicate
proposition can still slip past it. The direction of error is chosen --
withholding a true claim costs a line in a report, publishing a fused
one costs the thing this system is for.
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
    """Joiners suggesting more than one proposition, in order of appearance.

    **Not the production atomicity guard.** That is
    :func:`compound_propositions`, which splits on a new subject and a
    finite verb rather than on conjunctions, and which
    ``citations/guards.py`` calls.

    Retained as an offline-analysis API with one real consumer:
    ``examples/release-audit/build_audit.py`` records the markers found
    in each judged claim, so an audit can show *why* a claim reads as
    compound even where the proposition splitter did not refuse it. A
    marker list explains that to a reader; a boolean does not, which is
    why the one-line wrapper beside this (``looks_compound``) had no
    consumer and was removed.
    """
    lowered = (text or "").lower()
    found = [m for m in _MARKERS if m in lowered]
    if _CLAUSAL.search(text or ""):
        found.append("clausal 'while'/'but'")
    return sorted(set(found), key=lowered.find)


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
    read as sentence ends. Under-counting is safe here: it hands the
    claim to the clause check below rather than resolving it on
    punctuation.
    """
    masked = text or ""
    for abbreviation in _ABBREVIATIONS:
        masked = masked.replace(abbreviation, abbreviation.replace(".", "\u2024"))
    masked = masked.strip()
    if not masked:
        return 0
    return len(_SENTENCE_BREAK.split(masked))


# --- proposition-level atomicity -------------------------------------
#
# Counting sentences was never the real test. "X increased, Y decreased"
# is one sentence and two independently falsifiable propositions, and a
# gate that verifies one quote against one claim cannot honestly call
# that atomic.
#
# What follows is a clause heuristic, not a parser. It splits on
# coordinators and asks how many of the resulting segments carry a
# predicate of their own. Two predicates means two assertions. An
# enumeration -- "precision, recall, and F1 were reported" -- puts the
# single shared predicate in one segment only, so it survives.
#
# It is wrong sometimes, and the direction of the error is chosen: when
# a sentence cannot be shown to be single-proposition it is treated as
# compound and withheld. Withholding a true claim costs a line in a
# report; publishing a fused one is the failure this whole system
# exists to prevent.

# Contrastive coordinators. These join two assertions about different
# things essentially always -- "smaller but slower", "X improved while Y
# regressed" -- so they do not need a predicate count to be suspicious.
_CONTRASTIVE = re.compile(
    r"\s+(?:but|whereas|while|however|although|though|yet)\s+|;\s+", re.IGNORECASE
)

# Coordinators that may join either clauses or list items. Which one it
# is depends on what follows, so these only split for counting.
_COORDINATORS = re.compile(r",\s+and\s+|,\s+|\s+and\s+", re.IGNORECASE)

# Auxiliaries, copulas and modals. A segment containing one of these is
# making an assertion.
_AUXILIARIES = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "has",
        "have",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
    }
)

# Present-tense verbs common in research claims. A lexicon, not a list of
# forbidden phrasings: it answers "does this fragment assert something",
# which is the question the clause heuristic needs. Past tense is handled
# by the -ed rule below, so only -s forms are enumerated.
_PRESENT_VERBS = frozenset(
    {
        "reduces",
        "increases",
        "improves",
        "provides",
        "supports",
        "requires",
        "uses",
        "shows",
        "reports",
        "achieves",
        "enables",
        "allows",
        "handles",
        "offers",
        "delivers",
        "maintains",
        "remains",
        "yields",
        "produces",
        "generates",
        "causes",
        "leads",
        "makes",
        "gives",
        "takes",
        "needs",
        "costs",
        "scales",
        "performs",
        "operates",
        "executes",
        "processes",
        "stores",
        "returns",
        "matches",
        "detects",
        "identifies",
        "measures",
        "evaluates",
        "compares",
        "ranks",
        "selects",
        "filters",
        "computes",
        "lowers",
        "raises",
        "cuts",
        "adds",
        "removes",
        "avoids",
        "prevents",
        "ensures",
        "guarantees",
        "outperforms",
        "exceeds",
        "falls",
        "rises",
        "grows",
        "shrinks",
        "varies",
        "differs",
        "depends",
        "consumes",
        "occupies",
        "introduces",
        "eliminates",
        "simplifies",
        "complicates",
        "affects",
        "impacts",
        "degrades",
        "boosts",
        "speeds",
        "slows",
    }
)


# Words ending in -ed that are not past-tense verbs. Without these,
# "at sub-10ms speed" reads as an assertion and a perfectly atomic claim
# is withheld for containing the word "speed".
_NOT_VERBS_ENDING_ED = frozenset(
    {
        "speed",
        "indeed",
        "breed",
        "creed",
        "greed",
        "tweed",
        "steed",
        "hundred",
        "sacred",
        "kindred",
        "naked",
        "wicked",
        "rugged",
        "ragged",
        "seaweed",
        "crossbreed",
        "highspeed",
        "united",
        "limited",
    }
)


# Irregular past tenses. The -ed rule cannot see these, so without them
# "Latency fell, throughput rose" reads as a single proposition -- two
# assertions and no detected predicate in either.
_IRREGULAR_PAST = frozenset(
    {
        "fell",
        "rose",
        "grew",
        "shrank",
        "shrunk",
        "took",
        "made",
        "gave",
        "went",
        "became",
        "ran",
        "held",
        "kept",
        "left",
        "lost",
        "won",
        "met",
        "saw",
        "drove",
        "brought",
        "began",
        "broke",
        "chose",
        "led",
        "spent",
        "sent",
        "built",
        "came",
        "got",
        "drew",
        "threw",
        "wrote",
        "read",
        "beat",
        "cost",
        "hit",
        "let",
        "set",
        "put",
        "cut",
        "hurt",
        "burst",
        "arose",
        "outran",
        "overtook",
        "underwent",
    }
)


def _is_predicate(token: str) -> bool:
    """Whether a token could be the verb of its own assertion."""
    word = token.strip(".,;:!?()[]\"'").lower()
    if word in _AUXILIARIES or word in _PRESENT_VERBS or word in _IRREGULAR_PAST:
        return True
    # Past tense and participles. Length-bounded so "bed" and "red" do
    # not qualify, and filtered so nouns like "speed" do not either.
    if word in _NOT_VERBS_ENDING_ED:
        return False
    return len(word) > 4 and word.endswith("ed")


# A closed class, so listing it is safe in a way listing nouns or
# adjectives would not be. A token sitting directly after one of these
# is that preposition's object: "through stores", "with reads", "for
# writes". English does not put a finite verb there.
_PREPOSITIONS = frozenset(
    {
        "about",
        "above",
        "across",
        "after",
        "against",
        "along",
        "among",
        "around",
        "as",
        "at",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "between",
        "beyond",
        "by",
        "down",
        "during",
        "except",
        "for",
        "from",
        "in",
        "inside",
        "into",
        "near",
        "of",
        "off",
        "on",
        "onto",
        "outside",
        "over",
        "past",
        "per",
        "since",
        "through",
        "throughout",
        "to",
        "toward",
        "towards",
        "under",
        "underneath",
        "until",
        "up",
        "upon",
        "via",
        "with",
        "within",
        "without",
    }
)


def _carries_predicate(segment: str) -> bool:
    """Whether a coordinated segment asserts something of its own.

    Many English words are both a plural noun and a third-person verb --
    "stores", "reads", "writes", "requires". Judging a segment by
    whether *any* token looks like a predicate therefore read noun
    phrases as clauses, and the atomicity guard refused claims that were
    never compound. One live run lost a true, officially-sourced claim
    this way: "gives agents short-term memory through checkpointers and
    long-term memory through stores" was split at "and", and "stores"
    in the second half -- the object of "through" -- was counted as its
    verb.

    So a candidate directly preceded by a preposition does not count.
    The guard still fires on genuinely coordinated clauses, which carry
    their predicate after a subject rather than after a preposition.
    """
    tokens = segment.split()
    previous = ""
    for token in tokens:
        bare = previous.strip(".,;:!?()[]\"'").lower()
        if _is_predicate(token) and bare not in _PREPOSITIONS:
            return True
        previous = token
    return False


def compound_propositions(text: str) -> list[str]:
    """Reasons this claim looks like more than one proposition.

    Empty when it reads as a single assertion.
    """
    if not (text or "").strip():
        return []

    reasons: list[str] = []
    contrast = _CONTRASTIVE.search(text)
    if contrast:
        reasons.append(f"contrastive coordinator '{contrast.group(0).strip()}'")

    segments = [s for s in _COORDINATORS.split(text) if s.strip()]
    asserting = [s for s in segments if _carries_predicate(s)]
    if len(asserting) > 1:
        reasons.append(f"{len(asserting)} coordinated clauses each with their own predicate")

    if sentence_count(text) > 1:
        reasons.append(f"{sentence_count(text)} sentences")
    return reasons
