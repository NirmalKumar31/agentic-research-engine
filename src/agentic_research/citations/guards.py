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
# Epistemic hedges: uncertainty about whether something is so. Deleting
# one is an overclaim -- "might lack" becoming "lack" asserts as fact
# what the source declined to.
#
# "can" and "could" are deliberately absent. They usually express
# capability, not doubt: "vector databases can handle large datasets"
# becoming "handle large datasets" is faithful. Treating every modal as
# epistemic was measured against the release audit and rejected a true
# claim for exactly that reason.
_EPISTEMIC = ("may", "might", "possibly", "potentially", "perhaps", "maybe", "presumably")

_SENTENCE = re.compile(r"(?<=[.!?;])\s+")

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
class SourceIdentity:
    """Who published the quote, for the attribution guard only.

    Deliberately just identity. Quality score, search rank and source
    category are absent because they are not evidence of who said
    something, and because none of them may influence entailment.
    """

    domain: str = ""
    title: str = ""


@dataclass(frozen=True)
class GuardResult:
    name: str
    passed: bool
    detail: str = ""


# Function words carry no topic signal, so they are excluded when
# matching a claim to the sentence that supports it.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "which",
        "while",
        "but",
        "not",
        "no",
        "can",
        "may",
    }
)


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


def _epistemic_terms(text: str) -> list[str]:
    words = _words(text)
    return [w for w in _EPISTEMIC if w in words]


def _supporting_sentence(claim: str, evidence: str) -> str:
    """The sentence in the evidence that actually carries the claim.

    A quote can hold several unrelated sentences. "System A may fail
    under load. System B uses AES-256." must not block the claim
    "System B uses AES-256" because the word "may" appears somewhere
    else in the passage.

    Picked by content-word overlap -- the smallest thing that works.
    No parsing, and on a single-sentence quote it is the identity.
    """
    sentences = [part for part in _SENTENCE.split(evidence or "") if part.strip()]
    if len(sentences) <= 1:
        return evidence or ""
    claim_words = _words(claim) - _STOPWORDS
    if not claim_words:
        return evidence or ""
    return max(sentences, key=lambda part: len(claim_words & (_words(part) - _STOPWORDS)))


def hedge_guard(claim: str, evidence: str) -> GuardResult:
    """A claim may not delete the uncertainty its evidence expressed.

    The band-based modality guard catches strengthening -- "may" to
    "must" -- but cannot catch deletion, because its rule is "claim band
    must not exceed evidence band" and a claim with no modality sits in
    band 0, the weakest. So "may" to "must" fails while "may" to nothing
    passes, and deletion is the more common overclaim of the two. The
    release audit published exactly one unsupported claim and this was
    it, at 0.9946 entailment, so the classifier does not catch it
    either.

    Scoped to the sentence that supports the claim rather than the whole
    quote, so an unrelated hedge elsewhere in the passage does not
    withhold a faithful claim.
    """
    supporting = _supporting_sentence(claim, evidence)
    hedges = _epistemic_terms(supporting)
    if not hedges:
        return GuardResult("hedge", True, "evidence states no epistemic hedge")
    if _epistemic_terms(claim):
        return GuardResult("hedge", True, f"claim keeps the hedge: {', '.join(hedges)}")
    return GuardResult(
        "hedge",
        False,
        f"evidence hedges with '{', '.join(hedges)}'; the claim states it as fact",
    )


# A source writing in its own research voice is reporting its own
# finding, not stating a settled fact. "We demonstrate that X" and "X"
# are different assertions, and the difference is exactly what a reader
# needs to judge the claim.
_RESEARCH_VOICE = re.compile(
    r"\b(?:we\s+(?:demonstrate|show|find|found|propose|argue|observe|conclude|"
    r"present|report|introduce)|our\s+(?:results?|findings?|experiments?|analysis|"
    r"study|work|approach)|this\s+(?:paper|study|work|article|report)\s+"
    r"(?:demonstrates?|shows?|finds?|proposes?|argues?|presents?|reports?|concludes?))\b",
    re.IGNORECASE,
)

# Ways a claim can keep that framing.
_KEEPS_FRAMING = re.compile(
    r"\b(?:the\s+(?:authors?|study|paper|research|report|work|survey|analysis)|"
    r"researchers|according to|reportedly|is reported|are reported|was reported|"
    r"were reported|reports? that|found that|suggests? that|argues? that)\b",
    re.IGNORECASE,
)


def framing_guard(claim: str, evidence: str) -> GuardResult:
    """A source's own finding must not be published as settled fact.

    "We demonstrate that X" and a bare "X" are different assertions. The
    first is one paper reporting a result; the second is the field
    agreeing. Deleting the frame is the same transformation as dropping
    "according to the vendor's documentation", which this system already
    refuses -- it was only missed because the frame is first-person
    rather than a named attribution.

    The claim satisfies this by keeping any attribution at all: naming
    the study, the authors, or the publisher. It does not have to
    reproduce the source's wording.

    Scoped to the supporting sentence, so a methods sentence elsewhere in
    a long quote does not withhold an unrelated factual claim.
    """
    supporting = _supporting_sentence(claim, evidence)
    match = _RESEARCH_VOICE.search(supporting)
    if match is None:
        return GuardResult("framing", True, "evidence states no first-person finding")
    if _KEEPS_FRAMING.search(claim) or attributed_entities(claim):
        return GuardResult("framing", True, "claim keeps the attribution")
    return GuardResult(
        "framing",
        False,
        f"evidence frames this as its own finding ('{match.group(0)}'); "
        f"the claim states it as settled fact",
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


# Verbs that attribute a proposition to whoever precedes them.
_ATTRIBUTION_VERBS = (
    "states",
    "stated",
    "says",
    "said",
    "reports",
    "reported",
    "requires",
    "require",
    "required",
    "mandates",
    "mandated",
    "recommends",
    "recommended",
    "found",
    "finds",
    "notes",
    "noted",
    "writes",
    "wrote",
    "concludes",
    "concluded",
    "warns",
    "warned",
    "defines",
    "defined",
    "specifies",
    "specified",
    "advises",
    "advised",
    "prohibits",
    "prohibited",
)

# "the authors", "the study" and friends point at the cited source
# itself, so citing it establishes them. Only named third parties need
# checking.
_SELF_REFERENCE = re.compile(
    r"\bthe\s+(?:authors?|study|paper|report|research|guide|documentation|source|"
    r"specification|standard|framework|article|survey|benchmark)\b",
    re.IGNORECASE,
)

# An acronym (NIST, IEEE, WHO) or a capitalised multi-word name, taken
# only where an attribution structure makes the role unambiguous.
_NAME = r"(?:[A-Z][A-Za-z0-9.&-]*)(?:\s+[A-Z][A-Za-z0-9.&-]*){0,3}"
_ACRONYM_ATTRIBUTION = re.compile(
    rf"\b([A-Z]{{2,}}[A-Za-z0-9.&-]*)\s+(?:{'|'.join(_ATTRIBUTION_VERBS)})\b"
)
_EXPLICIT_ATTRIBUTION = re.compile(
    rf"\b(?:according to|per|cited by|as stated by|as reported by)\s+({_NAME})", re.IGNORECASE
)
_MIDSENTENCE_ATTRIBUTION = re.compile(
    rf"(?<!^)(?<![.!?]\s)\b({_NAME})\s+(?:{'|'.join(_ATTRIBUTION_VERBS)})\b"
)

_GENERIC_SUBJECTS = frozenset(
    {
        "it",
        "they",
        "this",
        "that",
        "these",
        "those",
        "he",
        "she",
        "we",
        "you",
        "the",
        "a",
        "an",
        "organizations",
        "organisations",
        "users",
        "developers",
        "companies",
        "teams",
        "systems",
    }
)


def attributed_entities(claim: str) -> list[str]:
    """Named third parties the claim credits a proposition to.

    Deliberately narrow. It fires on "according to X", on an acronym
    followed by an attribution verb, and on a capitalised name in the
    middle of a sentence followed by one. It does not fire on a
    sentence-initial capitalised common noun, because "Vector databases
    require significant memory" is a claim about vector databases and
    not an attribution to anyone.

    Missing an attribution means the guard abstains and the classifier
    decides, which is the same position the system was in before.
    Inventing one would withhold ordinary claims, so the bias is toward
    silence.
    """
    text = claim or ""
    found: list[str] = []
    for pattern in (_EXPLICIT_ATTRIBUTION, _ACRONYM_ATTRIBUTION, _MIDSENTENCE_ATTRIBUTION):
        for match in pattern.finditer(text):
            name = match.group(1).strip(" .,")
            if not name or name.lower() in _GENERIC_SUBJECTS:
                continue
            if _SELF_REFERENCE.fullmatch(name) or name.lower().startswith("the "):
                continue
            if name not in found:
                found.append(name)
    return found


def _establishes(name: str, quote: str, source: SourceIdentity | None) -> str:
    """Where, if anywhere, this attribution is borne out."""
    needle = name.lower()
    if needle in (quote or "").lower():
        return "the quoted text names it"
    if source is not None:
        # Domains drop punctuation: "nist.gov" must match "NIST",
        # "ieee.org" must match "IEEE".
        domain = re.sub(r"[^a-z0-9]", "", (source.domain or "").lower())
        compact = re.sub(r"[^a-z0-9]", "", needle)
        if compact and compact in domain:
            return f"the source domain is {source.domain}"
        if needle in (source.title or "").lower():
            return "the source title names it"
    return ""


def attribution_guard(
    claim: str, evidence: str, source: SourceIdentity | None = None
) -> GuardResult:
    """A claim crediting a named party must have that party established.

    The old generative verifier was shown the publisher alongside the
    quote, so it could tell "NIST requires X" backed by nist.gov from
    the same sentence backed by a vendor blog paraphrasing NIST. The NLI
    verifier sees only the quote, by design -- source reputation must
    not influence entailment. That left attribution unchecked, so it is
    checked here instead, from identity alone and never from quality
    score, rank or source category.

    An attribution is established when the quote itself names the party,
    or when the cited source *is* that party by domain or title.
    Otherwise the claim is withheld: a vendor page asserting what a
    standards body requires is not that standards body saying it.
    """
    named = attributed_entities(claim)
    if not named:
        return GuardResult("attribution", True, "claim attributes nothing to a named party")

    unestablished: list[str] = []
    established: list[str] = []
    for name in named:
        where = _establishes(name, evidence, source)
        (established if where else unestablished).append(f"{name} ({where})" if where else name)
    if unestablished:
        return GuardResult(
            "attribution",
            False,
            f"claim attributes to {', '.join(unestablished)}, "
            f"not established by the quote or the cited source",
        )
    return GuardResult("attribution", True, f"attribution established: {'; '.join(established)}")


ALL_GUARDS = (
    numeric_guard,
    modality_guard,
    hedge_guard,
    framing_guard,
    ranking_guard,
    causal_guard,
    exclusivity_guard,
)
GUARD_NAMES = (*(g(" ", " ").name for g in ALL_GUARDS), "attribution")


def run_guards(
    claim: str, evidence: str, source: SourceIdentity | None = None
) -> list[GuardResult]:
    """Every guard, always, so the audit record shows what each decided.

    ``source`` is identity only and reaches the attribution guard alone.
    It never becomes part of the NLI premise.
    """
    results = [guard(claim, evidence) for guard in ALL_GUARDS]
    results.append(attribution_guard(claim, evidence, source))
    return results


def guards_pass(results: list[GuardResult]) -> bool:
    return all(r.passed for r in results)


def failed_guards(results: list[GuardResult]) -> list[GuardResult]:
    return [r for r in results if not r.passed]
