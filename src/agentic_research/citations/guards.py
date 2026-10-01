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
from enum import StrEnum
from functools import lru_cache
from typing import Any

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
# A sentence carrying no modal marker asserts its claim flatly, and a flat
# assertion is *stronger* than "may" -- weaker only than necessity.
# Scoring it 0 put it below every hedge, so bare evidence became the
# weakest possible premise and *any* hedged claim failed against it: a
# claim reporting "SQLite deployment can consist of copying the file"
# from flat evidence was refused for being more cautious than its source.
# Three claims died this way in one live run.
#
# `repair.py` found this and fixed it locally with its own
# `_BARE_ASSERTION_LEVEL = 3`; the module that actually gates publication
# never got the fix. `test_modality_ladders_agree` now holds them together.
_BARE_ASSERTION = 3

_BANDS: tuple[tuple[int, tuple[str, ...]], ...] = ((1, _WEAK), (2, _TENDENCY), (4, _STRONG))

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
_EPISTEMIC = (
    "may",
    "might",
    "possibly",
    "potentially",
    "perhaps",
    "maybe",
    "presumably",
    "likely",
)

# Evidential hedges: the source is reporting an indication, not a fact.
# "Evidence suggests X" published as "X" is the same overclaim as
# dropping "may", expressed as a verb.
_EVIDENTIAL = ("suggests", "suggest", "indicates", "indicate", "appears to", "seems to")

# Frequency adverbs. Deleting one widens "often" to "always" by
# implication. Handled here rather than in the band rule for the same
# reason as the epistemic case: the band rule compares strengths and a
# claim with no adverb has no strength to compare.
_FREQUENCY = (
    "often",
    "typically",
    "generally",
    "usually",
    "commonly",
    "frequently",
    "sometimes",
    "occasionally",
)

# Deliberately absent, each for a stated reason:
#
#   can, could      capability, not doubt. "can process" -> "processes"
#                   is faithful, and treating them as epistemic was
#                   measured against the release audit and rejected a
#                   true claim.
#   unlikely        deleting it inverts polarity rather than
#                   strengthening degree. That is a negation error, and
#                   the classifier handles negation well; a guard here
#                   would fire on the wrong axis.
#   approximately   the numeric guard already pins the literal, and the
#                   residual difference between "5ms" and
#                   "approximately 5ms" is too small to justify
#                   withholding otherwise sound claims.

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


class SourceAuthority(StrEnum):
    """How close a source is to what it reports.

    Not a truth score and not a quality score. A primary source can be
    wrong and an aggregator can be right; this says only who is doing
    the reporting, which is what decides *which* of several supporting
    quotes should carry a claim.
    """

    PRIMARY = "primary"
    """The study, the specification, the official documentation."""
    SECONDARY = "secondary"
    """An account of a primary source: an article, an encyclopaedia."""
    AGGREGATOR = "aggregator"
    """An index, a social post, a listing. Useful for discovery."""
    UNKNOWN = "unknown"


# Ranked worst to best, so a larger number is a stronger source. Used
# only to order evidence that has *already* passed entailment and the
# guards; it never contributes to whether a claim is supported.
def authority_rank_of(source_type: str) -> int:
    """Authority as a number, for callers building a sort key.

    Exists so the rank table stays private to this module: the
    evidence layer needs the ordering, not the mapping.
    """
    return _AUTHORITY_RANK[authority_of(source_type)]


def authority_of(source_type: str) -> SourceAuthority:
    """Map a source's coarse kind to how close it is to what it reports.

    Kept beside the enum rather than in the evidence layer because it
    is a statement about the same distinction the enum makes, and a
    second copy elsewhere would drift.

    A standards body, a paper, a government publisher and first-party
    documentation are all the thing itself for their own subject. News
    and vendor pages are accounts of something else. Blogs and forums
    are where a reader goes to find the primary source, not instead of
    it -- useful for discovery, which is what AGGREGATOR means here.

    Anything unrecognised is UNKNOWN and therefore ranks last, which
    is the safe direction: an unclassified source never outranks a
    classified one on authority alone.
    """
    return _AUTHORITY_BY_TYPE.get(source_type, SourceAuthority.UNKNOWN)


_AUTHORITY_BY_TYPE: dict[str, SourceAuthority] = {
    "standards_body": SourceAuthority.PRIMARY,
    "academic": SourceAuthority.PRIMARY,
    "official_docs": SourceAuthority.PRIMARY,
    "government": SourceAuthority.PRIMARY,
    "news": SourceAuthority.SECONDARY,
    "vendor": SourceAuthority.SECONDARY,
    "reference": SourceAuthority.SECONDARY,
    "other": SourceAuthority.SECONDARY,
    "blog": SourceAuthority.AGGREGATOR,
    "forum": SourceAuthority.AGGREGATOR,
    "social": SourceAuthority.AGGREGATOR,
}


_AUTHORITY_RANK: dict[SourceAuthority, int] = {
    SourceAuthority.UNKNOWN: 0,
    SourceAuthority.AGGREGATOR: 1,
    SourceAuthority.SECONDARY: 2,
    SourceAuthority.PRIMARY: 3,
}


@dataclass(frozen=True)
class SourceIdentity:
    """Who published the quote.

    ``domain`` and ``title`` are what the attribution guard reads.

    ``authority`` and ``quality`` are read only when choosing between
    quotes that have already been accepted, and they are deliberately
    never part of the premise. A classifier told that a quote came
    from an authoritative domain would be scoring reputation, and
    entailment is the only thing it is allowed to score. A test asserts
    the scorer never sees either field.
    """

    domain: str = ""
    title: str = ""
    authority: SourceAuthority = SourceAuthority.UNKNOWN
    quality: float = 0.0

    @property
    def authority_rank(self) -> int:
        return _AUTHORITY_RANK[self.authority]


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
    """Strongest unhedged modality band in the text.

    The ladder is hedge (1), tendency (2), bare assertion (3), necessity
    (4). Text with no modal marker at all is a bare assertion, not an
    absence of one -- see :data:`_BARE_ASSERTION`.

    Hedged necessities are collapsed to their hedge first, so "may
    require" reports band 1 rather than band 4.
    """
    lowered = _HEDGED.sub(r"\1\2", (text or "").lower())
    band = 0
    for level, terms in _BANDS:
        if _phrases(lowered, terms):
            band = max(band, level)
    return band or _BARE_ASSERTION


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
    """A claim may not be more certain than its evidence.

    This guard owns *strengthening* -- "may" becoming "typically", or
    "can" becoming "must". It deliberately does not own *deletion* --
    "may X" becoming "X" -- which :func:`hedge_guard` checks against the
    single sentence that actually carries the claim, and which is not
    repairable by rewording. A bare claim is therefore exempted here, so
    that a deletion is reported once, by the guard that classifies it
    correctly, rather than being reclassified as repairable wording.

    Hedging *below* the evidence is always safe and is allowed: that is
    the direction the band fix above restored.
    """
    claim_band = modality_band(claim)
    evidence_band = modality_band(evidence)
    if claim_band == _BARE_ASSERTION:
        return GuardResult("modality", True, "claim asserts no modality of its own")
    if claim_band > evidence_band:
        return GuardResult(
            "modality",
            False,
            f"claim modality band {claim_band} exceeds evidence band {evidence_band}",
        )
    return GuardResult(
        "modality", True, f"claim band {claim_band} within evidence band {evidence_band}"
    )


def _hedge_terms(text: str) -> list[tuple[str, str]]:
    """Hedges present, each tagged with the kind of hedge it is."""
    words = _words(text)
    lowered = (text or "").lower()
    found = [("epistemic", w) for w in _EPISTEMIC if w in words]
    found += [("frequency", w) for w in _FREQUENCY if w in words]
    found += [("evidential", w) for w in _EVIDENTIAL if (w in lowered if " " in w else w in words)]
    return found


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
    "must" -- and deliberately abstains on deletion, because its rule is
    "claim band must not exceed evidence band" and a bare claim sits at
    :data:`_BARE_ASSERTION`, above every hedge. So "may" to "must" fails
    there while "may" to nothing is left to this guard, which is the
    right split: deletion is not repairable by rewording. Deletion is
    also the more common overclaim of the two. The
    release audit published exactly one unsupported claim and this was
    it, at 0.9946 entailment, so the classifier does not catch it
    either.

    Scoped to the sentence that supports the claim rather than the whole
    quote, so an unrelated hedge elsewhere in the passage does not
    withhold a faithful claim.
    """
    supporting = _supporting_sentence(claim, evidence)
    hedges = _hedge_terms(supporting)
    if not hedges:
        return GuardResult("hedge", True, "evidence states no hedge")
    kept = _hedge_terms(claim)
    if kept:
        return GuardResult("hedge", True, f"claim keeps a hedge: {kept[0][1]}")
    kinds = ", ".join(sorted({f"{kind} '{word}'" for kind, word in hedges}))
    return GuardResult("hedge", False, f"evidence hedges with {kinds}; the claim states it as fact")


# A source writing in its own research voice is reporting its own
# finding, not stating a settled fact. "We demonstrate that X" and "X"
# are different assertions, and the difference is exactly what a reader
# needs to judge the claim.
_RESEARCH_VOICE = re.compile(
    r"\b(?:"
    # First-person reporting, both tenses. The first version listed
    # present forms only, so "we observed" slipped through and a
    # source's own dataset was published as a general finding.
    r"we\s+(?:demonstrate|demonstrated|show|showed|find|found|propose|proposed|"
    r"argue|argued|observe|observed|conclude|concluded|present|presented|"
    r"report|reported|introduce|introduced|evaluate|evaluated|measure|measured|"
    r"test|tested|collect|collected|train|trained)"
    # Any first-person possessive. "our dataset", "our users", "our
    # benchmark" all scope a proposition to the source, and dropping the
    # possessive widens it to everyone.
    r"|our\s+\w+"
    r"|this\s+(?:paper|study|work|article|report)\s+"
    r"(?:demonstrates?|demonstrated|shows?|showed|finds?|found|proposes?|proposed|"
    r"argues?|argued|presents?|presented|reports?|reported|concludes?|concluded)"
    r")\b",
    re.IGNORECASE,
)

# Ways a claim can keep that framing.
_KEEPS_FRAMING = re.compile(
    r"\b(?:the\s+(?:authors?|study|paper|research|report|work|survey|analysis)|"
    r"researchers|according to|reportedly|is reported|are reported|was reported|"
    r"were reported|reports? that|found that|suggests? that|argues? that|"
    r"our\s+\w+|in\s+one\s+(?:study|dataset|experiment))\b",
    re.IGNORECASE,
)


def atomicity_guard(claim: str, evidence: str) -> GuardResult:
    """A substantive claim must assert exactly one proposition.

    Not a style rule. Every guard that reasons about "the sentence that
    supports this claim" needs the claim to be one assertion; given two,
    it scopes to whichever half shares more words and never examines the
    other. One audit published a two-sentence claim whose first half had
    deleted the source's own voice while the guards inspected its clean
    second half.

    Counting sentences is not enough, which is how this guard was wrong
    for one release: "X increased, Y decreased" is a single sentence and
    two independently falsifiable assertions, and a gate that verifies
    one quote against one claim cannot honestly call that atomic. The
    decision is delegated to
    :func:`agentic_research.citations.atomicity.compound_propositions`,
    which looks at clause structure. Sentence counting survives inside
    it as one of several signals.

    Conservative by construction: where the claim cannot be shown to
    assert one thing it is treated as compound and withheld. Withholding
    a true claim costs a line in a report; publishing a fused one costs
    the thing this system is for.
    """
    from agentic_research.citations.atomicity import compound_propositions

    reasons = compound_propositions(claim)
    if not reasons:
        return GuardResult(
            "atomicity", True, "claim contains one independently verifiable proposition"
        )
    return GuardResult(
        "atomicity",
        False,
        f"claim asserts more than one proposition ({'; '.join(reasons)}); "
        "only one can be verified against one quote",
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


@lru_cache(maxsize=1)
def _suffix_parser() -> Any:
    """Public Suffix List lookup, pinned and strictly offline.

    A hand-curated suffix set is incomplete by construction -- it had
    co.uk and com.au and would have mis-parsed pages.dev, s3 buckets
    and every ccTLD nobody thought of. tldextract ships a PSL snapshot;
    configured with no suffix URLs it never fetches at runtime, so
    behaviour is deterministic and the data is versioned with the
    dependency rather than with the network.
    """
    import tldextract

    # Private suffixes included: a page hosted at acme.pages.dev or
    # acme.github.io is registered by acme, not by the platform, and
    # identity is the question being asked. Without this the platform
    # label would be the "organisation" for every such host.
    return tldextract.TLDExtract(
        suffix_list_urls=(), fallback_to_snapshot=True, include_psl_private_domains=True
    )


def _registrable_labels(domain: str) -> set[str]:
    """Organisation labels of a host's registrable domain.

    Substring matching cannot be used here. "nist.gov" and
    "evilnist.gov" share the substring, and so do "nist.gov" and
    "nist.gov.example.com" -- the second of which is registered by
    whoever owns example.com and has nothing to do with NIST. Both
    passed before this existed.

    Only the registrable domain's own label identifies anyone. The
    public suffix does not: a claim attributed to "Gov" must not pass on
    every .gov host, and a subdomain must not either.

    Fails closed. A host whose suffix the PSL does not recognise yields
    no labels, so an attributed claim citing it is withheld rather than
    accepted on a guess.
    """
    host = (domain or "").strip().lower()
    host = host.split("//")[-1].split("/")[0].split("?")[0].split("#")[0]
    host = host.split("@")[-1].split(":")[0].strip(".")
    if not host:
        return set()

    try:
        extracted = _suffix_parser()(host)
    except Exception:  # pragma: no cover - parser should not raise
        return set()
    if not extracted.domain or not extracted.suffix:
        return set()
    return {extracted.domain}


def _normalise_entity(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _entity_tokens(name: str) -> list[str]:
    """The name as a token sequence.

    Hyphens stay inside a token, so "NIST-like" is one token and does
    not match "NIST": a framework described as NIST-like is explicitly
    not NIST, and splitting on the hyphen made it establish NIST as the
    publisher. Apostrophes, periods and spaces still separate, so
    "NIST's" and "U.S. Department" resolve as expected.
    """
    return [t.strip("-") for t in re.split(r"[^A-Za-z0-9-]+", name or "") if t.strip("-")]


def _names_in_text(entity: str, text: str) -> bool:
    """Whether the text names this party, at token boundaries.

    Substring matching cannot be used. Collapsing both sides to
    alphanumerics and asking for containment makes "WHO" match "people
    who use X", "US" match "business", "AI" match "retail chain" via
    "chain", and "NIST" match "a NIST-like framework". All four passed
    before this existed, and none is the organisation being cited.

    An acronym must appear as a standalone token in its own case -- "WHO"
    is the agency, "who" is a pronoun, and that distinction is the only
    thing separating them. A multi-word name must appear as a complete
    consecutive token sequence. Possessives and internal punctuation are
    tolerated on both sides, so "NIST's" and "U.S. Department" match.
    """
    wanted = _entity_tokens(entity)
    if not wanted:
        return False

    tokens = _entity_tokens(text)
    if not tokens:
        return False

    def matches(candidate: str, target: str) -> bool:
        stripped = candidate[:-1] if candidate.lower().endswith("s") else candidate
        if target.isupper() and len(target) <= 5:
            # Acronym: case-sensitive, so the pronoun "who" cannot
            # establish the World Health Organization.
            return candidate == target or stripped == target
        return candidate.lower() == target.lower() or stripped.lower() == target.lower()

    span = len(wanted)
    return any(
        all(matches(tokens[i + offset], wanted[offset]) for offset in range(span))
        for i in range(len(tokens) - span + 1)
    )


def _establishes(name: str, quote: str, source: SourceIdentity | None) -> str:
    """Where, if anywhere, this attribution is borne out.

    Two ways only, both narrow:

    * the quoted text names the party at token boundaries, or
    * the cited source *is* that party by registrable domain.

    Source title is deliberately not a third way. "NIST guidance
    explained by VendorCo" contains "NIST" while being published by
    VendorCo, and no constraint on where in the title the name appears
    separates that from a real NIST page. An unverifiable signal that
    looks verifiable is worse than no signal.
    """
    if _names_in_text(name, quote or ""):
        return "the quoted text names it"
    if source is not None:
        wanted = _normalise_entity(name)
        labels = {_normalise_entity(label) for label in _registrable_labels(source.domain)}
        if wanted and wanted in labels:
            return f"the registrable domain is {source.domain}"
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

    An attribution is established when the quote itself names the party
    at token boundaries, or when the cited source's registrable domain
    establishes that the source *is* that party. Source title is not an
    identity signal -- "NIST guidance explained by VendorCo" contains
    "NIST" while being published by VendorCo, so title matching was
    removed rather than constrained. Otherwise the claim is withheld: a
    vendor page asserting what a standards body requires is not that
    standards body saying it.
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
    atomicity_guard,
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
