"""Publication gate: publish only what the verifier actually supported.

Deterministic and conservative. Nothing is re-asked of a model and no
replacement prose is invented.

The rule is an allow-list, not a deny-list, and that distinction is the
whole point. An earlier version removed claims carrying a *failing*
verdict, which silently published every claim the verifier never reached:
a bounded live run checked 1 of 6 eligible claims and published all six.
A claim now earns publication by being checked and supported.

    supported        -> published
    partially        -> removed ("mostly true" reads as "true")
    unsupported      -> removed
    never checked    -> removed

Framing claims carry no evidence by design and are never gated on it.

The verification record is left intact, so removal stays auditable: the
issues explaining each failing verdict remain in the structured artifact
after the claim has gone from the report.

This is a filter, not a repair. Rewriting a partially supported claim into
one its evidence does support is a harder problem, deliberately not
attempted here.
"""

from __future__ import annotations

import re
from typing import Literal

from agentic_research.models import Claim, ClaimKind, Contradiction, ResearchReport

ClaimVerdict = Literal["supported", "partially_supported", "unsupported", "irrelevant"]
"""``irrelevant`` is supported and answers nothing that was asked."""

# Identity of a claim within one report. Exact, not fuzzy: the verifier
# records a verdict under this key while holding the claim object, and the
# gate looks it up under the same key. Text alone is not enough -- two
# sections can restate a finding -- so the evidence it cites is part of it.
ClaimKey = tuple[str, tuple[str, ...]]

_SUPPORTED: ClaimVerdict = "supported"


def claim_key(text: str, evidence_ids: list[str]) -> ClaimKey:
    """Stable identity for a claim within one report.

    The whole text, and the ids as a tuple. Identity used to reuse the
    200-character truncation the issue record applies for display, which
    is a presentation concern leaking into correctness: two claims that
    agree for 200 characters and then diverge -- one supported, one not --
    collapsed onto a single verdict. Joining the ids into a string had
    the same shape of problem, since an identifier containing a comma
    would alias two different citation sets.

    ``claim_text[:200]`` stays where it belongs, in CitationIssue.
    """
    return (text, tuple(evidence_ids))


def key_of(claim: Claim) -> ClaimKey:
    return claim_key(claim.text, claim.evidence_ids)


def _keep(claim: Claim, verdicts: dict[ClaimKey, ClaimVerdict]) -> bool:
    # Framing owes no evidence. An extracted finding owes a verbatim
    # quote, which it already passed to be citable at all, and restates a
    # single evidence item rather than synthesising across several --
    # there is no inference for entailment to check. Gating it removed
    # the entire degraded report when a provider outage stopped both
    # synthesis and verification, discarding every finding the run had
    # already paid to retrieve.
    if not claim.kind.requires_entailment:
        return True
    # A substantive claim with no evidence never reaches the verifier, so
    # it has no verdict and is not published. That is the intended
    # reading of "published claims link to verified source passages".
    return verdicts.get(key_of(claim)) == _SUPPORTED


def _normalised(text: str) -> str:
    """Case- and whitespace-insensitive form, for exact-duplicate only.

    Deliberately not fuzzy. Two claims that differ by a word are two
    claims, and collapsing them would be an editorial judgement made by
    a similarity threshold. This collapses only text that is already the
    same sentence.
    """
    return " ".join(text.lower().split()).rstrip(".")


_WORD = re.compile(r"[a-z0-9]+")


def _content_words(text: str) -> frozenset[str]:
    """The words a claim asserts with, stopwords removed.

    Possessives need no special handling: `[a-z0-9]+` splits
    "LangGraph's" into "langgraph" and "s", and the length filter drops
    the "s". An explicit strip was written here first and removed when a
    mutation survived it -- the mutation was right, the code was dead.
    """
    from agentic_research.citations.guards import _STOPWORDS

    return frozenset(w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1)


def _restates(a: Claim, b: Claim) -> bool:
    """Whether one claim is the other with its words rearranged.

    Narrower than it sounds, and deliberately not a similarity
    threshold -- `_normalised` above refuses those for good reason, and
    this does not relax it. The test is **set equality** on content
    words within **the same answer slot**: not "these two are similar",
    but "these two assert with exactly the same words, about the same
    part of the answer".

    The live case, published twice in one report from one source:

        LangChain components are the components on which LangGraph's
        orchestration layer is built.
        LangGraph is an orchestration layer built on LangChain
        components.

    The engine already noticed -- it printed "More than one published
    claim fills the relationship slot; they may repeat each other" --
    and published both anyway.

    The two sides of a comparison pair are safe from this by
    construction: each names a different subject, so their word sets
    differ. A slotless claim is never collapsed, because without a slot
    there is nothing to say the two address the same thing.
    """
    slot = (a.answer_slot or "").strip()
    if not slot or slot != (b.answer_slot or "").strip():
        return False
    words = _content_words(a.text)
    return bool(words) and words == _content_words(b.text)


def deduplicate_claims(report: ResearchReport) -> tuple[ResearchReport, int]:
    """Drop repeats of a claim that already appears earlier in the report.

    A synthesiser routinely states its strongest finding in the summary,
    again under key findings, and again in the body. One recording
    published the same sentence three times, which reads as three
    findings and inflates every claim count derived from the report.

    Identity is normalised text plus cited evidence ids -- the same
    sentence citing different evidence is a different claim and is kept.
    Order of precedence is fixed and documented: summary, then key
    findings, then sections in order. The earliest occurrence survives,
    because that is where the synthesiser chose to lead with it.

    Sections emptied by this are dropped; an empty heading is not a
    section.
    """
    seen: set[ClaimKey] = set()
    # Every substantive claim kept so far, across the whole report. A
    # restatement is usually in the same section as the thing it
    # restates, but nothing guarantees that, and the precedence rule
    # below is report-wide already.
    kept_claims: list[Claim] = []
    removed = 0

    def keep(claims: list[Claim]) -> list[Claim]:
        nonlocal removed
        kept: list[Claim] = []
        for claim in claims:
            # Framing is connective text. Two sections may legitimately
            # open the same way, and it carries no evidence to compare.
            if claim.kind is ClaimKind.FRAMING:
                kept.append(claim)
                continue
            # A restatement is a duplicate even though its text differs.
            if any(_restates(claim, earlier) for earlier in kept_claims):
                removed += 1
                continue
            key = (_normalised(claim.text), tuple(claim.evidence_ids))
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            kept_claims.append(claim)
            kept.append(claim)
        return kept

    summary = keep(report.summary_claims)
    findings = keep(report.key_findings)
    sections = []
    for section in report.sections:
        claims = keep(section.claims)
        if claims:
            sections.append(section.model_copy(update={"claims": claims}))

    if not removed:
        return report, 0

    return (
        report.model_copy(
            update={
                "summary_claims": summary,
                "key_findings": findings,
                "sections": sections,
            }
        ),
        removed,
    )


def _contradiction_supported(
    contradiction: Contradiction, verdicts: dict[ClaimKey, ClaimVerdict]
) -> bool:
    """Both summaries must have been checked and both supported.

    A contradiction is two assertions about what sources say, and they
    reached the report without any support check -- structural resolution
    proved only that evidence existed on each side. One unsupported side
    makes the pairing misleading even when the other is sound, so the
    whole contradiction goes.
    """
    left = verdicts.get(claim_key(contradiction.left_summary, contradiction.left_evidence_ids))
    right = verdicts.get(claim_key(contradiction.right_summary, contradiction.right_evidence_ids))
    return left == _SUPPORTED and right == _SUPPORTED


def filter_report_by_verification(
    report: ResearchReport,
    verdicts: dict[ClaimKey, ClaimVerdict],
    *,
    off_subject: int = 0,
    mislabelled_contrast: int = 0,
) -> tuple[ResearchReport, int]:
    """Keep only substantive claims with a supported verdict.

    Returns the filtered report and how many claims were removed. A claim
    restated in two places shares one key, so both copies resolve to the
    same verdict and are kept or removed together: leaving one copy of
    rejected text would publish exactly what the verifier rejected.

    Sections left with no claims are dropped, since an empty heading is
    not a section. No replacement text is generated for what was removed.
    """
    removed = 0

    def filter_claims(claims: list[Claim]) -> list[Claim]:
        nonlocal removed
        kept = [c for c in claims if _keep(c, verdicts)]
        removed += len(claims) - len(kept)
        return kept

    summary = filter_claims(report.summary_claims)
    findings = filter_claims(report.key_findings)
    sections = []
    for section in report.sections:
        claims = filter_claims(section.claims)
        if claims:
            sections.append(section.model_copy(update={"claims": claims}))

    contradictions = [c for c in report.contradictions if _contradiction_supported(c, verdicts)]
    dropped_contradictions = len(report.contradictions) - len(contradictions)

    if not removed and not dropped_contradictions:
        return report, 0

    limitations = list(report.limitations)
    if removed:
        # The cause, where the engine can tell which cause it was.
        #
        # "the cited evidence did not support them" describes an honest
        # near miss and a quote retrofitted to a sentence identically. A
        # hosted run refused five claims; four cited quotes that were not
        # about the claim's subject at all, at entailment 0.0013-0.0064.
        # A reader could not tell that from the sentence, and the two
        # need different fixes.
        cause = (
            f"{removed} generated claim(s) were excluded because the cited evidence "
            "did not support them, or because verification did not reach them within "
            "this run's budget."
        )
        if off_subject:
            cause += (
                f" Of those, {off_subject} cited a quote that was not about the "
                "claim's own subject, which points at how the claim was assembled "
                "rather than at the evidence."
            )
        if mislabelled_contrast:
            # Not an evidence failure at all, and the only cause here a
            # reader can see was self-inflicted: the claims were verified
            # and were about the question, and were deleted for the slot
            # they declared.
            cause += (
                f" A further {mislabelled_contrast} were verified and then deleted "
                "for declaring the contrast slot while describing a single subject; "
                "labelled with the axis they addressed, they could have been "
                "paired."
            )
        limitations.append(cause)
    if dropped_contradictions:
        limitations.append(
            f"{dropped_contradictions} reported disagreement(s) were excluded because "
            "the evidence did not support both sides as stated."
        )

    filtered = report.model_copy(
        update={
            "summary_claims": summary,
            "key_findings": findings,
            "sections": sections,
            "contradictions": contradictions,
            "limitations": limitations,
        }
    )
    return filtered, removed
