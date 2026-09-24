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

from typing import Literal

from agentic_research.models import Claim, ClaimKind, ResearchReport

ClaimVerdict = Literal["supported", "partially_supported", "unsupported"]

# Identity of a claim within one report. Exact, not fuzzy: the verifier
# records a verdict under this key while holding the claim object, and the
# gate looks it up under the same key. Text alone is not enough -- two
# sections can restate a finding -- so the evidence it cites is part of it.
ClaimKey = tuple[str, str]

_SUPPORTED: ClaimVerdict = "supported"


def claim_key(text: str, evidence_ids: list[str]) -> ClaimKey:
    """Stable identity for a claim within one report.

    Truncated at the same 200 characters the issue record uses, so both
    sides compare identical strings.
    """
    return (text[:200], ",".join(evidence_ids))


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
            key = (_normalised(claim.text), ",".join(claim.evidence_ids))
            if key in seen:
                removed += 1
                continue
            seen.add(key)
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


def filter_report_by_verification(
    report: ResearchReport, verdicts: dict[ClaimKey, ClaimVerdict]
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

    if not removed:
        return report, 0

    limitations = [
        *report.limitations,
        f"{removed} generated claim(s) were excluded because the cited evidence "
        "did not support them, or because verification did not reach them within "
        "this run's budget.",
    ]
    filtered = report.model_copy(
        update={
            "summary_claims": summary,
            "key_findings": findings,
            "sections": sections,
            "limitations": limitations,
        }
    )
    return filtered, removed
