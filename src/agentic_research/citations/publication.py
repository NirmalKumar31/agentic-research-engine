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
    if claim.kind is ClaimKind.FRAMING:
        return True
    # A substantive claim with no evidence never reaches the verifier, so
    # it has no verdict and is not published. That is the intended
    # reading of "published claims link to verified source passages".
    return verdicts.get(key_of(claim)) == _SUPPORTED


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
