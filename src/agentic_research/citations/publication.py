"""Publication gate: drop claims whose evidence did not support them.

Deterministic and conservative. A claim the verifier judged unsupported or
only partially supported is removed from the published report, not
rewritten. Nothing is re-asked of a model, and no replacement prose is
invented.

The verification record is left intact, so the removal stays auditable:
the issues explaining why each claim failed remain in the structured
artifact after the claim has gone from the report.

This is a filter, not a repair. Rewriting a partially supported claim into
one its evidence does support is a harder problem and is deliberately not
attempted here.
"""

from __future__ import annotations

from agentic_research.models import (
    CitationIssueType,
    CitationVerification,
    Claim,
    ClaimKind,
    ResearchReport,
)

# Verdicts that disqualify a claim from publication. Partial support is
# included: "mostly true" reads to a reader exactly like "true", and the
# report offers no way to tell them apart.
_REJECTING_ISSUES = frozenset(
    {
        CitationIssueType.UNSUPPORTED_CLAIM,
        CitationIssueType.PARTIALLY_SUPPORTED_CLAIM,
    }
)


def _key(text: str, evidence_ids: list[str]) -> tuple[str, str]:
    """Stable identity for a claim within one report.

    Uses the same truncation the verifier applied when recording the issue,
    so both sides compare identical strings. Text alone is not enough --
    two sections can restate a finding -- so the evidence it cites is part
    of the key.
    """
    return (text[:200], ",".join(evidence_ids))


def rejected_keys(verification: CitationVerification) -> set[tuple[str, str]]:
    """Claims the verifier judged unsupported or partially supported."""
    return {
        _key(issue.claim_text, issue.evidence_id.split(",") if issue.evidence_id else [])
        for issue in verification.issues
        if issue.type in _REJECTING_ISSUES
    }


def _keep(claim: Claim, rejected: set[tuple[str, str]]) -> bool:
    """Framing carries no evidence by design and is never gated on it."""
    if claim.kind is ClaimKind.FRAMING:
        return True
    return _key(claim.text, claim.evidence_ids) not in rejected


def filter_report_by_verification(
    report: ResearchReport, verification: CitationVerification
) -> tuple[ResearchReport, int]:
    """Remove claims that failed the support check, everywhere they appear.

    Returns the filtered report and how many claims were removed. A claim
    restated in two places is removed from both: leaving one copy would
    publish the exact text the verifier rejected.

    Sections left with no claims are dropped, since an empty heading is
    not a section. No replacement text is generated for what was removed.
    """
    rejected = rejected_keys(verification)
    if not rejected:
        return report, 0

    removed = 0

    def filter_claims(claims: list[Claim]) -> list[Claim]:
        nonlocal removed
        kept = [c for c in claims if _keep(c, rejected)]
        removed += len(claims) - len(kept)
        return kept

    summary = filter_claims(report.summary_claims)
    findings = filter_claims(report.key_findings)
    sections = []
    for section in report.sections:
        claims = filter_claims(section.claims)
        # A section whose every claim failed has nothing left to say.
        if claims:
            sections.append(section.model_copy(update={"claims": claims}))

    limitations = list(report.limitations)
    if removed:
        limitations.append(
            f"{removed} generated claim(s) were excluded because the cited "
            "evidence did not fully support them."
        )

    filtered = report.model_copy(
        update={
            "summary_claims": summary,
            "key_findings": findings,
            "sections": sections,
            "limitations": limitations,
        }
    )
    return filtered, removed
