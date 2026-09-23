"""Citation resolution, validation and repair.

Provenance is evidence-first. A model supplies ``evidence_ids`` on each claim
and nothing else; this module resolves those to source ids, rejects ids that
do not exist or point at evidence whose quote never aligned to its source,
and derives the citation markers the reader sees.

That ordering matters. The previous design had the model emit source ids
directly, which left no record of *which* evidence produced a sentence — so
entailment checking had to sample arbitrary evidence belonging to the cited
source and hope. A claim can now be audited exactly:

    Claim -> EvidenceItem -> quote/page -> SourceDocument -> DiscoveryRef
          -> SearchQuery -> SubQuestion

Checks run cheapest-first, which is also strongest-first:

1. **Structural** (free, deterministic) — do the referenced evidence ids
   exist, are they citable, do they resolve to retrieved sources, does every
   claim that owes evidence have some? A failure here is always real.
2. **Entailment** (one model call per claim) — does the cited evidence
   actually support this claim? A judgement, and the only part that costs.
"""

from __future__ import annotations

import re

from agentic_research.evidence.store import EvidenceStore
from agentic_research.models import (
    CitationIssue,
    CitationIssueType,
    CitationVerification,
    Claim,
    Contradiction,
    ResearchReport,
)

# Legacy prose markers such as [S3] or [S1, S2]. The schema no longer asks for
# these, but a model occasionally emits them anyway and they must not survive
# into rendered text where they would look like verified citations.
_MARKER = re.compile(r"\[(S\d+(?:-e\d+)?(?:\s*,\s*S\d+(?:-e\d+)?)*)\]")


def extract_markers(text: str) -> list[str]:
    """Pull any ids out of legacy bracket markers, in order."""
    found: list[str] = []
    for group in _MARKER.findall(text):
        for token in group.split(","):
            token = token.strip()
            if token and token not in found:
                found.append(token)
    return found


def strip_markers(text: str) -> str:
    """Remove bracket markers from prose.

    Applied to every claim before rendering. A marker the model invented is
    not a citation — it never went through evidence resolution — so leaving it
    in the text would show the reader an unverified reference that looks
    identical to a verified one.
    """
    return re.sub(r"\s{2,}", " ", _MARKER.sub("", text)).replace(" .", ".").strip()


def resolve_claim(claim: Claim, store: EvidenceStore) -> tuple[Claim, list[CitationIssue]]:
    """Validate a claim's evidence ids and derive its citation ids.

    Unknown or uncitable evidence references are dropped from the claim and
    reported as errors. Dropping the reference rather than the sentence is the
    conservative choice: the sentence may well be true and merely
    mis-referenced, but a dangling pointer is always wrong. The stripped claim
    then surfaces as uncited on re-verification, which is an honest
    description of its state.
    """
    issues: list[CitationIssue] = []
    kept: list[str] = []

    for evidence_id in dict.fromkeys(claim.evidence_ids):
        item = store.evidence_by_id(evidence_id)
        if item is None:
            issues.append(
                CitationIssue(
                    type=CitationIssueType.UNKNOWN_EVIDENCE,
                    severity="error",
                    claim_text=claim.text[:200],
                    evidence_id=evidence_id,
                    detail=f"{evidence_id} does not exist in this run",
                )
            )
            continue
        if not item.is_citable:
            issues.append(
                CitationIssue(
                    type=CitationIssueType.UNCITABLE_EVIDENCE,
                    severity="error",
                    claim_text=claim.text[:200],
                    evidence_id=evidence_id,
                    detail=(
                        f"{evidence_id} quote did not align with its source "
                        f"({item.quote_match.value})"
                    ),
                )
            )
            continue
        if store.source(item.source_id) is None:
            issues.append(
                CitationIssue(
                    type=CitationIssueType.UNKNOWN_SOURCE,
                    severity="error",
                    claim_text=claim.text[:200],
                    citation_id=item.source_id,
                    evidence_id=evidence_id,
                    detail=f"{evidence_id} points at unretrieved source {item.source_id}",
                )
            )
            continue
        kept.append(evidence_id)

    resolved = claim.model_copy(
        update={
            "text": strip_markers(claim.text),
            "evidence_ids": kept,
            "citation_ids": store.resolve_citations(kept),
        }
    )
    return resolved, issues


def resolve_contradiction(
    contradiction: Contradiction, store: EvidenceStore
) -> tuple[Contradiction, list[CitationIssue]]:
    """Validate both sides of a contradiction and derive their citations."""
    issues: list[CitationIssue] = []

    def keep(ids: list[str], side: str) -> list[str]:
        good: list[str] = []
        for evidence_id in dict.fromkeys(ids):
            item = store.evidence_by_id(evidence_id)
            if item is None or not item.is_citable:
                issues.append(
                    CitationIssue(
                        type=CitationIssueType.UNKNOWN_EVIDENCE,
                        severity="error",
                        claim_text=f"{contradiction.topic} ({side})"[:200],
                        evidence_id=evidence_id,
                        detail=f"{evidence_id} is unknown or not citable",
                    )
                )
                continue
            good.append(evidence_id)
        return good

    left = keep(contradiction.left_evidence_ids, "left")
    right = keep(contradiction.right_evidence_ids, "right")
    resolved = contradiction.model_copy(
        update={
            "left_summary": strip_markers(contradiction.left_summary),
            "right_summary": strip_markers(contradiction.right_summary),
            "left_evidence_ids": left,
            "right_evidence_ids": right,
            "left_citation_ids": store.resolve_citations(left),
            "right_citation_ids": store.resolve_citations(right),
        }
    )
    if not resolved.is_auditable:
        issues.append(
            CitationIssue(
                type=CitationIssueType.UNAUDITABLE_CONTRADICTION,
                severity="warning",
                claim_text=contradiction.topic[:200],
                detail="a reported disagreement needs citable evidence on both sides",
            )
        )
    return resolved, issues


def resolve_report(
    report: ResearchReport, store: EvidenceStore
) -> tuple[ResearchReport, list[CitationIssue]]:
    """Resolve every claim and contradiction in a report.

    Runs before verification and before rendering, so what the reader sees and
    what the verifier measures are the same object.
    """
    issues: list[CitationIssue] = []

    def resolve_all(claims: list[Claim]) -> list[Claim]:
        out: list[Claim] = []
        for claim in claims:
            resolved, claim_issues = resolve_claim(claim, store)
            issues.extend(claim_issues)
            out.append(resolved)
        return out

    contradictions: list[Contradiction] = []
    for contradiction in report.contradictions:
        resolved_c, c_issues = resolve_contradiction(contradiction, store)
        issues.extend(c_issues)
        contradictions.append(resolved_c)

    resolved_report = report.model_copy(
        update={
            "summary_claims": resolve_all(report.summary_claims),
            "key_findings": resolve_all(report.key_findings),
            "sections": [
                section.model_copy(update={"claims": resolve_all(section.claims)})
                for section in report.sections
            ],
            "contradictions": contradictions,
        }
    )
    return resolved_report, issues


def verify_structure(
    report: ResearchReport,
    store: EvidenceStore,
    *,
    resolution_issues: list[CitationIssue] | None = None,
) -> CitationVerification:
    """Run every check that needs no model.

    Expects an already-resolved report, so it measures what the reader will
    actually see.
    """
    result = CitationVerification(issues=list(resolution_issues or []))
    used_sources: set[str] = set()
    known_source_ids = {s.id for s in store.usable_sources()}

    for claim in report.all_claims():
        result.total_claims += 1
        if not claim.requires_evidence:
            # Framing text asserts nothing. Declared by kind rather than
            # guessed at by a keyword heuristic over the sentence.
            continue

        result.substantive_claims += 1
        result.total_evidence_refs += len(claim.evidence_ids)
        result.resolvable_evidence_refs += len(claim.evidence_ids)
        result.total_citations += len(claim.citation_ids)

        for source_id in claim.citation_ids:
            if source_id in known_source_ids:
                result.resolvable_citations += 1
                used_sources.add(source_id)
            else:
                result.issues.append(
                    CitationIssue(
                        type=CitationIssueType.UNKNOWN_SOURCE,
                        severity="error",
                        claim_text=claim.text[:200],
                        citation_id=source_id,
                        detail=f"{source_id} was never retrieved in this run",
                    )
                )

        if not claim.evidence_ids:
            result.issues.append(
                CitationIssue(
                    type=CitationIssueType.UNCITED_CLAIM,
                    severity="warning",
                    claim_text=claim.text[:200],
                    detail=f"{claim.kind.value} claim carries no evidence",
                )
            )

    for contradiction in report.contradictions:
        result.contradictions_total += 1
        if contradiction.is_auditable:
            result.contradictions_auditable += 1
        for source_id in (
            *contradiction.left_citation_ids,
            *contradiction.right_citation_ids,
        ):
            used_sources.add(source_id)

    unused = sorted(known_source_ids - used_sources, key=_source_sort_key)
    result.unused_source_ids = unused
    for source_id in unused:
        result.issues.append(
            CitationIssue(
                type=CitationIssueType.UNUSED_SOURCE,
                severity="info",
                citation_id=source_id,
                detail="retrieved but never cited",
            )
        )

    # Unresolved references were already dropped during resolution, so count
    # them back into the denominator; otherwise integrity would read 100%
    # precisely because the broken references were removed.
    dropped = sum(
        1
        for issue in result.issues
        if issue.type
        in (
            CitationIssueType.UNKNOWN_EVIDENCE,
            CitationIssueType.UNCITABLE_EVIDENCE,
        )
    )
    result.total_evidence_refs += dropped
    return result


def _source_sort_key(source_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in source_id.split("-")[0] if ch.isdigit())
    return (int(digits) if digits else 0, source_id)
