"""Citation extraction, validation and repair.

Checks run cheapest-first, which is also strongest-first:

1. **Structural** (free, deterministic) — does every marker point at a source
   that was actually retrieved? Does every factual claim carry a marker? Which
   retrieved sources were never used? These are decidable by lookup, and a
   failure here is always a real failure.
2. **Entailment** (one model call per claim) — does the cited evidence
   actually support the claim it is attached to? This is a judgement, it costs
   money, and it is sampled rather than exhaustive.

Running structural checks first means the expensive pass never gets spent on a
claim already known to be broken.
"""

from __future__ import annotations

import re

from agentic_research.models import (
    CitationIssue,
    CitationIssueType,
    CitationVerification,
    Claim,
    ResearchReport,
)

# Matches [S3] and the [S1, S2] form models sometimes produce despite the
# prompt asking for [S1][S2].
_MARKER = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")

# A claim with none of these is usually a framing or transition sentence
# rather than an assertion needing a source.
_FACTUAL_HINTS = re.compile(
    r"\b(\d|percent|%|faster|slower|cheaper|more|less|most|least|best|worst|"
    r"increase|decrease|reduce|improve|outperform|require|support|provide|"
    r"report|show|found|demonstrate|according|study|benchmark|release|version)\b",
    re.IGNORECASE,
)


def extract_markers(text: str) -> list[str]:
    """Pull source ids out of a claim's citation markers, in order."""
    found: list[str] = []
    for group in _MARKER.findall(text):
        for token in group.split(","):
            token = token.strip()
            if token and token not in found:
                found.append(token)
    return found


def strip_markers(text: str) -> str:
    return _MARKER.sub("", text).replace("  ", " ").strip()


def looks_factual(claim: Claim) -> bool:
    """Heuristic: does this claim assert something a source should back?

    Imperfect by nature. It is used to decide what to *warn* about, never to
    silently delete a claim, so a false positive costs a warning rather than
    content.
    """
    if claim.is_interpretation:
        return False
    text = strip_markers(claim.text)
    if len(text.split()) < 5:
        return False
    return bool(_FACTUAL_HINTS.search(text))


def verify_structure(report: ResearchReport, known_source_ids: set[str]) -> CitationVerification:
    """Run every check that needs no model.

    Marker ids are normalised onto ``Claim.citation_ids`` as a side effect, so
    later stages can work with the parsed list instead of re-parsing prose.
    """
    result = CitationVerification()
    used: set[str] = set()

    for claim in report.all_claims():
        result.total_claims += 1
        # Citations may arrive as a schema field, as prose markers, or both.
        markers = list(claim.citation_ids)
        for marker in extract_markers(claim.text):
            if marker not in markers:
                markers.append(marker)
        claim.citation_ids = markers
        factual = looks_factual(claim)
        if factual:
            result.factual_claims += 1

        for source_id in markers:
            result.total_citations += 1
            if source_id in known_source_ids:
                result.valid_citations += 1
                used.add(source_id)
            else:
                # The failure this whole subsystem exists to prevent.
                result.issues.append(
                    CitationIssue(
                        type=CitationIssueType.UNKNOWN_SOURCE,
                        severity="error",
                        claim_text=claim.text[:200],
                        citation_id=source_id,
                        detail=f"{source_id} was never retrieved in this run",
                    )
                )

        if factual and not markers:
            result.issues.append(
                CitationIssue(
                    type=CitationIssueType.UNCITED_CLAIM,
                    severity="warning",
                    claim_text=claim.text[:200],
                    detail="factual claim with no citation",
                )
            )

        if len(markers) != len(set(markers)):
            result.issues.append(
                CitationIssue(
                    type=CitationIssueType.REDUNDANT_CITATION,
                    severity="info",
                    claim_text=claim.text[:200],
                    detail="the same source is cited more than once in one claim",
                )
            )

    unused = sorted(known_source_ids - used, key=_source_sort_key)
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
    return result


def _source_sort_key(source_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in source_id if ch.isdigit())
    return (int(digits) if digits else 0, source_id)


def repair_report(report: ResearchReport, known_source_ids: set[str]) -> tuple[ResearchReport, int]:
    """Remove citation markers that point at sources we do not have.

    Deleting the marker rather than the sentence is the conservative choice:
    the sentence may well be true and supported by evidence the model failed
    to cite properly, but a dangling reference is always wrong. The stripped
    claim then shows up as an uncited claim on re-verification, which is an
    honest description of its state.
    """
    repaired = 0

    def fix(claim: Claim) -> Claim:
        nonlocal repaired
        present = list(claim.citation_ids) + [
            m for m in extract_markers(claim.text) if m not in claim.citation_ids
        ]
        bad = [cid for cid in present if cid not in known_source_ids]
        if not bad:
            return claim
        text = claim.text
        for source_id in bad:
            text = re.sub(rf"\[\s*{re.escape(source_id)}\s*\]", "", text)
            text = re.sub(
                rf"\[([^\]]*?),?\s*{re.escape(source_id)}\s*,?([^\]]*?)\]",
                lambda m: (
                    f"[{m.group(1).strip(', ')}{m.group(2).strip(', ')}]"
                    if (m.group(1).strip(", ") or m.group(2).strip(", "))
                    else ""
                ),
                text,
            )
            repaired += 1
        text = re.sub(r"\s{2,}", " ", text).replace(" .", ".").strip()
        kept = [c for c in present if c in known_source_ids]
        return claim.model_copy(update={"text": text, "citation_ids": kept})

    fixed = report.model_copy(
        update={
            "key_findings": [fix(c) for c in report.key_findings],
            "sections": [
                section.model_copy(update={"claims": [fix(c) for c in section.claims]})
                for section in report.sections
            ],
        }
    )
    return fixed, repaired
