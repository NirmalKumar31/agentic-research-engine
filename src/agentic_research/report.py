"""Markdown rendering of a finished run.

Kept separate from synthesis so that presentation can change without touching
the reasoning pipeline, and so a stored run can be re-rendered from its JSON
artifacts without re-running any model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime

from agentic_research.comparison import ComparisonPair, render_pairs
from agentic_research.evidence.store import EvidenceStore
from agentic_research.metrics import RunMetrics
from agentic_research.models import (
    CitationVerification,
    Claim,
    ClaimKind,
    EvidenceItem,
    ResearchReport,
    SourceDocument,
)

# Enough to be useful, bounded so a failed run does not emit a wall of
# quotations that nobody reads.
MAX_EXCERPTS = 12

_WHITESPACE = re.compile(r"\s+")


def _dedupe_key(text: str) -> str:
    """Identity for "the reader has already read this sentence".

    Exact wording after normalising whitespace, case and the trailing
    period. Deliberately not fuzzy: suppressing a *paraphrase* would hide
    a claim whose evidence differs, and two claims that merely resemble
    each other are reported by the coverage assessment instead.
    """
    return _WHITESPACE.sub(" ", (text or "").strip().rstrip(".").lower())


def _unrendered(claims: Sequence[Claim], already: set[str]) -> list[Claim]:
    """The claims in order, minus any the reader has already been shown.

    ``already`` is updated, so a claim repeated inside one section is
    dropped on its second appearance too.
    """
    kept: list[Claim] = []
    for claim in claims:
        key = _dedupe_key(claim.text)
        if key in already:
            continue
        already.add(key)
        kept.append(claim)
    return kept


def render_markdown(
    report: ResearchReport,
    sources: list[SourceDocument],
    verification: CitationVerification | None,
    metrics: RunMetrics | None = None,
    evidence: Sequence[EvidenceItem] | None = None,
    comparison_pairs: Sequence[ComparisonPair] = (),
) -> str:
    """Render the report, its sources, and an honest verification footer.

    ``comparison_pairs`` are complete contrasts assembled from verified
    side claims. They are rendered as a table rather than prose because
    a sentence joining two claims would be new text no quote was
    checked against -- see :mod:`agentic_research.comparison`.
    """
    store = EvidenceStore(sources, list(evidence or []))
    cited = report.cited_ids()

    # Every claim is rendered once, in the richest place it appears.
    #
    # `summary_claims`, `key_findings`, `sections[].claims` and
    # `comparison_pairs` are four views over one pool of claims, and
    # nothing used to reconcile them: a live run printed four claims
    # eight times -- the Summary repeating the first table verbatim and
    # a section repeating the second -- and the coverage assessment
    # filed the repetition as a *limitation* rather than suppressing it.
    #
    # The table claims the duplicates. It carries the axis label and
    # attributes each sentence to its subject, so it says strictly more
    # than the same sentence standing alone, and a pair that lost a cell
    # to an earlier heading would no longer be a contrast. A heading
    # left with nothing to show is dropped entirely rather than printed
    # empty.
    rendered: set[str] = {
        _dedupe_key(side.text) for pair in comparison_pairs or () for side in pair.sides
    }

    lines: list[str] = [f"# {report.title}", ""]
    # The question in full, once, under the title. A title is a label and
    # gets shortened; the question is the thing the report answers and
    # must never be delivered half-finished.
    if metrics is not None and metrics.query:
        lines += [f"**Question:** {metrics.query}", ""]

    summary_claims = _unrendered(report.summary_claims, rendered)
    if summary_claims:
        lines += ["## Summary", ""]
        # Rendered as prose, but each sentence is a verified Claim carrying
        # its own evidence, not an unchecked blob.
        lines.append(" ".join(_claim_text(c, store) for c in summary_claims))
        lines.append("")

    key_findings = _unrendered(report.key_findings, rendered)
    if key_findings:
        lines += ["## Key findings", ""]
        for claim in key_findings:
            lines.append(f"- {_claim_text(claim, store)}")
        lines.append("")

    for section in report.sections:
        section_claims = _unrendered(section.claims, rendered)
        if not section_claims:
            continue
        lines += [f"## {section.heading}", ""]
        for claim in section_claims:
            lines.append(_claim_text(claim, store))
            lines.append("")

    if comparison_pairs:
        lines += [
            "## How they differ",
            "",
            "_Each cell is a published claim with its own verified evidence. "
            "The comparison is the arrangement; no sentence here was written "
            "to join them._",
            "",
            render_pairs(comparison_pairs),
        ]

    if _needs_evidence_fallback(report, verification):
        lines += _evidence_only_section(store)

    if report.contradictions:
        lines += [
            "## Where sources disagree",
            "",
            "_Preserved rather than resolved; the underlying sources genuinely differ._",
            "",
        ]
        for contradiction in report.contradictions:
            left = _markers(contradiction.left_citation_ids)
            right = _markers(contradiction.right_citation_ids)
            flag = "" if contradiction.is_auditable else " _(one side unevidenced)_"
            lines.append(
                f"- **{contradiction.topic}** — {contradiction.left_summary.rstrip('.')}"
                f" {left}, while {contradiction.right_summary.rstrip('.')} {right}.{flag}"
            )
        lines.append("")

    if report.limitations:
        lines += ["## Limitations", ""]
        lines += [f"- {item}" for item in report.limitations]
        lines.append("")

    usable = [s for s in sources if s.is_usable and not s.duplicate_of]
    if usable:
        lines += ["## Sources", ""]
        for source in sorted(usable, key=_source_order):
            date = source.published_date.date().isoformat() if source.published_date else "n.d."
            # "retrieved, not cited" covers two different failures and
            # named neither. A source whose text yielded no citable quote
            # is an extraction problem; one that yielded several and was
            # passed over is a selection problem. They need opposite
            # fixes, and the live case that prompted this -- AWS
            # Prescriptive Guidance on LangChain and LangGraph, 0.97,
            # official docs, discussing *both* subjects where first-party
            # documentation structurally cannot -- was undiagnosable from
            # the report alone.
            used = ""
            if source.id not in cited:
                quotes = len([e for e in store.citable_evidence() if e.source_id == source.id])
                used = (
                    f" _(retrieved, not cited — {quotes} citable quote"
                    f"{'s' if quotes != 1 else ''} extracted)_"
                    if quotes
                    else " _(retrieved, no citable quote could be extracted)_"
                )
            lines.append(
                f"- **[{source.id}]** [{source.title}]({source.url}) — "
                f"{source.domain}, {source.source_type.value}, {date}, "
                f"quality {source.quality_score:.2f}{used}"
            )
        lines.append("")

    if verification is not None:
        lines += _verification_section(verification)

    if metrics is not None:
        lines += _metrics_section(metrics)

    lines += [
        "---",
        "",
        f"_Generated by Agentic Research Engine on "
        f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}._",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _needs_evidence_fallback(
    report: ResearchReport, verification: CitationVerification | None
) -> bool:
    """True when nothing survived the gate but real evidence exists.

    A report whose every claim was withheld is correct and useless: the
    reader gets a title and a source list. The evidence is still there
    and still verified as verbatim, so it is shown as itself rather than
    left out because no synthesis passed.
    """
    if verification is None:
        return False
    return not report.substantive_claims()


def _evidence_only_section(store: EvidenceStore) -> list[str]:
    """Exact source excerpts, presented as excerpts and nothing more.

    These are not claims and are never counted as claims. Nothing here
    is synthesised, generalised or joined up -- each line is the
    source's own words with the source named, which is the one thing
    this system can still assert when synthesis fails verification.
    """
    citable = store.citable_evidence()
    if not citable:
        return []

    lines = [
        "## Source excerpts",
        "",
        "_No synthesized claim passed evidence verification. Showing exact "
        "source excerpts instead. These are verbatim quotations, not findings, "
        "and no conclusion has been drawn from them._",
        "",
    ]
    for item in citable[:MAX_EXCERPTS]:
        source = store.source(item.source_id)
        title = (source.title if source else "") or "unknown source"
        page = f", p. {item.page}" if item.page else ""
        lines.append(f'- "{item.quote}" — **[{item.source_id}]** {title}{page}')
    lines.append("")
    return lines


def _markers(citation_ids: list[str], pages: dict[str, int] | None = None) -> str:
    """Render citation markers, with page numbers where genuinely known."""
    pages = pages or {}
    parts = []
    for cid in citation_ids:
        page = pages.get(cid)
        parts.append(f"[{cid}, p. {page}]" if page else f"[{cid}]")
    return "".join(parts)


def _claim_text(claim: Claim, store: EvidenceStore | None = None) -> str:
    """Render a claim with markers derived from its resolved citations.

    Markers are generated from ``citation_ids``, which the engine derived from
    ``evidence_ids``. Nothing bracket-shaped that a model typed into the prose
    survives to this point, so every marker a reader sees was verified.
    """
    text = claim.text.strip()
    pages: dict[str, int] = {}
    if store is not None:
        for evidence_id in claim.evidence_ids:
            item = store.evidence_by_id(evidence_id)
            if item is not None and item.page:
                pages.setdefault(item.source_id, item.page)
    markers = _markers(claim.citation_ids, pages)
    if markers:
        text = f"{text.rstrip('.')}. {markers}"
    if claim.kind is ClaimKind.SYNTHESIS:
        text = f"{text} *(synthesis)*"
    return text


def _source_order(source: SourceDocument) -> tuple[int, str]:
    digits = "".join(ch for ch in source.id if ch.isdigit())
    return (int(digits) if digits else 0, source.id)


def _verification_section(verification: CitationVerification) -> list[str]:
    lines = ["## Citation verification", ""]
    # A published report with no claims has no references to get wrong,
    # so the rate is 1.0 by construction. Printing "100%" there reads as a
    # quality result for a document that asserts nothing.
    rate = (
        f"{verification.evidence_integrity_rate:.0%}"
        if verification.has_evidence_references
        else "n/a, no references"
    )
    lines.append(
        f"- Evidence references: {verification.total_evidence_refs}, "
        f"{verification.resolvable_evidence_refs} resolved to citable evidence "
        f"({rate}). "
        "Citation markers are derived from those references by the engine, so "
        "citation integrity is a structural invariant rather than a measurement."
    )
    lines.append(
        "- Evidence-owing claims carrying a citation: "
        + (
            f"{verification.citation_coverage_rate:.0%} of {verification.substantive_claims}"
            if verification.substantive_claims
            else "n/a, no substantive claims were published"
        )
    )
    if verification.checked_claims:
        breakdown = verification.support_breakdown
        scope = (
            "every eligible claim"
            if verification.entailment_exhaustive
            else (
                f"a sample of {verification.checked_claims} of "
                f"{verification.checkable_claims} eligible claims"
            )
        )
        lines.append(
            f"- Entailment checked against each claim's own evidence, over {scope}: "
            f"{breakdown['supported']} supported, "
            f"{breakdown['partially_supported']} partially supported, "
            f"{breakdown['unsupported']} unsupported, "
            f"{breakdown['not_checked']} not checked"
        )
    if verification.contradictions_total:
        lines.append(
            f"- Contradictions: {verification.contradictions_auditable} of "
            f"{verification.contradictions_total} have citable evidence on both sides"
        )
    if verification.repaired:
        lines.append(
            "- References to nonexistent or unverifiable evidence were removed "
            "automatically; the affected sentences now appear without a citation."
        )
    if verification.unused_source_ids:
        lines.append(f"- Retrieved but never cited: {', '.join(verification.unused_source_ids)}")
    problems = [i for i in verification.issues if i.severity in ("error", "warning")]
    if problems:
        lines += ["", "<details><summary>Open citation issues</summary>", ""]
        for issue in problems[:20]:
            lines.append(f"- `{issue.type.value}` {issue.detail} — {issue.claim_text[:120]}")
        lines += ["", "</details>"]
    lines.append("")
    return lines


def _metrics_section(metrics: RunMetrics) -> list[str]:
    lines = ["## Run metrics", "", "| Metric | Value |", "| --- | --- |"]
    rows = [
        ("Mode", metrics.mode),
        ("Research rounds", metrics.research_rounds),
        ("Stopped because", metrics.stop_reason or "coverage sufficient"),
        ("Search queries", metrics.search_queries),
        ("Unique sources", metrics.unique_sources),
        ("Usable sources", metrics.usable_sources),
        ("Distinct domains", metrics.distinct_domains),
        ("Fetches avoided by dedup", metrics.fetches_avoided),
        ("Evidence items", metrics.evidence_items),
        ("Quotes verbatim (exact-normalised)", f"{metrics.quote_fidelity_rate:.0%}"),
        ("Quotes fuzzy (excluded from citation)", f"{metrics.fuzzy_quote_rate:.0%}"),
        ("LLM calls", metrics.llm_calls),
        ("Tokens (in/out)", f"{metrics.input_tokens:,} / {metrics.output_tokens:,}"),
        ("Estimated cost", metrics.cost_display),
        ("Duration", f"{metrics.duration_s:.1f}s"),
    ]
    lines += [f"| {label} | {value} |" for label, value in rows]
    lines.append("")
    return lines
