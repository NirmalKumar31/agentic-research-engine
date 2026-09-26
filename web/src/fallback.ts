import type { Evidence, Report } from "./types";

/** Most excerpts worth showing. Matches MAX_EXCERPTS in report.py. */
export const MAX_EXCERPTS = 12;

/**
 * Every substantive claim in a report, in reading order.
 */
export function substantiveClaims(report: Report) {
  return [
    ...report.summary_claims,
    ...report.key_findings,
    ...report.sections.flatMap((s) => s.claims),
  ];
}

/**
 * The excerpts to show when the publication gate withheld everything.
 *
 * A report whose every claim failed verification is correct and
 * unreadable: a title and a source list. The evidence is still there and
 * still verified verbatim, so it is shown as itself.
 *
 * Empty whenever a claim survived, so excerpts never appear alongside
 * findings — a reader must not have to work out which of the two they
 * are looking at. Only exact-normalized quotes qualify: a quote that
 * could not be matched to its source is not something to show as the
 * source's own words.
 */
export function evidenceOnlyExcerpts(
  report: Report | null,
  evidence: Evidence[],
): Evidence[] {
  if (!report || substantiveClaims(report).length > 0) return [];
  return evidence
    .filter((e) => e.quote_match === "exact_normalized")
    .slice(0, MAX_EXCERPTS);
}
