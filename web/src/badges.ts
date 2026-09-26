import type { ExampleSummary } from "./api";

/**
 * Capability badges, derived from what a recording actually contains.
 *
 * These used to be a lookup keyed by recording id, which asserted a
 * capability the artifact had to be trusted to still have. It stopped
 * being true: a re-recording put the page-level citations in a
 * different run, and the site went on advertising "PDF page
 * provenance" on a recording with none, because the badge described an
 * intention rather than a file.
 *
 * Anything measurable is counted from the payload. The theme label
 * stays keyed by id because it is a name, not a claim.
 */

/** Stable thematic names. Not capability assertions. */
const THEMES: Record<string, string> = {
  "rag-vector-vs-search": "Technical comparison",
  "nist-ai-risk-framework": "Standards framework",
  "fraud-detection-imbalanced": "Methods survey",
};

export function themeFor(example: ExampleSummary): string | null {
  return THEMES[example.id] ?? null;
}

/**
 * Measured capabilities, in the order they should be shown.
 *
 * Each entry is derived; none is asserted. A recording with no
 * page-level citations gets no page badge, which is the whole point.
 */
export function capabilitiesFor(example: ExampleSummary): string[] {
  const out: string[] = [];
  if (example.page_citation_count > 0) {
    const n = example.page_citation_count;
    out.push(`${n} page-level citation${n === 1 ? "" : "s"}`);
  }
  if (example.sources > 1) out.push(`${example.sources} sources`);
  if (example.published_claims > 0) {
    const n = example.published_claims;
    out.push(`${n} verified claim${n === 1 ? "" : "s"}`);
  } else {
    out.push("All claims withheld");
  }
  return out;
}
