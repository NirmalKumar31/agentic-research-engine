import { describe, expect, it } from "vitest";

import { evidenceOnlyExcerpts, MAX_EXCERPTS } from "./fallback";
import type { Claim, Evidence, Report } from "./types";

/**
 * When the publication gate withholds every claim, the report falls back
 * to the sources' own words.
 *
 * The risk is not that excerpts fail to appear — it is that they appear
 * next to findings, or instead of them, and a reader takes a verbatim
 * quotation for something the system concluded. These fix that boundary.
 */

const report = (claims: Partial<Report> = {}): Report => ({
  title: "T",
  summary_claims: [],
  key_findings: [],
  sections: [],
  contradictions: [],
  limitations: [],
  ...claims,
});

const claim = (text: string): Claim =>
  ({ text, kind: "factual", evidence_ids: ["S1-e1"], citation_ids: ["S1"] }) as Claim;

const evidence = (id: string, match = "exact_normalized"): Evidence =>
  ({
    id,
    source_id: "S1",
    quote: `quote ${id}`,
    quote_match: match,
    page: null,
  }) as Evidence;

describe("evidence-only fallback", () => {
  it("shows excerpts when no claim survived", () => {
    const out = evidenceOnlyExcerpts(report(), [evidence("S1-e1")]);
    expect(out).toHaveLength(1);
  });

  it("shows nothing when a claim was published", () => {
    const out = evidenceOnlyExcerpts(report({ key_findings: [claim("A finding.")] }), [
      evidence("S1-e1"),
    ]);
    expect(out).toEqual([]);
  });

  it("counts claims from every part of the report", () => {
    const withSection = report({
      sections: [{ heading: "H", claims: [claim("A sectioned claim.")] }],
    });
    expect(evidenceOnlyExcerpts(withSection, [evidence("S1-e1")])).toEqual([]);
  });

  it("shows nothing when no quote is citable", () => {
    const out = evidenceOnlyExcerpts(report(), [
      evidence("S1-e1", "fuzzy"),
      evidence("S1-e2", "none"),
    ]);
    expect(out).toEqual([]);
  });

  it("excludes unmatched quotes but keeps exact ones", () => {
    const out = evidenceOnlyExcerpts(report(), [
      evidence("S1-e1", "fuzzy"),
      evidence("S1-e2"),
    ]);
    expect(out.map((e) => e.id)).toEqual(["S1-e2"]);
  });

  it("caps how many excerpts are shown", () => {
    const many = Array.from({ length: 40 }, (_, i) => evidence(`S1-e${i}`));
    expect(evidenceOnlyExcerpts(report(), many)).toHaveLength(MAX_EXCERPTS);
  });

  it("handles a missing report without throwing", () => {
    expect(evidenceOnlyExcerpts(null, [evidence("S1-e1")])).toEqual([]);
  });
});
