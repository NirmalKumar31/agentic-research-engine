import { describe, expect, it } from "vitest";

import type { ExampleSummary } from "./api";
import { capabilitiesFor, themeFor } from "./badges";

/**
 * Capability badges must describe the artifact, not an intention.
 *
 * The previous design keyed them by recording id, so "PDF page
 * provenance" stayed on the NIST card after a re-recording put the
 * page-level citations somewhere else entirely. A reader had no way to
 * tell. These tests pin the badges to the payload.
 */

const example = (over: Partial<ExampleSummary> = {}): ExampleSummary => ({
  id: "nist-ai-risk-framework",
  question: "q",
  label: "L",
  description: "d",
  mode: "local",
  recorded_at: "",
  sources: 5,
  evidence_items: 30,
  citable_evidence: 24,
  page_citation_count: 0,
  has_page_provenance: false,
  published_claims: 8,
  duration_s: 900,
  ...over,
});

describe("capability badges are derived", () => {
  it("claims page provenance only when pages are actually cited", () => {
    expect(capabilitiesFor(example({ page_citation_count: 0 })).join(" ")).not.toContain("page");
    expect(capabilitiesFor(example({ page_citation_count: 5 }))).toContain(
      "5 page-level citations",
    );
  });

  it("does not assert page provenance from the recording id", () => {
    const nist = capabilitiesFor(example({ id: "nist-ai-risk-framework", page_citation_count: 0 }));
    expect(nist.join(" ")).not.toContain("page");
  });

  it("singularises a lone page citation", () => {
    expect(capabilitiesFor(example({ page_citation_count: 1 }))).toContain(
      "1 page-level citation",
    );
  });

  it("counts sources rather than asserting a number", () => {
    expect(capabilitiesFor(example({ sources: 3 }))).toContain("3 sources");
  });

  it("says plainly when everything was withheld", () => {
    expect(capabilitiesFor(example({ published_claims: 0 }))).toContain("All claims withheld");
  });

  it("reports the verified claim count when there is one", () => {
    expect(capabilitiesFor(example({ published_claims: 1 }))).toContain("1 verified claim");
  });
});

describe("theme labels", () => {
  it("are names, not capability claims", () => {
    for (const id of [
      "rag-vector-vs-search",
      "nist-ai-risk-framework",
      "fraud-detection-imbalanced",
    ]) {
      const theme = themeFor(example({ id }));
      expect(theme).toBeTruthy();
      expect(theme!.toLowerCase()).not.toMatch(/page|pdf|citation|verified|source/);
    }
  });

  it("returns null for an unknown recording", () => {
    expect(themeFor(example({ id: "something-new" }))).toBeNull();
  });
});
