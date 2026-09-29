import { describe, expect, it } from "vitest";
import { answered, declaredSlots, slotStatuses } from "./contract";
import live from "./fixtures/live-v121.json";
import type { Contract, Report } from "./types";

/**
 * The panel, against a real hosted run rather than a fixture written to
 * make it pass.
 *
 * Captured from the deployed demo on v1.2.1 at commit 1d21b110, the
 * run committed under examples/live-validation/v121-20260929-024544.
 * Its interest is the case that motivated the fix: the claim declaring
 * `direct_contrast` was withheld on entailment, and the one declaring
 * `relationship` published -- so the core requirement is discharged by
 * an alternative, and the report answers the question without ever
 * stating a contrast.
 */

const contract = live.contract as Contract;
const report = live.report as Report;

describe("the panel against a real hosted run", () => {
  it("reads the slots the run actually published", () => {
    expect(declaredSlots(report).sort()).toEqual(["dimension", "relationship"]);
  });

  it("reports the question as answered, via the relationship slot", () => {
    const core = slotStatuses(contract, report).find((s) => s.core)!;
    expect(core.name).toBe("direct_contrast");
    expect(core.filled).toBe(true);
    expect(core.filledBy).toBe("relationship");
    expect(answered(contract, report)).toBe(true);
  });

  it("agrees with the engine, which published no such limitation", () => {
    // The engine computes coverage independently for the report's
    // limitations. If the panel and the engine ever disagreed, the page
    // would contradict the report printed beneath it.
    const blunt = report.limitations.filter((l) =>
      l.includes("did not answer the question"),
    );
    expect(blunt).toEqual([]);
    expect(answered(contract, report)).toBe(true);
  });

  it("would report it unanswered without the relationship claim", () => {
    // Non-vacuity, done by removing the thing that carried the answer
    // rather than by asserting something is unfilled -- all three slots
    // are filled here, `direct_contrast` by its alternative, which is
    // the whole point.
    const withoutRelationship: Report = {
      ...report,
      summary_claims: report.summary_claims.filter(
        (c) => c.answer_slot !== "relationship",
      ),
      key_findings: report.key_findings.filter(
        (c) => c.answer_slot !== "relationship",
      ),
      sections: report.sections.map((s) => ({
        ...s,
        claims: s.claims.filter((c) => c.answer_slot !== "relationship"),
      })),
    };
    expect(declaredSlots(withoutRelationship)).not.toContain("relationship");
    expect(answered(contract, withoutRelationship)).toBe(false);
  });

  it("carries the alternative in the payload at all", () => {
    // The bug this fixture found: to_dict() dropped satisfied_by, so a
    // client could not see that a core slot had been discharged and
    // rendered "does not answer the question" under a report saying it
    // did.
    const core = contract.required_slots.find((s) => s.core)!;
    expect(core.satisfied_by).toEqual(["relationship"]);
  });
});
