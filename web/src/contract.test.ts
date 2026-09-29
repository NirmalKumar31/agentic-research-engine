import { describe, expect, it } from "vitest";
import { answered, declaredSlots, slotStatuses } from "./contract";
import type { Claim, Contract, Report } from "./types";

function claim(text: string, slot?: string): Claim {
  return {
    text,
    kind: "factual",
    answer_slot: slot,
    evidence_ids: ["S1-e1"],
    citation_ids: ["S1"],
  };
}

function report(claims: Claim[]): Report {
  return {
    title: "T",
    summary_claims: claims,
    key_findings: [],
    sections: [],
    contradictions: [],
    limitations: [],
  };
}

const COMPARISON: Contract = {
  question: "How does a large language model differ from a neural network?",
  question_type: "comparison",
  entities: ["large language model (LLM)", "neural network"],
  dimensions: [],
  constraints: [],
  ambiguities: [],
  usable: true,
  unusable_reason: "",
  required_slots: [
    {
      name: "direct_contrast",
      description: "An explicit statement of how the subjects differ",
      core: true,
      satisfied_by: ["relationship"],
    },
    { name: "dimension", description: "A named dimension", core: false },
    { name: "relationship", description: "How they relate", core: false },
  ],
};

describe("what the published report filled", () => {
  it("counts only slots a published claim declared", () => {
    expect(declaredSlots(report([claim("a", "dimension"), claim("b")]))).toEqual([
      "dimension",
    ]);
  });

  it("drops a blank declaration rather than treating it as a slot", () => {
    // Smaller local models omit the field on every claim. Such a claim
    // still publishes; it just fills nothing.
    expect(declaredSlots(report([claim("a", ""), claim("b")]))).toEqual([]);
  });

  it("reads claims from every part of the report", () => {
    const r = report([claim("summary", "dimension")]);
    r.key_findings = [claim("finding", "relationship")];
    r.sections = [{ heading: "S", claims: [claim("section", "direct_contrast")] }];
    expect(declaredSlots(r).sort()).toEqual([
      "dimension",
      "direct_contrast",
      "relationship",
    ]);
  });
});

describe("whether the question was answered", () => {
  it("a direct contrast answers a comparison", () => {
    expect(answered(COMPARISON, report([claim("x", "direct_contrast")]))).toBe(true);
  });

  it("a relationship also answers it, and says which slot arrived", () => {
    // The hosted acceptance case: the engine found that one subject is
    // a subset of the other, which is the answer to how they differ.
    const statuses = slotStatuses(COMPARISON, report([claim("x", "relationship")]));
    const core = statuses.find((s) => s.name === "direct_contrast")!;
    expect(core.filled).toBe(true);
    expect(core.filledBy).toBe("relationship");
    expect(answered(COMPARISON, report([claim("x", "relationship")]))).toBe(true);
  });

  it("dimensions alone do not", () => {
    // The failure the contract exists for: naming axes along which two
    // things differ, without saying how they differ or how they relate.
    const r = report([claim("x", "dimension"), claim("y", "dimension")]);
    expect(answered(COMPARISON, r)).toBe(false);
    expect(slotStatuses(COMPARISON, r).find((s) => s.core)!.filled).toBe(false);
  });

  it("publishing claims that declare nothing does not", () => {
    expect(answered(COMPARISON, report([claim("x"), claim("y")]))).toBe(false);
  });

  it("publishing nothing does not", () => {
    expect(answered(COMPARISON, report([]))).toBe(false);
  });

  it("a contract with no core slot is never answered", () => {
    // Guards against reading "no core slots" as "nothing missing".
    const shapeless: Contract = { ...COMPARISON, required_slots: [] };
    expect(answered(shapeless, report([claim("x", "dimension")]))).toBe(false);
  });

  it("every core slot must be filled, not just one", () => {
    // A multi-part question makes each part its own core slot.
    const multipart: Contract = {
      ...COMPARISON,
      question_type: "synthesis",
      required_slots: [
        { name: "part_1", description: "first", core: true },
        { name: "part_2", description: "second", core: true },
      ],
    };
    expect(answered(multipart, report([claim("x", "part_1")]))).toBe(false);
    expect(
      answered(multipart, report([claim("x", "part_1"), claim("y", "part_2")])),
    ).toBe(true);
  });
});
