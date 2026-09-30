import { describe, expect, it } from "vitest";
import { answered, slotStatuses } from "./contract";
import type { AnswerCoverage, Contract, Report } from "./types";

/**
 * The panel must not contradict the report.
 *
 * A comparison can be answered two ways the client cannot re-derive:
 * claims about each subject meeting on a named axis, and a relationship
 * claim establishing there is no contrast to find. Both depend on what
 * the claims assert rather than which slot they declared, so neither is
 * visible from the contract.
 *
 * This interface has shipped the failure once. On v1.2.1 a run
 * published a claim filling `relationship`, the engine's own
 * limitations therefore did not say the question was unanswered, and
 * the panel — recomputing from the contract alone — rendered "this
 * report does not answer the question" directly above a report saying
 * the opposite.
 *
 * A mutation sweep found this path had no test at all: disabling the
 * client's use of the engine's assessment left every frontend test
 * passing.
 */

const contract: Contract = {
  question: "How does a large language model differ from a neural network?",
  question_type: "comparison",
  entities: ["large language model", "neural network"],
  comparison_subjects: ["large language model", "neural network"],
  shape_source: "wording",
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
      satisfied_by: [],
    },
    {
      name: "relationship",
      description: "How the subjects relate",
      core: false,
      satisfied_by: [],
    },
  ],
} as unknown as Contract;

const report = (slot: string): Report =>
  ({
    title: "T",
    summary_claims: [
      {
        text: "A large language model is a kind of neural network.",
        kind: "finding",
        answer_slot: slot,
        evidence_ids: [],
        citation_ids: [],
      },
    ],
    key_findings: [],
    sections: [],
    limitations: [],
    contradictions: [],
  }) as unknown as Report;

const coverage = (over: Partial<AnswerCoverage> = {}): AnswerCoverage => ({
  satisfied_slots: [],
  answered: false,
  absent_entities: [],
  relationship_discharge: "",
  ...over,
});

describe("the panel follows the engine's own assessment", () => {
  it("shows the core slot filled when the engine discharged it", () => {
    const statuses = slotStatuses(
      contract,
      report("relationship"),
      coverage({
        satisfied_slots: ["direct_contrast", "relationship"],
        answered: true,
        relationship_discharge: "subtype",
      }),
    );
    const core = statuses.find((s) => s.core)!;
    expect(core.name).toBe("direct_contrast");
    expect(core.filled).toBe(true);
    expect(core.filledBy).toBe("subtype");
  });

  it("reports the question answered, matching the report", () => {
    expect(
      answered(
        contract,
        report("relationship"),
        coverage({
          satisfied_slots: ["direct_contrast", "relationship"],
          answered: true,
          relationship_discharge: "subtype",
        }),
      ),
    ).toBe(true);
  });

  it("does not invent a discharge the engine did not make", () => {
    // Non-vacuity: a vague relationship claim must still read as
    // unanswered, or the client would be rubber-stamping instead of
    // following.
    expect(
      answered(
        contract,
        report("relationship"),
        coverage({ satisfied_slots: ["relationship"] }),
      ),
    ).toBe(false);
  });

  it("falls back to deriving from claims when no coverage is supplied", () => {
    // Older recordings carry no assessment. The panel must still work,
    // and a claim that directly declares the core slot fills it.
    expect(answered(contract, report("direct_contrast"), null)).toBe(true);
    expect(answered(contract, report("relationship"), null)).toBe(false);
  });

  it("names the route when a pair rather than a relationship answered it", () => {
    const statuses = slotStatuses(
      contract,
      report("knowledge_update"),
      coverage({ satisfied_slots: ["direct_contrast"], answered: true }),
    );
    expect(statuses.find((s) => s.core)!.filledBy).toBe("verified side claims");
  });
});
