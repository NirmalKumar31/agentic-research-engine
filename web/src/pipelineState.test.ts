import { describe, expect, it } from "vitest";

import { advance, emptyPipeline, type PipelineState } from "./pipelineState";
import type { ProgressEvent } from "./types";

/**
 * Pipeline reducer behaviour, pinned against two defects a real live run
 * exposed.
 *
 * A run fetched 4 of 6 sources -- one publisher returned 403 to an
 * automated request -- and the Retrieve stage rendered as failed, telling
 * the visitor retrieval had failed when it had produced the evidence the
 * report was built from.
 *
 * The same run rendered "0/0 covered" for a six-dimension plan, because
 * the denominator was reconstructed as covered/ratio and neither is
 * usable when nothing is covered.
 */

function fold(events: ProgressEvent[]): PipelineState {
  return events.reduce(advance, emptyPipeline());
}

const ev = (event: string, rest: Record<string, unknown> = {}): ProgressEvent =>
  ({ event, ...rest }) as ProgressEvent;

describe("partial retrieval is a warning, not a failure", () => {
  it("marks the stage warning rather than failed", () => {
    const state = fold([ev("source_failed", { source_id: "S2", status: "http_403" })]);
    expect(state.states.retrieve).toBe("warning");
    expect(state.states.retrieve).not.toBe("failed");
  });

  it("counts each failed item", () => {
    const state = fold([
      ev("source_failed", { source_id: "S2" }),
      ev("source_failed", { source_id: "S4" }),
    ]);
    expect(state.issues.retrieve).toBe(2);
  });

  it("lets the stage complete once usable sources are registered", () => {
    const state = fold([
      ev("sources_deduplicated", { results: 48, unique: 30, selected: 6, avoided: 3 }),
      ev("source_failed", { source_id: "S2" }),
      ev("source_failed", { source_id: "S4" }),
      ev("sources_registered", { usable: 4 }),
      ev("evidence_extracted", { source_id: "S1", items: 3 }),
    ]);
    // Reaching Extract completes Retrieve, but the warning is retained.
    expect(state.states.extract).toBe("active");
    expect(state.states.retrieve).toBe("warning");
    expect(state.counts.sources).toBe(4);
    expect(state.counts.sourcesAttempted).toBe(6);
  });

  it("does not stop later stages from running", () => {
    const state = fold([
      ev("source_failed", { source_id: "S2" }),
      ev("sources_registered", { usable: 4 }),
      ev("synthesizing", { evidence: 6 }),
      ev("citations_verified", { total: 6 }),
      ev("completed", { stop_reason: "coverage judged sufficient" }),
    ]);
    expect(state.states.synthesize).toBe("done");
    expect(state.states.verify).toBe("done");
    expect(state.finished).toBe(true);
  });

  it("treats a failed search the same way", () => {
    const state = fold([
      ev("queries_generated", { count: 6 }),
      ev("search_failed", { query_id: "Q3" }),
      ev("search_completed", { query_id: "Q1", query: "x", results: 8 }),
    ]);
    expect(state.states.search).toBe("warning");
    expect(state.issues.search).toBe(1);
  });

  it("keeps a warning visible through completion", () => {
    const state = fold([
      ev("source_failed", { source_id: "S2" }),
      ev("completed", { stop_reason: "done" }),
    ]);
    expect(state.states.retrieve).toBe("warning");
  });
});

describe("coverage denominator", () => {
  it("uses the total the engine reports", () => {
    const state = fold([ev("coverage_evaluated", { covered: 2, total: 6, ratio: 0.33 })]);
    expect(state.counts.coverage).toEqual({ covered: 2, total: 6 });
  });

  it("reports 0/6 rather than 0/0 when nothing is covered", () => {
    const state = fold([
      ev("plan_generated", { count: 6 }),
      ev("coverage_evaluated", { covered: 0, total: 6, ratio: 0 }),
    ]);
    expect(state.counts.coverage).toEqual({ covered: 0, total: 6 });
  });

  it("falls back to the plan size for recordings predating the field", () => {
    const state = fold([
      ev("plan_generated", { count: 6 }),
      ev("coverage_evaluated", { covered: 0, ratio: 0 }),
    ]);
    expect(state.counts.coverage).toEqual({ covered: 0, total: 6 });
  });

  it("still derives from the ratio when neither is available", () => {
    const state = fold([ev("coverage_evaluated", { covered: 3, ratio: 0.5 })]);
    expect(state.counts.coverage).toEqual({ covered: 3, total: 6 });
  });

  it("does not invent a denominator when there is genuinely nothing", () => {
    const state = fold([ev("coverage_evaluated", { covered: 0, total: 0, ratio: 0 })]);
    expect(state.counts.coverage).toEqual({ covered: 0, total: 0 });
  });
});

describe("heartbeats and unknown events do not advance the pipeline", () => {
  it("ignores an event belonging to no stage", () => {
    const before = fold([ev("plan_generated", { count: 6 })]);
    const after = advance(before, ev("something_unmapped"));
    expect(after.states).toEqual(before.states);
    expect(after.counts).toEqual(before.counts);
  });
});

describe("the retrieve denominator is what was attempted", () => {
  it("counts selected pages, not every unique URL found", () => {
    // 30 distinct URLs across the searches, 6 chosen within budget.
    // Retrieve is judged on the 6 it tried, not the 30 it saw.
    const state = fold([
      ev("sources_deduplicated", { results: 48, unique: 30, selected: 6, avoided: 3 }),
      ev("source_failed", { source_id: "S2" }),
      ev("sources_registered", { usable: 5 }),
    ]);
    expect(state.counts.sourcesAttempted).toBe(6);
    expect(state.counts.sources).toBe(5);
  });

  it("falls back to unique for recordings without selected", () => {
    const state = fold([ev("sources_deduplicated", { results: 48, unique: 6, avoided: 3 })]);
    expect(state.counts.sourcesAttempted).toBe(6);
  });
});
