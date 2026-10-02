import { describe, expect, it } from "vitest";
import telemetrySource from "./telemetry.ts?raw";
import panelSource from "./RunTelemetry.tsx?raw";
import {
  describeCounts,
  headlineStats,
  formatSeconds,
  formatUsd,
  stageBreakdown,
  telemetryGroups,
} from "./telemetry";
import type { Metrics } from "./types";

const metrics = (over: Partial<Metrics> = {}): Metrics =>
  ({
    duration_s: 168.7,
    research_rounds: 1,
    search_queries: 3,
    unique_sources: 6,
    usable_sources: 6,
    distinct_domains: 4,
    evidence_items: 23,
    exact_quotes: 23,
    fuzzy_quotes: 0,
    unmatched_quotes: 0,
    llm_calls: 14,
    input_tokens: 21645,
    output_tokens: 9646,
    known_cost_usd: 0.007,
    cost_is_complete: true,
    citation_integrity_rate: 1,
    evidence_integrity_rate: 1,
    citation_coverage_rate: 1,
    claim_support_rate: 1,
    support_breakdown: {},
    entailment_exhaustive: true,
    content_origins: {},
    model_assignments: { planner: "gpt-5-mini", synthesizer: "gpt-5" },
    mode: "cloud",
    stop_reason: "stopped after round 1",
    ...over,
  }) as Metrics;

const rowsOf = (m: Metrics, title: string) =>
  telemetryGroups(m).find((g) => g.title === title)?.rows ?? [];

const labels = (m: Metrics, title: string) => rowsOf(m, title).map((r) => r.label);

describe("a number the server did not send is omitted, never shown as zero", () => {
  // "0 cached tokens" and "this build did not record cached tokens" are
  // different facts. A metrics panel that blurs them is worse than none,
  // because a reader cannot tell which they are looking at.
  it("omits optional counters that are absent", () => {
    expect(labels(metrics(), "Tokens")).not.toContain("Cached input");
    expect(labels(metrics(), "Model calls")).not.toContain("Provider requests");
  });

  it("shows them once the server reports them", () => {
    const m = metrics({ cached_input_tokens: 4096, provider_requests: 17 });
    expect(labels(m, "Tokens")).toContain("Cached input");
    expect(labels(m, "Model calls")).toContain("Provider requests");
  });

  it("drops a group with nothing to show rather than printing an empty heading", () => {
    const titles = telemetryGroups(metrics()).map((g) => g.title);
    expect(titles).not.toContain("Errors");
    expect(titles).not.toContain("Budgets this run was held to");
  });

  it("keeps a zero that was actually measured", () => {
    // `errors: 0` is a measurement: the run recorded no failures.
    const titles = telemetryGroups(metrics({ errors: 0 })).map((g) => g.title);
    expect(titles).toContain("Errors");
  });
});

describe("stage timings", () => {
  const staged = metrics({
    stage_seconds: { synthesize: 152.57, plan_research: 51.46, dedupe_sources: 0.003 },
  });

  it("orders stages slowest first", () => {
    expect(stageBreakdown(staged).map((s) => s.stage)).toEqual([
      "synthesize",
      "plan_research",
      "dedupe_sources",
    ]);
  });

  it("takes shares of the summed stage time, not of wall clock", () => {
    // Nodes fan out and run concurrently, so stage times can exceed
    // duration_s. Dividing by wall clock would print shares over 100%,
    // which reads as an engine bug rather than as parallelism.
    const total = stageBreakdown(staged).reduce((sum, s) => sum + s.share, 0);
    expect(total).toBeCloseTo(1, 6);
    for (const slice of stageBreakdown(staged)) {
      expect(slice.share).toBeLessThanOrEqual(1);
    }
  });

  it("says so when the parts exceed the whole", () => {
    const m = metrics({ duration_s: 10, stage_seconds: { a: 8, b: 8 } });
    const note = rowsOf(m, "Timing").find((r) => r.label === "Summed across stages")?.note;
    expect(note).toMatch(/concurrently/);
  });

  it("survives a run with no stage timings", () => {
    expect(stageBreakdown(metrics())).toEqual([]);
    expect(labels(metrics(), "Timing")).toContain("Total");
  });
});

describe("cost is reported as a floor when it is one", () => {
  it("marks an incomplete price with the engine's own caveat", () => {
    const m = metrics({ cost_is_complete: false, unpriced_categories: 7 });
    const row = rowsOf(m, "Cost").find((r) => r.label === "Estimated");
    expect(row?.note).toMatch(/floor, not a total/);
    expect(row?.note).toContain("7");
  });

  it("adds no caveat when every call was priced", () => {
    const row = rowsOf(metrics(), "Cost").find((r) => r.label === "Estimated");
    expect(row?.note).toBeUndefined();
  });
});

describe("temperature and budgets come from the run, not from constants", () => {
  it("reports the temperature the run actually used", () => {
    const m = metrics({ environment: { settings: { llm_temperature: 0.2 } } });
    const row = rowsOf(m, "Models and sampling").find((r) => r.label === "Temperature");
    expect(row?.value).toBe("0.2");
  });

  it("reports a temperature of zero rather than hiding it as falsy", () => {
    const m = metrics({ environment: { settings: { llm_temperature: 0 } } });
    const row = rowsOf(m, "Models and sampling").find((r) => r.label === "Temperature");
    expect(row?.value).toBe("0");
  });

  it("shows the ceilings the run was held to", () => {
    const m = metrics({ environment: { settings: { max_sources: 6, max_llm_calls: 20 } } });
    const budget = rowsOf(m, "Budgets this run was held to");
    expect(budget.find((r) => r.label === "Max sources")?.value).toBe("6");
    expect(budget.find((r) => r.label === "Max model calls")?.value).toBe("20");
  });
});

describe("nothing outside the known fields can reach the page", () => {
  // The server builds `environment` from an explicit allowlist and reads
  // no environment variables, so a credential should never be in this
  // payload. This asserts the client does not forward unknown keys even
  // if one ever appeared -- defence in depth, not a substitute for the
  // server-side allowlist.
  it("ignores unexpected keys in environment.settings", () => {
    const m = metrics({
      environment: {
        settings: { max_sources: 6, openai_api_key: "sk-must-never-render" } as never,
      },
    });
    const rendered = JSON.stringify(telemetryGroups(m));
    expect(rendered).not.toContain("sk-must-never-render");
    expect(rendered).not.toContain("openai_api_key");
  });

  it("ignores unexpected top-level keys in environment", () => {
    const m = metrics({
      environment: { api_key: "sk-leak", provenance: { engine_version: "1.9.0" } } as never,
    });
    const rendered = JSON.stringify(telemetryGroups(m));
    expect(rendered).not.toContain("sk-leak");
    expect(rendered).toContain("1.9.0");
  });

  it("never mentions a credential field by name in its own source", () => {
    for (const source of [telemetrySource, panelSource]) {
      expect(source).not.toMatch(/api[_-]?key/i);
      expect(source).not.toMatch(/\bsecret\b/i);
      expect(source).not.toMatch(/\btoken\b.*=.*["']/i);
    }
  });
});

describe("formatting", () => {
  it("renders sub-second stages without rounding them to zero", () => {
    expect(formatSeconds(0.003)).toBe("0.003s");
  });

  it("renders minutes past sixty seconds", () => {
    expect(formatSeconds(168.7)).toBe("2m 49s");
  });

  it("keeps four decimals on sub-cent costs", () => {
    expect(formatUsd(0.007)).toBe("$0.0070");
  });

  it("describes counts biggest first and drops zeroes", () => {
    expect(describeCounts({ planner: 2, researcher: 6, critic: 0 })).toBe(
      "researcher 6 · planner 2",
    );
  });

  it("returns nothing for an absent breakdown", () => {
    expect(describeCounts(undefined)).toBe("");
  });
});


describe("headline numbers are visible without opening anything", () => {
  // The panel first shipped entirely collapsed, and the next thing asked
  // of it was for the numbers it was already carrying. That is a
  // discoverability answer, not a capability one.
  it("surfaces duration, calls, tokens, cost and sources", () => {
    expect(headlineStats(metrics()).map((s) => s.label)).toEqual([
      "duration",
      "model calls",
      "tokens",
      "cost",
      "sources read",
    ]);
  });

  it("sums input and output into one token figure", () => {
    const stat = headlineStats(metrics()).find((s) => s.label === "tokens");
    expect(stat?.value).toBe("31,291");
  });

  it("marks the cost as a floor when the engine says it is one", () => {
    const stat = headlineStats(metrics({ cost_is_complete: false })).find(
      (s) => s.label === "cost",
    );
    expect(stat?.note).toBe("floor");
  });

  it("keeps a zero cost rather than hiding it", () => {
    // A local run really did cost nothing; that is a fact, not a gap.
    const stat = headlineStats(metrics({ known_cost_usd: 0 })).find((s) => s.label === "cost");
    expect(stat?.value).toBe("$0.0000");
    expect(stat?.note).toBeUndefined();
  });

  it("the panel renders the headline outside the disclosure", () => {
    // Asserted on the source: if the strip moved back inside <details>
    // the feature would silently revert to the thing that was missed.
    const headlineIndex = panelSource.indexOf("telemetry__headline");
    const detailsIndex = panelSource.indexOf("<details");
    expect(headlineIndex).toBeGreaterThan(-1);
    expect(detailsIndex).toBeGreaterThan(-1);
    expect(headlineIndex).toBeLessThan(detailsIndex);
  });
});
