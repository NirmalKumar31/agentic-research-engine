import { describe, expect, it } from "vitest";
import appSource from "./App.tsx?raw";
import type { DemoConfig } from "./api";
import { roundsLabel } from "./config";

const config = (over: Partial<DemoConfig> = {}): DemoConfig =>
  ({
    service_mode: "live",
    live_research_enabled: true,
    local_models_available: false,
    recorded_examples: 3,
    mode: "cloud",
    max_query_chars: 300,
    max_rounds: 1,
    max_sources: 6,
    max_runtime_seconds: 240,
    runs_per_hour: 10,
    ...over,
  }) as DemoConfig;

describe("demo limits are described from the server's numbers", () => {
  it("says one round when the server allows one", () => {
    expect(roundsLabel(config({ max_rounds: 1 }))).toBe("1 research round");
  });

  it("follows the server when the limit changes", () => {
    // The point of the fix: the page was hard-coded to "1 research
    // round", so raising the limit would have left it advertising a
    // bound the engine no longer had.
    expect(roundsLabel(config({ max_rounds: 2 }))).toBe(
      "up to 2 research rounds",
    );
    expect(roundsLabel(config({ max_rounds: 3 }))).toBe(
      "up to 3 research rounds",
    );
  });

  it("names no number when it has none", () => {
    expect(roundsLabel(null)).toBe("bounded research");
    expect(roundsLabel(config({ max_rounds: 0 }))).toBe("bounded research");
  });

  it("the page no longer hard-codes a round count", () => {
    expect(appSource).not.toContain("1 research round");
    expect(appSource).toContain("roundsLabel(config)");
  });
});
