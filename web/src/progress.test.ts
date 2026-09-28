import { describe as group, expect, it } from "vitest";

import { describe as label } from "./progress";
import type { ProgressEvent } from "./types";

/**
 * The verifier wake is slow and, until it was announced, invisible.
 *
 * A remote verifier at minimum replicas 0 takes roughly a minute to
 * start, and that happens after "started" and before the first
 * pipeline stage. The stream carried heartbeats through it and nothing
 * else, so the page sat on step 1 with no explanation for over a third
 * of the run and read as hung.
 */
group("verifier wake", () => {
  const event = (extra: Record<string, unknown>) => extra as unknown as ProgressEvent;

  it("names the wait and how long it may take", () => {
    expect(
      label(
        event({
          event: "verifier_waking",
          detail: "Waking the verifier",
          expected_seconds: 90,
        }),
      ),
    ).toBe("Waking the verifier (up to ~90s)");
  });

  it("omits the budget when none is configured", () => {
    // A local verifier loads from disk; quoting a remote wake budget
    // there would be a number invented for the occasion.
    expect(
      label(
        event({
          event: "verifier_waking",
          detail: "Loading the verifier",
          expected_seconds: 0,
        }),
      ),
    ).toBe("Loading the verifier");
  });

  it("reports when the verifier is ready", () => {
    expect(label(event({ event: "verifier_ready" }))).toBe("Verifier ready");
  });

  it("still labels the events that were already there", () => {
    expect(label(event({ event: "started" }))).toBe("Starting research");
  });
});
