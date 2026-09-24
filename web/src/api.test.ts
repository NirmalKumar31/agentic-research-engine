import { describe, expect, it, vi } from "vitest";

import { __test } from "./api";

/**
 * The capacity flag decides whether the UI offers a retry.
 *
 * A provider quota failure must not be shown with a "try again"
 * affordance: the retry fails the same way and spends another provider
 * request doing it. The plain-HTTP error path forwarded the flag; the
 * streamed path dropped it, so a 429 arriving mid-run looked like an
 * ordinary failure.
 */

function handlers() {
  return {
    onProgress: vi.fn(),
    onResult: vi.fn(),
    onError: vi.fn(),
    onDone: vi.fn(),
  };
}

const block = (event: string, data: unknown) =>
  `event: ${event}\ndata: ${JSON.stringify(data)}`;

describe("streamed errors carry the capacity flag", () => {
  it("forwards capacity_reached when the provider quota is gone", () => {
    const h = handlers();
    __test.dispatch(block("error", { error: "Quota gone.", capacity_reached: true }), h);

    expect(h.onError).toHaveBeenCalledWith("Quota gone.", true);
  });

  it("reports an ordinary failure as not capacity-related", () => {
    const h = handlers();
    __test.dispatch(block("error", { error: "The research run failed." }), h);

    expect(h.onError).toHaveBeenCalledWith("The research run failed.", false);
  });

  it("coerces a non-boolean flag rather than passing it through", () => {
    const h = handlers();
    __test.dispatch(block("error", { error: "x", capacity_reached: "yes" }), h);

    expect(h.onError).toHaveBeenCalledWith("x", true);
  });

  it("falls back to a generic message when none is given", () => {
    const h = handlers();
    __test.dispatch(block("error", {}), h);

    expect(h.onError).toHaveBeenCalledWith("The run failed.", false);
  });
});

describe("heartbeats are ignored", () => {
  it("a comment block dispatches nothing", () => {
    const h = handlers();
    __test.dispatch(": keepalive", h);

    expect(h.onProgress).not.toHaveBeenCalled();
    expect(h.onError).not.toHaveBeenCalled();
    expect(h.onResult).not.toHaveBeenCalled();
    expect(h.onDone).not.toHaveBeenCalled();
  });
});
