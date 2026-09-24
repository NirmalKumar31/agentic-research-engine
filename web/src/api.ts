import type { ProgressEvent, RunResult } from "./types";

export interface StreamHandlers {
  onProgress: (event: ProgressEvent) => void;
  onResult: (result: RunResult) => void;
  onError: (message: string, capacityReached?: boolean) => void;
  onDone: () => void;
}

/**
 * Consume the SSE stream from POST /api/research.
 *
 * EventSource cannot issue a POST, so the body is read manually. The buffer
 * is split on the blank-line terminator rather than per chunk, because a
 * single event routinely arrives split across reads.
 */
export async function runResearch(
  query: string,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/research", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query }),
    signal,
  });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    let capacity = false;
    try {
      const body = await response.json();
      message = body.error ?? message;
      capacity = Boolean(body.capacity_reached);
    } catch {
      // Non-JSON error body; the status-derived message stands.
    }
    handlers.onError(message, capacity);
    handlers.onDone();
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    handlers.onError("Streaming is not supported by this browser.");
    handlers.onDone();
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let split = buffer.indexOf("\n\n");
    while (split !== -1) {
      dispatch(buffer.slice(0, split), handlers);
      buffer = buffer.slice(split + 2);
      split = buffer.indexOf("\n\n");
    }
  }
  handlers.onDone();
}

function dispatch(block: string, handlers: StreamHandlers): void {
  let name = "";
  let raw = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event: ")) name = line.slice(7).trim();
    else if (line.startsWith("data: ")) raw += line.slice(6);
  }
  // Server heartbeats arrive as SSE comments (": keepalive"), which carry
  // neither field. Dropping them here is what keeps them from advancing
  // pipeline state or being counted as engine events.
  if (!name || !raw) return;

  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(raw);
  } catch {
    return;
  }

  switch (name) {
    case "progress":
    case "started":
      handlers.onProgress(
        name === "started" ? { event: "started", ...payload } : (payload as ProgressEvent),
      );
      break;
    case "result":
      handlers.onResult(payload as unknown as RunResult);
      break;
    case "error":
      handlers.onError(String(payload.error ?? "The run failed."));
      break;
    default:
      break;
  }
}

export interface DemoConfig {
  /** What the service actually offers. `mode` is boot configuration and
   *  is null while replaying, because no model is reachable then. */
  service_mode: "replay" | "live";
  live_research_enabled: boolean;
  local_models_available: boolean;
  recorded_examples: number;
  mode: string | null;
  max_query_chars: number;
  max_rounds: number;
  max_sources: number;
  max_runtime_seconds: number;
  runs_per_hour: number;
}

export async function fetchConfig(): Promise<DemoConfig | null> {
  try {
    const response = await fetch("/api/config");
    return response.ok ? ((await response.json()) as DemoConfig) : null;
  } catch {
    return null;
  }
}

/** Summary of a recorded run, as listed on the homepage. */
export interface ExampleSummary {
  id: string;
  question: string;
  label: string;
  description: string;
  mode: string;
  recorded_at: string;
  sources: number;
  evidence_items: number;
  citable_evidence: number;
  has_pdf_evidence: boolean;
  duration_s: number;
}

export async function fetchExamples(): Promise<ExampleSummary[]> {
  try {
    const response = await fetch("/api/examples");
    if (!response.ok) return [];
    const body = (await response.json()) as { examples: ExampleSummary[] };
    return body.examples ?? [];
  } catch {
    return [];
  }
}

export interface RecordedRun {
  recorded: true;
  meta: ExampleSummary & { provenance?: Record<string, unknown> };
  result: RunResult;
}

export async function fetchExample(id: string): Promise<RecordedRun | null> {
  try {
    const response = await fetch(`/api/examples/${encodeURIComponent(id)}`);
    return response.ok ? ((await response.json()) as RecordedRun) : null;
  } catch {
    return null;
  }
}

/**
 * Replay a recorded run's own progress events over SSE.
 *
 * Shares `dispatch` with the live stream because the event shapes are
 * identical by construction -- the recorder stores what the graph emitted.
 * A GET, not a POST, so nothing here can be mistaken for starting a run.
 */
export async function replayExample(
  id: string,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`/api/examples/${encodeURIComponent(id)}/stream`, { signal });

  if (!response.ok) {
    handlers.onError(`Could not load that recording (${response.status}).`);
    handlers.onDone();
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    handlers.onError("Streaming is not supported by this browser.");
    handlers.onDone();
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let split = buffer.indexOf("\n\n");
    while (split !== -1) {
      dispatch(buffer.slice(0, split), handlers);
      buffer = buffer.slice(split + 2);
      split = buffer.indexOf("\n\n");
    }
  }
  handlers.onDone();
}
