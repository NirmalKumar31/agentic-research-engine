import type { ProgressEvent } from "./types";

/**
 * Stage model for the research pipeline visualisation.
 *
 * Every value here is derived from events the graph actually emitted. No
 * stage advances on a timer, no count is interpolated, and nothing is
 * shown before the engine reports it. If the run dies at Search, Search
 * stays active and the later stages stay idle -- which is the truth.
 */

export type StageState = "idle" | "active" | "done" | "failed";

export interface Stage {
  id: string;
  label: string;
  /** One line explaining the stage, used by the How it works section. */
  blurb: string;
}

export const STAGES: Stage[] = [
  { id: "analyze", label: "Analyse", blurb: "Work out what the question is actually asking." },
  { id: "plan", label: "Plan", blurb: "Decompose it into researchable sub-questions." },
  { id: "search", label: "Search", blurb: "Run a query per sub-question, in parallel." },
  { id: "retrieve", label: "Retrieve", blurb: "Deduplicate, then fetch each unique page once." },
  { id: "extract", label: "Extract", blurb: "Pull evidence as quotes checked against the text." },
  { id: "coverage", label: "Coverage", blurb: "Count what is answered; loop if there are gaps." },
  { id: "synthesize", label: "Synthesise", blurb: "Write the report from the evidence gathered." },
  { id: "verify", label: "Verify", blurb: "Resolve every citation back to its evidence." },
];

/** Which stage an event belongs to, or null if it belongs to none. */
function stageFor(event: string): string | null {
  if (event.startsWith("analyz") || event === "query_analyzed") return "analyze";
  if (event.startsWith("plan")) return "plan";
  if (event.includes("quer") || event.startsWith("search")) return "search";
  if (event.startsWith("source")) return "retrieve";
  if (event.startsWith("evidence")) return "extract";
  if (event.includes("coverage") || event.includes("followup")) return "coverage";
  if (event.startsWith("synthes")) return "synthesize";
  if (event.includes("citation") || event.startsWith("verif")) return "verify";
  return null;
}

const FAILURE_EVENTS = new Set(["search_failed", "source_failed"]);

export interface FanOutQuery {
  id: string;
  text: string;
  results: number | null;
}

export interface PipelineState {
  states: Record<string, StageState>;
  /** Queries seen this run, for the fan-out visualisation. */
  queries: FanOutQuery[];
  /** Counts the engine reported. Absent until it does. */
  counts: {
    subQuestions: number | null;
    searches: number | null;
    sources: number | null;
    evidence: number | null;
    claims: number | null;
    coverage: { covered: number; total: number } | null;
  };
  finished: boolean;
}

export function emptyPipeline(): PipelineState {
  const states: Record<string, StageState> = {};
  for (const stage of STAGES) states[stage.id] = "idle";
  return {
    states,
    queries: [],
    counts: {
      subQuestions: null,
      searches: null,
      sources: null,
      evidence: null,
      claims: null,
      coverage: null,
    },
    finished: false,
  };
}

/**
 * Fold one event into the pipeline state.
 *
 * Reaching a later stage marks earlier ones done: the graph is sequential
 * between barriers, so arriving at Extract means Retrieve finished. That
 * is inference from the engine's own ordering, not decoration.
 */
export function advance(prev: PipelineState, event: ProgressEvent): PipelineState {
  const name = String(event.event);
  const num = (k: string) => (typeof event[k] === "number" ? (event[k] as number) : null);

  const next: PipelineState = {
    ...prev,
    states: { ...prev.states },
    queries: prev.queries,
    counts: { ...prev.counts },
  };

  if (name === "completed") {
    for (const stage of STAGES) {
      if (next.states[stage.id] === "active") next.states[stage.id] = "done";
    }
    next.finished = true;
    return next;
  }

  const id = stageFor(name);
  if (!id) return next;

  const index = STAGES.findIndex((s) => s.id === id);
  if (index >= 0) {
    for (let i = 0; i < index; i += 1) {
      if (next.states[STAGES[i].id] !== "failed") next.states[STAGES[i].id] = "done";
    }
    if (FAILURE_EVENTS.has(name)) {
      next.states[id] = "failed";
    } else if (next.states[id] !== "failed") {
      next.states[id] = "active";
    }
  }

  switch (name) {
    case "plan_generated":
      next.counts.subQuestions = num("count");
      break;
    case "queries_generated":
      next.counts.searches = num("count");
      break;
    case "search_completed": {
      const text = String(event.query ?? "");
      const qid = String(event.query_id ?? `Q${prev.queries.length + 1}`);
      next.queries = [
        ...prev.queries.filter((q) => q.id !== qid),
        { id: qid, text, results: num("results") },
      ];
      break;
    }
    case "search_failed": {
      const qid = String(event.query_id ?? "");
      next.queries = [
        ...prev.queries.filter((q) => q.id !== qid),
        { id: qid, text: String(event.query ?? ""), results: null },
      ];
      break;
    }
    case "sources_registered":
      next.counts.sources = num("usable");
      break;
    case "synthesizing":
      next.counts.evidence = num("evidence");
      break;
    case "citations_verified":
      next.counts.claims = num("total");
      break;
    case "coverage_evaluated": {
      const covered = num("covered");
      const ratio = typeof event.ratio === "number" ? (event.ratio as number) : null;
      if (covered !== null && ratio && ratio > 0) {
        next.counts.coverage = { covered, total: Math.round(covered / ratio) };
      } else if (covered !== null) {
        next.counts.coverage = { covered, total: covered };
      }
      break;
    }
    default:
      break;
  }
  return next;
}

/**
 * Human-readable line for the activity feed.
 *
 * Deliberately narrower than the raw event set: a visitor does not need
 * to read `sources_registered`. The raw stream stays available behind a
 * toggle for anyone who does.
 */
export function humanise(event: ProgressEvent): string | null {
  const n = (k: string) => Number(event[k] ?? 0);
  const s = (k: string) => String(event[k] ?? "");
  switch (event.event) {
    case "started":
      return "Starting research";
    case "query_analyzed":
      return `Read the question as: ${s("intent")}`;
    case "plan_generated":
      return `Decomposed into ${n("count")} research questions`;
    case "queries_generated":
      return `Searching ${n("count")} queries in parallel`;
    case "sources_deduplicated":
      return `${n("results")} results narrowed to ${n("unique")} unique pages, ${n(
        "avoided",
      )} duplicate fetches avoided`;
    case "sources_registered":
      return `Retrieved ${n("usable")} usable sources`;
    case "source_failed":
      return `A source could not be used (${s("status")})`;
    case "search_failed":
      return "A search failed; the run continues with the others";
    case "coverage_evaluated":
      return event.sufficient
        ? `Coverage sufficient across ${n("covered")} dimensions`
        : `Coverage incomplete: ${n("covered")} covered, gaps remain`;
    case "synthesizing":
      return `Writing the report from ${n("evidence")} evidence items`;
    case "citations_verified":
      return `Verified ${n("total")} citations against their evidence`;
    case "completed":
      return `Finished — ${s("stop_reason")}`;
    default:
      return null;
  }
}
