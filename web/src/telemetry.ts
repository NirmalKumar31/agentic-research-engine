/**
 * What the engine actually did, assembled for a reader who wants to check.
 *
 * All derivation lives here as pure functions so it can be tested without
 * rendering, and so the panel cannot quietly compute something different
 * from what the tests assert.
 *
 * Two rules this module follows.
 *
 * It never invents a number. A field the server did not send is omitted
 * rather than shown as zero: "0 cached tokens" and "this build did not
 * record cached tokens" are different facts, and a run metrics panel that
 * blurs them is worse than no panel. `omitEmpty` marks the rows that may
 * disappear for that reason.
 *
 * It shows the engine's own caveats rather than rounding past them.
 * `cost_is_complete` being false means some response used a token
 * category with no recorded rate, so the cost shown is a floor, and the
 * row says so.
 */

import type { Metrics } from "./types";

export interface TelemetryRow {
  label: string;
  value: string;
  /** Shown under the value: a caveat or a unit the number needs. */
  note?: string;
}

export interface TelemetryGroup {
  title: string;
  /** Why a reader should care, one line. */
  blurb?: string;
  rows: TelemetryRow[];
}

export interface StageSlice {
  stage: string;
  seconds: number;
  /** Share of the summed stage time, 0-1. Not of wall clock: stages run
   *  concurrently, so the parts can exceed the whole. */
  share: number;
}

const INTEGER = new Intl.NumberFormat("en-US");

export function formatInt(n: number): string {
  return INTEGER.format(Math.round(n));
}

export function formatSeconds(s: number): string {
  if (!Number.isFinite(s)) return "—";
  if (s < 1) return `${s.toFixed(3)}s`;
  if (s < 60) return `${s.toFixed(1)}s`;
  const minutes = Math.floor(s / 60);
  return `${minutes}m ${Math.round(s - minutes * 60)}s`;
}

export function formatUsd(n: number): string {
  if (n === 0) return "$0.0000";
  if (n < 0.01) return `$${n.toFixed(4)}`;
  return `$${n.toFixed(2)}`;
}

export function formatPercent(rate: number | null | undefined): string | null {
  if (rate === null || rate === undefined || !Number.isFinite(rate)) return null;
  return `${Math.round(rate * 100)}%`;
}

/** Turn a snake_case node name into something readable. */
export function stageLabel(stage: string): string {
  return stage.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

/**
 * Per-stage timings, slowest first.
 *
 * Shares are of the summed stage time rather than of `duration_s`. Nodes
 * fan out and run concurrently, so the stage times can add up to more
 * than the wall clock, and dividing by wall clock would print shares
 * over 100% that look like a bug in the engine rather than parallelism.
 */
export function stageBreakdown(metrics: Metrics): StageSlice[] {
  const stages = metrics.stage_seconds ?? {};
  const entries = Object.entries(stages).filter(([, s]) => Number.isFinite(s));
  const total = entries.reduce((sum, [, s]) => sum + s, 0);
  return entries
    .map(([stage, seconds]) => ({
      stage,
      seconds,
      share: total > 0 ? seconds / total : 0,
    }))
    .sort((a, b) => b.seconds - a.seconds);
}

/** `{a: 2, b: 1}` as `"a 2 · b 1"`, biggest first. Empty string for none. */
export function describeCounts(counts: Record<string, number> | undefined): string {
  if (!counts) return "";
  const entries = Object.entries(counts).filter(([, n]) => n > 0);
  if (entries.length === 0) return "";
  return entries
    .sort((a, b) => b[1] - a[1])
    .map(([k, n]) => `${k.replace(/_/g, " ")} ${formatInt(n)}`)
    .join(" · ");
}

function row(label: string, value: string | null | undefined, note?: string): TelemetryRow[] {
  if (value === null || value === undefined || value === "") return [];
  return [{ label, value, ...(note ? { note } : {}) }];
}

function counted(label: string, n: number | undefined, note?: string): TelemetryRow[] {
  if (n === undefined || n === null) return [];
  return row(label, formatInt(n), note);
}

/** A count worth showing only when it happened. */
function whenNonZero(label: string, n: number | undefined, note?: string): TelemetryRow[] {
  if (!n) return [];
  return row(label, formatInt(n), note);
}

export function timingGroup(metrics: Metrics): TelemetryGroup {
  const stages = stageBreakdown(metrics);
  const summed = stages.reduce((sum, s) => sum + s.seconds, 0);
  return {
    title: "Timing",
    blurb: "Wall clock for the run, and for each node of the graph.",
    rows: [
      ...row("Total", formatSeconds(metrics.duration_s)),
      ...row(
        "Summed across stages",
        stages.length > 0 ? formatSeconds(summed) : null,
        summed > metrics.duration_s
          ? "Longer than the total because retrieval nodes run concurrently."
          : undefined,
      ),
      ...counted("Research rounds", metrics.research_rounds),
      ...row("Stopped because", metrics.stop_reason || null),
    ],
  };
}

export function modelGroup(metrics: Metrics): TelemetryGroup {
  const settings = metrics.environment?.settings;
  const temperature = settings?.llm_temperature;
  return {
    title: "Models and sampling",
    blurb: "Which model answered in which role, and how it was sampled.",
    rows: [
      ...Object.entries(metrics.model_assignments ?? {}).map(([role, model]) => ({
        label: stageLabel(role),
        value: model,
      })),
      ...row("Temperature", temperature === undefined ? null : String(temperature)),
      ...row("Mode", metrics.mode || null),
      ...row("Provider mode", settings?.llm_mode ?? null),
    ],
  };
}

export function callsGroup(metrics: Metrics): TelemetryGroup {
  const retries =
    (metrics.structured_repairs ?? 0) +
    (metrics.compatibility_retries ?? 0) +
    (metrics.transport_retries ?? 0);
  return {
    title: "Model calls",
    blurb:
      "A logical call can become several provider requests: a schema repair, " +
      "a temperature-compatibility retry, a transport retry. Both counts are " +
      "shown because only the second is billed.",
    rows: [
      ...counted("Logical calls", metrics.llm_calls),
      ...counted("Provider requests", metrics.provider_requests),
      ...counted("Billable requests", metrics.billable_provider_requests),
      ...whenNonZero("Failed requests", metrics.failed_provider_requests),
      ...whenNonZero("Failed calls", metrics.llm_failed_calls),
      ...row("By role", describeCounts(metrics.calls_by_role) || null),
      ...row("By provider", describeCounts(metrics.calls_by_provider) || null),
      ...row("By model", describeCounts(metrics.provider_requests_by_model) || null),
      ...whenNonZero("Retries and repairs", retries || undefined, "Schema, compatibility, transport."),
      ...whenNonZero("Rate-limit refusals", metrics.rate_limit_refusals),
    ],
  };
}

export function tokenGroup(metrics: Metrics): TelemetryGroup {
  const total = metrics.input_tokens + metrics.output_tokens;
  return {
    title: "Tokens",
    rows: [
      ...counted("Input", metrics.input_tokens),
      ...counted("Output", metrics.output_tokens),
      ...row("Total", formatInt(total)),
      ...whenNonZero("Cached input", metrics.cached_input_tokens, "Read from the provider's cache."),
      ...whenNonZero("Cache writes", metrics.cache_write_tokens),
      ...whenNonZero("Reasoning", metrics.reasoning_tokens, "Billed, never shown in output."),
    ],
  };
}

export function costGroup(metrics: Metrics): TelemetryGroup {
  const incomplete = metrics.cost_is_complete === false;
  const note = incomplete
    ? metrics.unpriced_calls
      ? `A floor, not a total: ${formatInt(metrics.unpriced_calls)} call(s) used a model with no recorded rate.`
      : `A floor, not a total: ${formatInt(metrics.unpriced_categories ?? 0)} response(s) used a token category with no recorded rate.`
    : undefined;
  return {
    title: "Cost",
    rows: [
      ...row("Estimated", formatUsd(metrics.known_cost_usd), note),
      ...row("Priced completely", metrics.cost_is_complete === undefined ? null : incomplete ? "no" : "yes"),
      ...whenNonZero("Search credits", metrics.search_credits),
    ],
  };
}

export function retrievalGroup(metrics: Metrics): TelemetryGroup {
  return {
    title: "Retrieval",
    blurb: "What was searched, fetched, and thrown away before any model read it.",
    rows: [
      ...counted("Search queries", metrics.search_queries),
      ...whenNonZero("Searches failed", metrics.searches_failed),
      ...counted("Results returned", metrics.search_results),
      ...counted("Pages fetched", metrics.pages_fetched),
      ...whenNonZero("Fetch failures", metrics.fetch_failures),
      ...row("Fetch outcomes", describeCounts(metrics.fetch_status_breakdown) || null),
      ...whenNonZero("Fetches avoided by dedup", metrics.fetches_avoided),
      ...whenNonZero("Duplicate URLs", metrics.duplicate_urls),
      ...whenNonZero("Identical content", metrics.content_duplicates),
      ...counted("Unique sources", metrics.unique_sources),
      ...counted("Usable sources", metrics.usable_sources),
      ...counted("Distinct domains", metrics.distinct_domains),
      ...whenNonZero("Retrieved but never cited", metrics.unused_sources),
    ],
  };
}

export function verificationGroup(metrics: Metrics): TelemetryGroup {
  return {
    title: "Evidence and verification",
    blurb:
      "Every published sentence is checked against its own quote by an NLI " +
      "model before it may appear.",
    rows: [
      ...counted("Evidence items extracted", metrics.evidence_items),
      ...counted("Citable", metrics.citable_evidence),
      ...row(
        "Quotes found verbatim",
        metrics.evidence_items > 0
          ? formatPercent(metrics.exact_quotes / metrics.evidence_items)
          : null,
        "Exact-normalised: whitespace and smart punctuation may differ, words may not.",
      ),
      ...row("Citation integrity", formatPercent(metrics.citation_integrity_rate)),
      ...whenNonZero("Fuzzy quotes", metrics.fuzzy_quotes, "Never citable."),
      ...whenNonZero("Quotes not found in source", metrics.unmatched_quotes),
      ...row("Claim support rate", formatPercent(metrics.claim_support_rate)),
      ...row("Partial support rate", formatPercent(metrics.partial_support_rate)),
      ...row("Support breakdown", describeCounts(metrics.support_breakdown) || null),
      ...row(
        "Every eligible claim checked",
        metrics.entailment_exhaustive === undefined ? null : metrics.entailment_exhaustive ? "yes" : "no",
      ),
      ...whenNonZero("Contradictions preserved", metrics.contradictions_total),
    ],
  };
}

export function budgetGroup(metrics: Metrics): TelemetryGroup {
  const s = metrics.environment?.settings;
  if (!s) return { title: "Budgets this run was held to", rows: [] };
  return {
    title: "Budgets this run was held to",
    blurb: "Ceilings applied before the run started. A public run is capped tightly on purpose.",
    rows: [
      ...counted("Max research rounds", s.max_research_rounds),
      ...counted("Max sources", s.max_sources),
      ...counted("Max sources per round", s.max_sources_per_round),
      ...counted("Max model calls", s.max_llm_calls),
      ...row("Search depth", s.search_depth ?? null),
      ...counted("Parallel searches", s.max_parallel_searches),
      ...counted("Parallel fetches", s.max_parallel_fetches),
    ],
  };
}

export function buildGroup(metrics: Metrics): TelemetryGroup {
  const env = metrics.environment;
  const prov = env?.provenance;
  const git = prov?.git;
  const packages = env?.packages ?? {};
  const pinned = ["langgraph", "langchain-core", "pydantic"]
    .filter((name) => packages[name])
    .map((name) => `${name} ${packages[name]}`)
    .join(" · ");
  return {
    title: "Build",
    blurb: "Enough to reproduce the run, which is the point of recording any of this.",
    rows: [
      ...row("Engine version", prov?.engine_version ?? null),
      ...row(
        "Commit",
        git?.short_commit ?? git?.commit ?? null,
        git?.branch ? `branch ${git.branch}${git.dirty ? ", dirty" : ""}` : undefined,
      ),
      ...row("Prompt version", prov?.prompt_version ?? null),
      ...row("Schema version", prov?.schema_version ?? null),
      ...row("Python", env?.python ?? null),
      ...row("Key packages", pinned || null),
    ],
  };
}

export function errorGroup(metrics: Metrics): TelemetryGroup {
  return {
    title: "Errors",
    blurb: "A step can fail without failing the run; the report says what was lost.",
    rows: [
      ...counted("Errors", metrics.errors),
      ...row("Kinds", describeCounts(metrics.error_kinds) || null),
    ],
  };
}

/**
 * Every group with something to show, in reading order.
 *
 * A group whose rows all came back empty is dropped rather than rendered
 * as an empty heading -- the same rule the markdown report follows.
 */
export function telemetryGroups(metrics: Metrics): TelemetryGroup[] {
  return [
    timingGroup(metrics),
    modelGroup(metrics),
    callsGroup(metrics),
    tokenGroup(metrics),
    costGroup(metrics),
    retrievalGroup(metrics),
    verificationGroup(metrics),
    budgetGroup(metrics),
    buildGroup(metrics),
    errorGroup(metrics),
  ].filter((group) => group.rows.length > 0);
}
