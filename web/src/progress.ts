import type { ProgressEvent } from "./types";

/**
 * Turn a graph event into a line of UI text.
 *
 * Every entry corresponds to a node that actually ran. Nothing here
 * invents a step or interpolates a fake percentage: if the engine does not
 * emit it, the user does not see it.
 */
export function describe(event: ProgressEvent): string | null {
  const n = (key: string) => Number(event[key] ?? 0);
  const s = (key: string) => String(event[key] ?? "");

  switch (event.event) {
    case "started":
      return `Starting research`;
    case "warning":
      return `Warning: ${s("message")}`;
    case "analyzing_query":
      return "Analysing the question";
    case "query_analyzed":
      return `Intent: ${s("intent")}`;
    case "planning":
      return "Planning research dimensions";
    case "plan_generated":
      return `Planned ${n("count")} sub-questions`;
    case "generating_queries":
      return `Writing search queries for round ${n("round")}`;
    case "queries_generated":
      return `Round ${n("round")}: searching ${n("count")} queries`;
    case "search_completed":
      return `Searched "${s("query").slice(0, 60)}" — ${n("results")} results`;
    case "search_failed":
      return `Search failed: ${s("query").slice(0, 60)}`;
    case "sources_deduplicated":
      return `${n("results")} results → ${n("unique")} unique (${n("avoided")} duplicate fetches avoided), retrieving ${n("selected")}`;
    case "source_retrieved":
      return `Retrieved [${s("source_id")}] ${s("title")}`;
    case "source_failed":
      return `Could not use [${s("source_id")}] (${s("status")})`;
    case "sources_registered":
      return `${n("usable")} usable sources, extracting from ${n("extracting")}`;
    case "evidence_extracted":
      return `Extracted ${n("items")} findings from ${s("source_id")}`;
    case "assessing_coverage":
      return "Assessing coverage";
    case "coverage_evaluated":
      return `Coverage ${Math.round(n("ratio") * 100)}% — ${
        event.sufficient ? "sufficient" : "gaps remain"
      }`;
    case "generating_followups":
      return `Identified ${n("gaps")} gaps`;
    case "followups_generated":
      return `Following up on ${n("count")} gap(s)`;
    case "synthesizing":
      return `Synthesising from ${n("evidence")} evidence items`;
    case "synthesized":
      return `Draft written: ${n("sections")} sections, ${n("findings")} findings`;
    case "verifying_citations":
      return "Verifying citations against their evidence";
    case "citations_verified":
      return `Verified ${n("total")} citations`;
    case "completed":
      return `Complete (${s("stop_reason")})`;
    default:
      return null;
  }
}

/** Coarse stage, used to show where the run is without faking a percentage. */
export function stageOf(event: ProgressEvent): string {
  const e = String(event.event);
  if (e.includes("analy") || e.includes("plan")) return "planning";
  if (e.includes("search") || e.includes("quer")) return "searching";
  if (e.includes("source") || e.includes("evidence")) return "reading";
  if (e.includes("coverage") || e.includes("followup")) return "assessing";
  if (e.includes("synth")) return "writing";
  if (e.includes("citation") || e.includes("verif")) return "verifying";
  if (e === "completed") return "done";
  return "running";
}
