/** Shapes mirroring the API's serialised result. */

export type ClaimKind = "factual" | "synthesis" | "framing";

export interface Claim {
  text: string;
  kind: ClaimKind;
  evidence_ids: string[];
  citation_ids: string[];
}

export interface Section {
  heading: string;
  claims: Claim[];
}

export interface Contradiction {
  topic: string;
  left_summary: string;
  left_evidence_ids: string[];
  left_citation_ids: string[];
  right_summary: string;
  right_evidence_ids: string[];
  right_citation_ids: string[];
  auditable: boolean;
}

export interface Report {
  title: string;
  summary_claims: Claim[];
  key_findings: Claim[];
  sections: Section[];
  contradictions: Contradiction[];
  limitations: string[];
}

export interface Evidence {
  id: string;
  source_id: string;
  sub_question_id: string;
  claim: string;
  quote: string;
  quote_match: "exact_normalized" | "fuzzy" | "none";
  page: number | null;
  stance: string;
  confidence: number;
  citable: boolean;
  query_id: string;
  cross_attributed: boolean;
}

export interface Source {
  id: string;
  url: string;
  title: string;
  domain: string;
  source_type: string;
  content_origin: string;
  quality_score: number;
  page_count: number | null;
  usable: boolean;
  fetch_status: string;
}

export interface SubQuestion {
  id: string;
  text: string;
  rationale: string;
  is_followup: boolean;
}

export interface Plan {
  strategy: string;
  sub_questions: SubQuestion[];
}

export interface Verification {
  substantive_claims?: number;
  total_citations?: number;
  resolvable_citations?: number;
  total_evidence_refs?: number;
  resolvable_evidence_refs?: number;
  checked_claims?: number;
  checkable_claims?: number;
  supported_claims?: number;
  partially_supported_claims?: number;
  unsupported_claims?: number;
  entailment_exhaustive?: boolean;
  contradictions_total?: number;
  contradictions_auditable?: number;
  unused_source_ids?: string[];
  issues?: { type: string; severity: string; detail: string; claim_text: string }[];
}

export interface Metrics {
  duration_s: number;
  research_rounds: number;
  search_queries: number;
  unique_sources: number;
  usable_sources: number;
  distinct_domains: number;
  evidence_items: number;
  exact_quotes: number;
  fuzzy_quotes: number;
  unmatched_quotes: number;
  llm_calls: number;
  input_tokens: number;
  output_tokens: number;
  known_cost_usd: number;
  cost_is_complete: boolean;
  citation_integrity_rate: number | null;
  evidence_integrity_rate: number | null;
  citation_coverage_rate: number | null;
  claim_support_rate: number | null;
  support_breakdown: Record<string, number>;
  entailment_exhaustive: boolean;
  content_origins: Record<string, number>;
  model_assignments: Record<string, string>;
  mode: string;
  stop_reason: string;
}

export interface RunResult {
  run_id: string;
  report: Report | null;
  plan: Plan | null;
  evidence: Evidence[];
  sources: Source[];
  verification: Verification | null;
  metrics: Metrics;
  markdown: string;
}

/** A progress event emitted by a graph node. */
export interface ProgressEvent {
  event: string;
  [key: string]: unknown;
}
