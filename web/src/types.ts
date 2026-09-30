/** Shapes mirroring the API's serialised result. */

/**
 * `extracted` appears only in the degraded listing that replaces a report
 * when synthesis fails: one evidence item restated, quote-verified but
 * not entailment-checked.
 */
export type ClaimKind = "factual" | "synthesis" | "framing" | "extracted";

export interface Claim {
  text: string;
  kind: ClaimKind;
  /**
   * Which part of the answer contract the claim set out to fill.
   *
   * Empty when the synthesiser declared nothing, which smaller local
   * models routinely do. Such a claim can still publish; it just
   * counts toward no slot.
   */
  answer_slot?: string;
  evidence_ids: string[];
  citation_ids: string[];
}

/** One thing the answer has to contain. */
export interface AnswerSlot {
  name: string;
  description: string;
  core: boolean;
  /** Other slots that also discharge this one, when the question
   * admits more than one shape of answer. */
  satisfied_by?: string[];
}

/**
 * What the question was decided to require, before anything was
 * retrieved. Null for recordings made before contracts existed.
 */
export interface Contract {
  question: string;
  question_type: string;
  entities: string[];
  dimensions: string[];
  constraints: string[];
  ambiguities: string[];
  usable: boolean;
  unusable_reason: string;
  required_slots: AnswerSlot[];
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
  is_followup: boolean;
}

export interface Plan {
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
  issues?: {
    type: string;
    severity: string;
    detail: string;
    claim_text: string;
  }[];
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

/**
 * The engine's own coverage assessment.
 *
 * `contract.ts` derives slot status on the client on purpose, so the
 * page and the report's limitations cannot disagree. This is the part
 * that *cannot* be derived there: whether a subject the question named
 * appears in any retrieved source is a fact about the sources, and the
 * client never receives their text.
 *
 * Null for recordings made before the assessment was carried.
 */
export interface AnswerCoverage {
  satisfied_slots: string[];
  answered: boolean;
  /** Subjects no retrieved source mentions. Usually empty. */
  absent_entities: string[];
  /**
   * The relationship kind that stood in for a contrast, if one did.
   *
   * A comparison can be answered by finding there is no contrast to
   * find — one subject being a kind of the other. The contract cannot
   * say so, because it depends on what the claim asserts rather than
   * which slot it declared, so the fact travels here. Without it the
   * panel recomputes "unanswered" and renders that above a report
   * saying the opposite, which is a contradiction this interface has
   * shipped once before.
   */
  relationship_discharge: string;
}

export interface RunResult {
  run_id: string;
  report: Report | null;
  plan: Plan | null;
  contract: Contract | null;
  answer_coverage: AnswerCoverage | null;
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
