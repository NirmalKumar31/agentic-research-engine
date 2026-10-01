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

/**
 * The run's own build and configuration record.
 *
 * Every field is an explicit allowlist entry on the server
 * (`environment.capture`), which reads no environment variables at all,
 * so no credential can reach here. Host details the server does collect
 * -- platform, processor, cpu count -- are deliberately not surfaced:
 * they say nothing about the run and this page is public.
 *
 * All optional. Recordings were captured by older builds.
 */
export interface RunEnvironment {
  python?: string;
  packages?: Record<string, string>;
  provenance?: {
    engine_version?: string;
    prompt_version?: string;
    schema_version?: string;
    git?: {
      commit?: string;
      short_commit?: string;
      branch?: string | null;
      dirty?: boolean | null;
    };
  };
  settings?: {
    llm_mode?: string;
    llm_temperature?: number;
    max_research_rounds?: number;
    max_sources?: number;
    max_sources_per_round?: number;
    max_llm_calls?: number;
    max_parallel_searches?: number;
    max_parallel_fetches?: number;
    search_depth?: string;
  };
}

export interface Metrics {
  duration_s: number;
  /** Wall-clock seconds per graph node, summed over rounds. */
  stage_seconds?: Record<string, number>;
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

  // Provider accounting. `llm_calls` is logical calls; one can emit
  // several provider requests through repair and retry, which is why
  // both are shown rather than one standing in for the other.
  provider_requests?: number;
  billable_provider_requests?: number;
  failed_provider_requests?: number;
  llm_failed_calls?: number;
  calls_by_role?: Record<string, number>;
  calls_by_provider?: Record<string, number>;
  provider_requests_by_model?: Record<string, number>;
  structured_repairs?: number;
  compatibility_retries?: number;
  transport_retries?: number;
  rate_limit_refusals?: number;

  // Token detail beyond in/out.
  cached_input_tokens?: number;
  cache_write_tokens?: number;
  reasoning_tokens?: number;
  unpriced_calls?: number;
  unpriced_categories?: number;

  // Retrieval.
  search_credits?: number;
  search_results?: number;
  searches_failed?: number;
  pages_fetched?: number;
  fetch_failures?: number;
  fetch_status_breakdown?: Record<string, number>;
  fetches_avoided?: number;
  duplicate_urls?: number;
  content_duplicates?: number;
  provider_content_reused?: number;
  domain_concentration?: number;

  // Evidence and verification.
  citable_evidence?: number;
  cross_attributed_evidence?: number;
  citations_total?: number;
  evidence_refs_total?: number;
  unused_sources?: number;
  contradictions_total?: number;
  partial_support_rate?: number | null;

  errors?: number;
  error_kinds?: Record<string, number>;
  environment?: RunEnvironment;
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
