"""Internal domain models.

These are the objects the engine stores and reasons about. They are kept
separate from :mod:`agentic_research.schemas`, which holds the smaller shapes
we ask language models to produce. The split matters: identifiers, timestamps
and content hashes are assigned by the engine, and a model that is allowed to
invent its own source IDs will eventually cite a source that does not exist.

The type chain below is the provenance chain, and it is stored rather than
inferred::

    SubQuestion -> SearchQuery -> SearchResult -> SourceDocument
                -> EvidenceItem -> Claim -> Citation
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Stance(StrEnum):
    """How a piece of evidence relates to the sub-question it was gathered for.

    Contradictions are first-class. Collapsing disagreeing sources into one
    consensus is the single easiest way for a research tool to be confidently
    wrong, so the stance travels with the evidence all the way to synthesis.
    """

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class SourceType(StrEnum):
    """Coarse provenance class, used as one input to source quality.

    Deliberately coarse: this is a heuristic about *what kind of thing* a page
    is, not a claim about whether it is true.
    """

    OFFICIAL_DOCS = "official_docs"
    ACADEMIC = "academic"
    STANDARDS_BODY = "standards_body"
    NEWS = "news"
    VENDOR = "vendor"
    BLOG = "blog"
    FORUM = "forum"
    OTHER = "other"


class OutputFormat(StrEnum):
    COMPARISON = "comparison"
    OVERVIEW = "overview"
    HOWTO = "howto"
    TIMELINE = "timeline"
    DECISION_SUPPORT = "decision_support"


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class QueryAnalysis(BaseModel):
    """Structured understanding of what the user actually asked for."""

    original_query: str
    normalized_query: str = Field(description="Ambiguity resolved, made self-contained")
    intent: str
    entities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    output_format: OutputFormat = OutputFormat.OVERVIEW
    time_sensitive: bool = False
    recency_horizon_months: int | None = Field(
        default=None,
        description="If time sensitive, how recent sources should be to count as current",
    )
    requires_web_research: bool = True


class SubQuestion(BaseModel):
    """One research dimension of the overall question."""

    id: str = Field(description="Stable human-readable id, e.g. SQ1")
    text: str
    rationale: str = Field(description="Why answering the parent question needs this")
    priority: int = Field(default=2, ge=1, le=3, description="1 = highest")
    round_introduced: int = Field(default=1, ge=1)
    is_followup: bool = False
    parent_gap: str | None = Field(
        default=None, description="For follow-ups, the coverage gap that motivated it"
    )


class ResearchPlan(BaseModel):
    analysis: QueryAnalysis
    sub_questions: list[SubQuestion]
    strategy_note: str = ""

    def by_id(self, sub_question_id: str) -> SubQuestion | None:
        return next((q for q in self.sub_questions if q.id == sub_question_id), None)


class SearchQuery(BaseModel):
    """A web search string, tied back to the sub-question that motivated it."""

    id: str = Field(description="Stable id, e.g. Q1")
    sub_question_id: str
    text: str
    round_number: int = Field(ge=1)
    depth: str = Field(default="basic", description="Provider search depth")
    rationale: str = ""


# ---------------------------------------------------------------------------
# Search and retrieval
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """A normalised hit from any search provider.

    Providers return wildly different payloads. Everything downstream depends
    on this shape instead, which is what makes swapping Tavily for Brave a
    provider change rather than a graph change.
    """

    url: str
    title: str
    snippet: str = ""
    score: float | None = None
    published_date: datetime | None = None
    raw_content: str | None = Field(
        default=None, description="Page text if the provider supplied it"
    )
    provider: str = "unknown"
    query_id: str = ""
    query_text: str = ""
    retrieved_at: datetime = Field(default_factory=_utcnow)

    @field_validator("title", "snippet", mode="before")
    @classmethod
    def _blank_if_none(cls, v: str | None) -> str:
        return v or ""


class FetchStatus(StrEnum):
    OK = "ok"
    PROVIDER_CONTENT = "provider_content"
    """Content came from the search provider; no separate fetch was needed."""
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    TOO_LARGE = "too_large"
    UNSUPPORTED_TYPE = "unsupported_type"
    EMPTY = "empty"
    ERROR = "error"


class SourceDocument(BaseModel):
    """A deduplicated, content-bearing source.

    ``id`` doubles as the citation marker (``S3`` renders as ``[S3]``). IDs are
    assigned in the deduplication barrier rather than by parallel workers, so
    they are stable and race-free.
    """

    id: str
    url: str
    canonical_url: str
    title: str
    domain: str
    text: str = ""
    content_hash: str = ""
    source_type: SourceType = SourceType.OTHER
    published_date: datetime | None = None
    fetch_status: FetchStatus = FetchStatus.OK
    fetch_error: str | None = None
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_reasons: list[str] = Field(default_factory=list)
    search_score: float | None = None
    found_by_queries: list[str] = Field(default_factory=list)
    answers_sub_questions: list[str] = Field(default_factory=list)
    duplicate_of: str | None = None
    retrieved_at: datetime = Field(default_factory=_utcnow)
    word_count: int = 0

    @property
    def is_usable(self) -> bool:
        return self.fetch_status in (FetchStatus.OK, FetchStatus.PROVIDER_CONTENT) and bool(
            self.text.strip()
        )

    @staticmethod
    def hash_text(text: str) -> str:
        """Hash of normalised text, used for near-duplicate detection."""
        normalised = " ".join(text.split()).lower()
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """A single extracted, attributable finding.

    ``quote`` must be text that genuinely appears in the source document; it is
    validated against the source after extraction. That check is what stops the
    engine from citing a real URL for a sentence the page never contained.
    """

    id: str = Field(description="e.g. S3-e1 — unique because one worker owns one source")
    source_id: str
    sub_question_id: str
    query_id: str = ""
    claim: str = Field(description="The finding, stated in one sentence")
    quote: str = Field(description="Verbatim supporting span from the source")
    stance: Stance = Stance.NEUTRAL
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    quote_verified: bool = Field(
        default=False, description="Whether the quote was located in the source text"
    )
    extracted_at: datetime = Field(default_factory=_utcnow)

    @property
    def confidence(self) -> float:
        """Confidence that this item is a faithful reading of its source.

        Explicitly *not* a probability that the claim is true in the world.
        An unverifiable quote is the strongest available signal of extraction
        drift, so it dominates the score.
        """
        return round(self.relevance * (1.0 if self.quote_verified else 0.4), 3)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


class SubQuestionCoverage(BaseModel):
    sub_question_id: str
    evidence_count: int = 0
    distinct_sources: int = 0
    has_contradiction: bool = False
    verdict: str = Field(default="uncovered", description="covered | weak | uncovered")
    note: str = ""


class CoverageAssessment(BaseModel):
    """Whether the evidence gathered so far can answer the question.

    ``coverage_ratio`` is computed arithmetically from per-sub-question verdicts
    rather than asked of a model, because a model asked to "score coverage from
    0 to 1" produces a number with no defined meaning.
    """

    round_number: int
    per_question: list[SubQuestionCoverage] = Field(default_factory=list)
    coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    covered: list[str] = Field(default_factory=list)
    weak: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    domain_concentration: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Share of sources from the most common domain"
    )
    recommended_followups: list[str] = Field(default_factory=list)
    sufficient: bool = False
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Report and citations
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """One assertion in the report, with the sources it rests on."""

    text: str
    citation_ids: list[str] = Field(default_factory=list)
    is_interpretation: bool = Field(
        default=False,
        description="Author-level synthesis rather than a source-attributable fact",
    )


class ReportSection(BaseModel):
    heading: str
    claims: list[Claim] = Field(default_factory=list)
    sub_question_ids: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    title: str
    executive_summary: str
    sections: list[ReportSection] = Field(default_factory=list)
    key_findings: list[Claim] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    def all_claims(self) -> list[Claim]:
        claims = list(self.key_findings)
        for section in self.sections:
            claims.extend(section.claims)
        return claims

    def cited_ids(self) -> set[str]:
        return {cid for claim in self.all_claims() for cid in claim.citation_ids}


class CitationIssueType(StrEnum):
    UNKNOWN_SOURCE = "unknown_source"
    """Cited an ID that was never retrieved. Hard failure."""
    UNSUPPORTED_CLAIM = "unsupported_claim"
    """Cited source exists but does not support the claim."""
    UNCITED_CLAIM = "uncited_claim"
    """A factual assertion with no citation at all."""
    UNUSED_SOURCE = "unused_source"
    """Retrieved and paid for, never referenced. Informational only."""
    REDUNDANT_CITATION = "redundant_citation"


class CitationIssue(BaseModel):
    type: CitationIssueType
    severity: str = Field(default="warning", description="error | warning | info")
    claim_text: str = ""
    citation_id: str = ""
    detail: str = ""


class CitationVerification(BaseModel):
    total_claims: int = 0
    factual_claims: int = 0
    total_citations: int = 0
    valid_citations: int = 0
    supported_claims: int = 0
    checked_claims: int = 0
    issues: list[CitationIssue] = Field(default_factory=list)
    unused_source_ids: list[str] = Field(default_factory=list)
    repaired: bool = False

    @property
    def citation_validity_rate(self) -> float:
        """Share of citation markers pointing at a genuinely retrieved source."""
        if self.total_citations == 0:
            return 1.0
        return round(self.valid_citations / self.total_citations, 4)

    @property
    def citation_coverage_rate(self) -> float:
        """Share of factual claims carrying at least one citation."""
        if self.factual_claims == 0:
            return 1.0
        uncited = sum(1 for i in self.issues if i.type is CitationIssueType.UNCITED_CLAIM)
        return round(max(0, self.factual_claims - uncited) / self.factual_claims, 4)

    @property
    def support_rate(self) -> float:
        """Share of entailment-checked claims judged supported by their sources."""
        if self.checked_claims == 0:
            return 0.0
        return round(self.supported_claims / self.checked_claims, 4)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)


class RunError(BaseModel):
    """A non-fatal failure recorded into state instead of raised.

    Parallel workers cannot rely on the graph's node-level error handler, so
    they record failures here and the run continues degraded.
    """

    stage: str
    kind: str
    message: str
    context: str = ""
    occurred_at: datetime = Field(default_factory=_utcnow)


__all__ = [
    "CitationIssue",
    "CitationIssueType",
    "CitationVerification",
    "Claim",
    "CoverageAssessment",
    "EvidenceItem",
    "FetchStatus",
    "OutputFormat",
    "QueryAnalysis",
    "ReportSection",
    "ResearchPlan",
    "ResearchReport",
    "RunError",
    "SearchQuery",
    "SearchResult",
    "SourceDocument",
    "SourceType",
    "Stance",
    "SubQuestion",
    "SubQuestionCoverage",
]
