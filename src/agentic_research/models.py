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

from pydantic import BaseModel, Field, computed_field, field_validator


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
    GOVERNMENT = "government"
    """A government publisher that is not a standards authority.

    Distinct from STANDARDS_BODY because publishing under .gov does not
    make an organisation one. NIST issues standards; a county health page
    does not, and scoring them alike overstates the second."""
    NEWS = "news"
    VENDOR = "vendor"
    BLOG = "blog"
    FORUM = "forum"
    OTHER = "other"


class ContentOrigin(StrEnum):
    """Where a source's text actually came from.

    This matters for interpreting quote fidelity. A quote aligned against
    text the search provider handed us is aligned against *that* text, not
    against an independently re-fetched origin page. Reporting the two as the
    same measurement would overstate what was verified.
    """

    PROVIDER_RAW = "provider_raw"
    """Returned by the search provider alongside the result."""
    HTML_FETCH = "html_fetch"
    """Fetched over HTTP by this engine and extracted from HTML."""
    PDF_EXTRACT = "pdf_extract"
    """Fetched over HTTP and extracted from a PDF."""
    NONE = "none"
    """No usable text was obtained."""


class QuoteMatch(StrEnum):
    """How closely an extracted quote aligns with its source text.

    A single boolean conflated two very different situations, and "verbatim"
    was being claimed for a 0.88 similarity match. Only ``EXACT_NORMALIZED``
    is honestly verbatim: it permits whitespace and smart-punctuation
    normalisation and nothing else.
    """

    EXACT_NORMALIZED = "exact_normalized"
    """Found in the source after normalising whitespace and punctuation."""
    FUZZY = "fuzzy"
    """Close but not identical. Retained for diagnostics; not citable."""
    NONE = "none"
    """Could not be aligned with the source at all."""


class ClaimKind(StrEnum):
    """What kind of assertion a claim is, and therefore what it owes.

    Replaces the old ``is_interpretation`` boolean, which acted as a blanket
    exemption from citation: marking a claim as interpretation removed it from
    verification entirely. Synthesis is *more* evidence-dependent than a single
    fact, not less, so it carries evidence too.
    """

    FACTUAL = "factual"
    """States something a single source establishes. Requires evidence."""
    SYNTHESIS = "synthesis"
    """Draws a conclusion across sources. Requires evidence, usually several."""
    FRAMING = "framing"
    """Non-substantive connective or structural text. Requires none."""
    EXTRACTED = "extracted"
    """One evidence item restated, not a claim written across sources.

    Emitted only by the degraded evidence listing that replaces a report
    when synthesis fails. Its guarantee is different in kind, not weaker
    by omission: the quote was matched verbatim against its source, and
    the text is the extractor's own summary of that one quote. There is
    no synthesis to outrun the evidence, so entailment gating does not
    apply -- and the report says so rather than presenting these as
    entailment-verified claims."""

    @property
    def requires_evidence(self) -> bool:
        return self is not ClaimKind.FRAMING

    @property
    def requires_entailment(self) -> bool:
        """Whether publication depends on an entailment verdict."""
        return self in (ClaimKind.FACTUAL, ClaimKind.SYNTHESIS)


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
    rationale: str = Field(
        default="",
        description=(
            "Why answering the parent question needs this. Engine-populated for "
            "follow-ups and fallbacks; the planner no longer emits it."
        ),
    )
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
    """Retained with a default so historical artifacts still load. The
    planner no longer emits it and nothing reads it."""

    def by_id(self, sub_question_id: str) -> SubQuestion | None:
        return next((q for q in self.sub_questions if q.id == sub_question_id), None)


class DiscoveryRef(BaseModel):
    """One concrete path by which a source was discovered.

    Previously a source held ``found_by_queries`` and ``answers_sub_questions``
    as two independent lists, which lost the relationship between them: given
    a source found by Q2 (serving SQ1) and Q7 (serving SQ4), nothing recorded
    which query belonged to which sub-question. Evidence then borrowed
    ``found_by_queries[0]``, which is right only by luck.
    """

    model_config = {"frozen": True}

    query_id: str
    sub_question_id: str


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
    SCANNED_PDF = "scanned_pdf"
    """A PDF with no extractable text. Needs OCR, which is out of scope."""
    BLOCKED = "blocked"
    """Refused by the outbound URL policy before any connection was made."""
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
    discovered_by: list[DiscoveryRef] = Field(
        default_factory=list,
        description="Every (query, sub-question) path that surfaced this source",
    )
    content_origin: ContentOrigin = ContentOrigin.NONE
    page_count: int | None = Field(default=None, description="For PDFs: pages extracted")
    page_offsets: list[int] = Field(
        default_factory=list,
        description=(
            "For PDFs: character offset in `text` where each page begins, so a "
            "quote's location can be mapped back to a page number."
        ),
    )
    duplicate_of: str | None = None
    retrieved_at: datetime = Field(default_factory=_utcnow)
    word_count: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def found_by_queries(self) -> list[str]:
        """Distinct query ids that surfaced this source, in discovery order."""
        return list(dict.fromkeys(d.query_id for d in self.discovered_by))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def answers_sub_questions(self) -> list[str]:
        """Distinct sub-questions whose queries surfaced this source."""
        return list(dict.fromkeys(d.sub_question_id for d in self.discovered_by))

    def discovery_for(self, sub_question_id: str) -> DiscoveryRef | None:
        """The discovery path belonging to a given sub-question, if any.

        Returns ``None`` when this source was never retrieved on behalf of that
        sub-question. Callers must represent that honestly rather than
        substituting an unrelated query.
        """
        return next((d for d in self.discovered_by if d.sub_question_id == sub_question_id), None)

    def page_of_offset(self, offset: int) -> int | None:
        """1-based page containing a character offset, for PDF sources."""
        if not self.page_offsets or offset < 0:
            return None
        page = 0
        for index, start in enumerate(self.page_offsets):
            if start <= offset:
                page = index + 1
            else:
                break
        return page or None

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
    discovery: DiscoveryRef | None = Field(
        default=None,
        description=(
            "The actual (query, sub-question) path that retrieved this item's "
            "source on behalf of this item's sub-question. None when the source "
            "was retrieved for a different sub-question and this finding was "
            "noticed in passing."
        ),
    )
    cross_attributed: bool = Field(
        default=False,
        description=(
            "True when this item answers a sub-question that no query "
            "retrieving this source was serving. Recorded rather than papered "
            "over with an unrelated query id."
        ),
    )
    claim: str = Field(description="The finding, stated in one sentence")
    quote: str = Field(description="Supporting span copied from the source")
    quote_match: QuoteMatch = QuoteMatch.NONE
    page: int | None = Field(default=None, description="1-based page number for PDF sources")
    stance: Stance = Stance.NEUTRAL
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    extracted_at: datetime = Field(default_factory=_utcnow)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def query_id(self) -> str:
        """Query that retrieved this item's source for its sub-question.

        Empty when the finding was cross-attributed; see ``cross_attributed``.
        """
        return self.discovery.query_id if self.discovery else ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def quote_verified(self) -> bool:
        """Whether the quote is genuinely present in the source.

        Only an exact-normalised match counts. A fuzzy match means the model
        rewrote the span, which is precisely the drift this check exists to
        catch.
        """
        return self.quote_match is QuoteMatch.EXACT_NORMALIZED

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_citable(self) -> bool:
        """Whether this item may ground a claim in the final report.

        Fuzzy and unmatched quotes are kept for diagnostics but must never
        become the basis of a citation.
        """
        return self.quote_verified

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float:
        """Confidence that this item is a faithful reading of its source.

        Explicitly *not* a probability that the claim is true in the world.
        Quote alignment dominates, because an unlocatable quote is the
        strongest available signal of extraction drift.
        """
        weight = {
            QuoteMatch.EXACT_NORMALIZED: 1.0,
            QuoteMatch.FUZZY: 0.4,
            QuoteMatch.NONE: 0.15,
        }[self.quote_match]
        return round(self.relevance * weight, 3)


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
    """One assertion in the report, anchored to the exact evidence behind it.

    ``evidence_ids`` is the primary link and the only one a model supplies.
    ``citation_ids`` is *derived* by the engine by resolving each evidence id
    to its source. Letting a model emit both invites the two to disagree, and
    makes "which evidence supports this sentence?" unanswerable — which is
    what the earlier design got wrong: verification had to guess by pulling
    arbitrary evidence belonging to a cited source.
    """

    text: str
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Exact EvidenceItem ids supporting this claim, e.g. ['S3-e2']",
    )
    citation_ids: list[str] = Field(
        default_factory=list,
        description="Source ids, derived from evidence_ids by the engine",
    )
    kind: ClaimKind = ClaimKind.FACTUAL

    @property
    def requires_evidence(self) -> bool:
        return self.kind.requires_evidence

    @property
    def is_grounded(self) -> bool:
        return bool(self.evidence_ids) or not self.requires_evidence


class Contradiction(BaseModel):
    """A disagreement between sources, auditable on both sides.

    Free-form prose could name a disagreement without either side being
    checkable, and bracketed text inside it bypassed citation verification
    entirely. Both sides now carry evidence ids and go through the same
    resolution and verification path as any other claim.
    """

    topic: str = Field(description="What the sources disagree about, in a few words")
    left_summary: str
    left_evidence_ids: list[str] = Field(default_factory=list)
    right_summary: str
    right_evidence_ids: list[str] = Field(default_factory=list)
    left_citation_ids: list[str] = Field(default_factory=list)
    right_citation_ids: list[str] = Field(default_factory=list)

    @property
    def evidence_ids(self) -> list[str]:
        return [*self.left_evidence_ids, *self.right_evidence_ids]

    @property
    def is_auditable(self) -> bool:
        """Both sides must be traceable, or it is an assertion, not a finding."""
        return bool(self.left_evidence_ids) and bool(self.right_evidence_ids)


class ReportSection(BaseModel):
    heading: str
    claims: list[Claim] = Field(default_factory=list)
    sub_question_ids: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """The final report.

    The executive summary is a list of claims rather than a prose blob. It is
    the most prominent text in the output, so excluding it from citation
    coverage and entailment checking — as a plain string necessarily did —
    meant the least verified content was the most read.
    """

    title: str
    summary_claims: list[Claim] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    key_findings: list[Claim] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @property
    def executive_summary(self) -> str:
        """The summary rendered as prose, for display and back-compatibility."""
        return " ".join(c.text.strip() for c in self.summary_claims if c.text.strip())

    def all_claims(self) -> list[Claim]:
        """Every claim subject to verification, summary included."""
        claims = [*self.summary_claims, *self.key_findings]
        for section in self.sections:
            claims.extend(section.claims)
        return claims

    def substantive_claims(self) -> list[Claim]:
        """Claims that owe evidence. Framing text is excluded by kind, not by
        a keyword heuristic guessing at whether a sentence sounds factual."""
        return [c for c in self.all_claims() if c.requires_evidence]

    def all_evidence_ids(self) -> set[str]:
        ids = {eid for claim in self.all_claims() for eid in claim.evidence_ids}
        for contradiction in self.contradictions:
            ids.update(contradiction.evidence_ids)
        return ids

    def cited_ids(self) -> set[str]:
        ids = {cid for claim in self.all_claims() for cid in claim.citation_ids}
        for contradiction in self.contradictions:
            ids.update(contradiction.left_citation_ids)
            ids.update(contradiction.right_citation_ids)
        return ids


class CitationIssueType(StrEnum):
    UNKNOWN_EVIDENCE = "unknown_evidence"
    """Claim referenced an evidence id that does not exist. Hard failure."""
    UNCITABLE_EVIDENCE = "uncitable_evidence"
    """Evidence exists but its quote never aligned to the source. Hard failure:
    a claim must not rest on text we could not find in the page."""
    UNKNOWN_SOURCE = "unknown_source"
    """Resolved to a source that was never retrieved. Hard failure."""
    UNSUPPORTED_CLAIM = "unsupported_claim"
    """Cited evidence exists but does not establish the claim."""
    PARTIALLY_SUPPORTED_CLAIM = "partially_supported_claim"
    """Cited evidence is related but weaker or narrower than the claim."""
    UNCITED_CLAIM = "uncited_claim"
    """A claim that owes evidence and carries none."""
    UNAUDITABLE_CONTRADICTION = "unauditable_contradiction"
    """A reported disagreement without evidence on both sides."""
    UNUSED_SOURCE = "unused_source"
    """Retrieved and paid for, never referenced. Informational only."""
    REDUNDANT_CITATION = "redundant_citation"


class SupportVerdict(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    NOT_CHECKED = "not_checked"


class CitationIssue(BaseModel):
    type: CitationIssueType
    severity: str = Field(default="warning", description="error | warning | info")
    claim_text: str = ""
    citation_id: str = ""
    evidence_id: str = ""
    detail: str = ""


class CitationVerification(BaseModel):
    """Outcome of verifying a report's citations.

    Metric names here are deliberately literal. The old name
    ``citation_validity`` implied a source supported its claim, when all it
    measured was that the id resolved to something retrieved. Those are very
    different guarantees and only one of them was being made.
    """

    total_claims: int = 0
    substantive_claims: int = 0
    """Claims that owe evidence (factual or synthesis), by declared kind."""
    total_citations: int = 0
    resolvable_citations: int = 0
    total_evidence_refs: int = 0
    resolvable_evidence_refs: int = 0

    supported_claims: int = 0
    partially_supported_claims: int = 0
    unsupported_claims: int = 0
    checked_claims: int = 0
    """Claims actually put through entailment checking."""
    checkable_claims: int = 0
    """Claims eligible for entailment checking, whether or not sampled."""
    not_checked_claims: int = 0
    """Eligible claims the bounded verifier never reached.

    Reported rather than inferred. These are the claims that used to be
    published unverified: the gate removed only claims with a *failing*
    verdict, so an unchecked claim carried no issue and survived."""
    entailment_exhaustive: bool = False
    """True when every eligible claim was checked (benchmark mode)."""

    contradictions_total: int = 0
    contradictions_auditable: int = 0
    """Both sides carry citable evidence. Structural, not semantic."""
    contradiction_sides_checkable: int = 0
    contradiction_sides_checked: int = 0
    contradictions_semantically_supported: int = 0
    """Both summaries were checked and both came back supported.

    Distinct from ``contradictions_auditable`` on purpose. A contradiction
    summary is model-written prose, and structural traceability says only
    that evidence exists on both sides -- not that either sentence follows
    from it. Overloading 'auditable' to mean both would hide exactly the
    gap this counter exists to close."""

    duplicate_claims_removed: int = 0
    """Exact duplicate copies removed before verification."""

    # Publication gate. Kept separate from the support counts above so the
    # synthesiser's actual output stays visible: a report with nothing
    # unsupported in it because four claims were removed is not the same
    # as one that never generated a bad claim.
    generated_substantive_claims: int = 0
    """Substantive claims the synthesiser produced, before filtering."""
    removed_after_verification: int = 0
    """Claims dropped because their evidence did not support them."""
    final_published_claims: int = 0
    """Substantive claims surviving into the published report."""

    issues: list[CitationIssue] = Field(default_factory=list)
    unused_source_ids: list[str] = Field(default_factory=list)
    repaired: bool = False

    # -- integrity ---------------------------------------------------------

    @property
    def citation_integrity_rate(self) -> float:
        """Share of citations resolving to a source retrieved in this run.

        Says nothing about whether the source supports the claim. That is
        ``claim_support_rate``.
        """
        if self.total_citations == 0:
            return 1.0
        return round(self.resolvable_citations / self.total_citations, 4)

    @property
    def evidence_integrity_rate(self) -> float:
        """Share of evidence references resolving to citable evidence.

        Returns 1.0 on an empty denominator, which is the right structural
        invariant -- there is no broken reference -- but is not a quality
        measurement. A report that published nothing has no references to
        get wrong. Use :attr:`has_evidence_references` before presenting
        this figure anywhere a reader will read it as a score.
        """
        if self.total_evidence_refs == 0:
            return 1.0
        return round(self.resolvable_evidence_refs / self.total_evidence_refs, 4)

    @property
    def has_evidence_references(self) -> bool:
        """Whether the integrity rate has a non-zero denominator.

        Exists so a caller cannot accidentally render "100%" for 0/0. The
        README did exactly that beside an empty report.
        """
        return self.total_evidence_refs > 0

    @property
    def citation_validity_rate(self) -> float:
        """Deprecated alias of :attr:`citation_integrity_rate`.

        Retained so older artifacts and readers still resolve, but no longer
        published under this name.
        """
        return self.citation_integrity_rate

    # -- coverage ----------------------------------------------------------

    @property
    def citation_coverage_rate(self) -> float:
        """Share of evidence-owing claims carrying at least one citation.

        1.0 on an empty denominator for the same reason as
        :attr:`evidence_integrity_rate`: a report with no claims has none
        without a citation. Check ``substantive_claims`` before rendering
        it as a percentage.
        """
        if self.substantive_claims == 0:
            return 1.0
        uncited = sum(1 for i in self.issues if i.type is CitationIssueType.UNCITED_CLAIM)
        return round(max(0, self.substantive_claims - uncited) / self.substantive_claims, 4)

    # -- support -----------------------------------------------------------

    @property
    def claim_support_rate(self) -> float:
        """Share of *checked* claims judged fully supported.

        Partial support is excluded from the numerator and reported
        separately rather than folded silently into either bucket.
        """
        if self.checked_claims == 0:
            return 0.0
        return round(self.supported_claims / self.checked_claims, 4)

    @property
    def partial_support_rate(self) -> float:
        if self.checked_claims == 0:
            return 0.0
        return round(self.partially_supported_claims / self.checked_claims, 4)

    @property
    def support_breakdown(self) -> dict[str, int]:
        not_checked = max(0, self.checkable_claims - self.checked_claims)
        return {
            SupportVerdict.SUPPORTED.value: self.supported_claims,
            SupportVerdict.PARTIALLY_SUPPORTED.value: self.partially_supported_claims,
            SupportVerdict.UNSUPPORTED.value: self.unsupported_claims,
            SupportVerdict.NOT_CHECKED.value: not_checked,
        }

    @property
    def support_rate(self) -> float:
        """Deprecated alias of :attr:`claim_support_rate`."""
        return self.claim_support_rate

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
    "ClaimKind",
    "ContentOrigin",
    "Contradiction",
    "CoverageAssessment",
    "DiscoveryRef",
    "EvidenceItem",
    "FetchStatus",
    "OutputFormat",
    "QueryAnalysis",
    "QuoteMatch",
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
    "SupportVerdict",
]
