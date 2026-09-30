"""Schemas for structured LLM output.

These are the shapes handed to ``with_structured_output``. They are separate
from the domain models in :mod:`agentic_research.models` for three reasons:

1. Identifiers and timestamps are assigned by the engine. A model that invents
   its own source IDs will eventually cite a source that was never retrieved.
2. ``Field(description=...)`` is rendered into the JSON schema and is read by
   the model, so these descriptions are part of the prompt. Editing them
   changes behaviour.
3. Schemas are kept flat and small on purpose. A 4B local model that handles a
   two-level schema reliably will start emitting malformed nesting at four,
   and the engine is meant to run on local models as a first-class mode.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalysisOut(BaseModel):
    """Structured reading of the user's question."""

    normalized_query: str = Field(
        description="The question rewritten to be self-contained and unambiguous"
    )
    intent: str = Field(description="What the user is trying to accomplish, one sentence")
    entities: list[str] = Field(
        default_factory=list,
        description=(
            "Key technologies, organisations or concepts named, including the setting "
            "the question is about. Context for retrieval, not the sides of a comparison"
        ),
    )
    comparison_subjects: list[str] = Field(
        default_factory=list,
        description=(
            "For a comparison only: exactly the things being compared with each other. "
            "Never the domain or setting they operate in. For 'how does X differ from Y "
            "for Z', this is X and Y; Z belongs in entities"
        ),
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Explicit scope limits stated by the user, such as a domain or time period",
    )
    output_format: Literal[
        "comparison",
        "overview",
        "howto",
        "timeline",
        "decision_support",
        "causal_analysis",
        "metric",
        "list",
        "synthesis",
    ] = Field(
        description=(
            "The shape of answer this question calls for. Choose by what the "
            "answer must contain, following the selection rules; 'overview' "
            "only when no other shape fits"
        )
    )
    dimensions: list[str] = Field(
        default_factory=list,
        description=(
            "For a comparison only: the axes the question asks the subjects to be "
            "compared on, such as cost or latency. Empty when the question names none"
        ),
    )
    parts: list[str] = Field(
        default_factory=list,
        description=(
            "For a multi-part question only: each distinct question asked, rewritten "
            "to stand alone. Required when output_format is 'synthesis'"
        ),
    )
    time_sensitive: bool = Field(
        description="True if the correct answer depends on recent developments"
    )
    recency_horizon_months: int | None = Field(
        default=None, description="If time sensitive, how old a source may be and still count"
    )
    requires_web_research: bool = Field(
        description="False only for pure definition or arithmetic questions needing no sources"
    )


class SubQuestionOut(BaseModel):
    text: str = Field(description="A focused, independently researchable question")
    priority: int = Field(description="1 = essential, 2 = useful, 3 = nice to have")


class PlanOut(BaseModel):
    """Decomposition of the question into research dimensions.

    Carries only what the engine consumes. ``strategy_note`` and a
    per-sub-question ``rationale`` used to be required here and were
    written, stored and never read -- no query, coverage or synthesis step
    consumed either. They were not free: a 4B planner spent most of its
    2,000-token output allowance narrating, then truncated mid-JSON before
    finishing the sub-question list, and the run fell back to a single
    dimension. Asking only for the fields that are used is the fix; raising
    the cap would just buy more room for prose.
    """

    sub_questions: list[SubQuestionOut] = Field(
        description=(
            "Distinct dimensions of the question. Each must be researchable on its own and "
            "must not restate another. Prefer different angles (cost, performance, security, "
            "maturity, tradeoffs) over rewordings of the same angle."
        ),
        min_length=2,
        max_length=8,
    )


class QueryOut(BaseModel):
    sub_question_id: str = Field(description="The SQ id this query serves, e.g. SQ2")
    text: str = Field(
        description=(
            "A web search query in the question's own vocabulary. Short keywords, "
            "not a sentence, and not a stack of specialist terms. No quotes or operators"
        )
    )
    rationale: str = Field(
        default="",
        description=(
            "In a few words, what part of the answer this query is meant to supply "
            "and why it is phrased this way"
        ),
    )


class QueriesOut(BaseModel):
    queries: list[QueryOut] = Field(
        description=(
            "One or two search queries per sub-question. Must be meaningfully different from "
            "each other and from any previously issued query supplied in the prompt."
        ),
        max_length=16,
    )


class EvidenceOut(BaseModel):
    sub_question_id: str = Field(description="Which sub-question this finding addresses")
    claim: str = Field(description="The finding stated in one self-contained sentence")
    quote: str = Field(
        description=(
            "A verbatim span copied exactly from the source text, 10-60 words. "
            "Do not paraphrase, correct or abbreviate it; it is checked against the source."
        )
    )
    stance: Literal["supports", "contradicts", "neutral"] = Field(
        description=(
            "'supports' if it affirms a likely answer, 'contradicts' if it cuts against one "
            "or disagrees with common claims, 'neutral' for background"
        )
    )
    relevance: float = Field(description="0.0 to 1.0, how directly this addresses the question")


class ExtractionOut(BaseModel):
    """Findings pulled from one source document."""

    evidence: list[EvidenceOut] = Field(
        description=(
            "Findings genuinely present in this source. Return an empty list if the source "
            "is irrelevant. Never invent a finding to avoid returning nothing."
        ),
        max_length=6,
    )


class CoverageOut(BaseModel):
    """The judgement half of coverage analysis.

    The engine computes the counting half (how many evidence items, how many
    distinct domains, coverage ratio) itself. A model is only asked for things
    that genuinely need reading comprehension.
    """

    weak_sub_question_ids: list[str] = Field(
        default_factory=list,
        description="Sub-questions whose evidence is thin, one-sided or off-target",
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Substantive disagreements between sources, each described in one sentence",
    )
    missing_angles: list[str] = Field(
        default_factory=list,
        description="Important aspects of the question that no evidence touches yet",
    )
    reasoning: str = Field(description="Two or three sentences justifying the assessment")


class FollowupOut(BaseModel):
    text: str = Field(description="A new, narrower research question targeting a specific gap")
    gap: str = Field(description="The gap it is meant to close")


class FollowupsOut(BaseModel):
    followups: list[FollowupOut] = Field(
        description="Targeted follow-up questions. Must not repeat existing sub-questions.",
        max_length=4,
    )


# Bounded at the schema boundary. Under the generative verifier this
# stopped a renderer from showing six of eight quotes while the verdict
# claimed all eight were seen. The NLI gate scores each quote separately
# so that particular failure is gone, but the bound stays: a claim
# resting on more than eight items is not atomic, whatever else it is.
MAX_EVIDENCE_PER_CLAIM = 8


class ClaimOut(BaseModel):
    """One assertion in the report.

    The model supplies ``evidence_ids`` only. It never names a source: the
    engine resolves evidence -> source itself. Asking a model for both invites
    the two to disagree, and the earlier design — where the model emitted
    source ids directly — made "which evidence supports this sentence?"
    unanswerable, so verification had to guess by sampling arbitrary evidence
    belonging to the cited source.
    """

    text: str = Field(
        description=(
            "One assertion, in plain prose. Do not put citation markers or "
            "brackets in this text; list the evidence ids in evidence_ids."
        )
    )
    answer_slot: str = Field(
        default="",
        description=(
            "Which part of the question this claim answers. Copy one name "
            "exactly from the list of required answer slots given above. A "
            "claim that answers none of them should not be written: it may "
            "be true, well sourced, and still not an answer to what was "
            "asked."
        ),
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=MAX_EVIDENCE_PER_CLAIM,
        description=(
            "Ids of the specific evidence items this claim rests on, copied "
            "exactly from the evidence list, e.g. ['S3-e2', 'S7-e1']. Cite the "
            "evidence you actually used, not everything about the topic. "
            f"At most {MAX_EVIDENCE_PER_CLAIM}."
        ),
    )
    kind: Literal["factual", "synthesis"] = Field(
        description=(
            "'factual' = states something one evidence item establishes. "
            "'synthesis' = a conclusion drawn across several evidence items; "
            "still requires evidence_ids, usually more than one."
        )
    )


class ContradictionOut(BaseModel):
    """A disagreement between sources, with evidence on both sides."""

    topic: str = Field(description="What the sources disagree about, in a few words")
    left_summary: str = Field(description="What one side reports")
    left_evidence_ids: list[str] = Field(
        description="Evidence ids supporting the first position", min_length=1
    )
    right_summary: str = Field(description="What the other side reports")
    right_evidence_ids: list[str] = Field(
        description="Evidence ids supporting the second position", min_length=1
    )


class SectionOut(BaseModel):
    heading: str
    claims: list[ClaimOut] = Field(max_length=10)


class ReportOut(BaseModel):
    """The final report, before citation verification."""

    title: str
    summary_claims: list[ClaimOut] = Field(
        description=(
            "3-5 claims answering the question directly. These are the most "
            "prominent statements in the report and carry evidence ids exactly "
            "like body claims do."
        ),
        min_length=1,
        max_length=6,
    )
    sections: list[SectionOut] = Field(
        description="Body sections organised by research dimension", max_length=8
    )
    key_findings: list[ClaimOut] = Field(
        description="The most important takeaways, each with its evidence ids",
        max_length=8,
    )
    contradictions: list[ContradictionOut] = Field(
        default_factory=list,
        description=(
            "Where sources genuinely disagree. Report the disagreement rather "
            "than resolving it. Both sides need evidence ids."
        ),
        max_length=6,
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="What this research could not establish, and why",
    )


__all__ = [
    "AnalysisOut",
    "ClaimOut",
    "ContradictionOut",
    "CoverageOut",
    "EvidenceOut",
    "ExtractionOut",
    "FollowupOut",
    "FollowupsOut",
    "PlanOut",
    "QueriesOut",
    "QueryOut",
    "ReportOut",
    "SectionOut",
    "SubQuestionOut",
]


class RelevanceVerdictOut(BaseModel):
    """One claim, judged against the question it is meant to answer."""

    claim_index: int = Field(
        description="The index of the claim being judged, copied from the list"
    )
    answers_question: bool = Field(
        description=(
            "True only if this claim helps answer the question that was asked. "
            "A claim can be entirely true, well sourced, and still not an "
            "answer: a definition of one subject does not answer how two "
            "subjects differ."
        )
    )
    reason: str = Field(
        default="",
        description="One short sentence. What it answers, or what it is about instead.",
    )


class RelevanceOut(BaseModel):
    """Judgements for every candidate claim in one call."""

    verdicts: list[RelevanceVerdictOut] = Field(default_factory=list)


class RewriteOut(BaseModel):
    """One claim, reworded to satisfy a rule it broke."""

    claim_index: int = Field(description="Index of the claim, copied from the list")
    rewritten: str = Field(
        default="",
        description=(
            "The same claim, reworded so it no longer breaks the stated rule. "
            "Use only what the quote already says. Add no fact, no number, no "
            "source, and do not state the claim more strongly than the "
            "original did. Leave empty if it cannot be fixed by rewording."
        ),
    )


class RepairOut(BaseModel):
    verdicts: list[RewriteOut] = Field(default_factory=list)
