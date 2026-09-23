"""Deterministic stand-ins for every external dependency.

Lets the whole graph run in milliseconds with no network, no credentials and
no cost, while still exercising the real state machine, the real reducers, the
real deduplication and the real citation verification. Only the three I/O
boundaries are faked: model calls, search and page fetching.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from agentic_research.config import ModelRole, ModelSpec, Provider, Settings
from agentic_research.llm.base import LLMCallRecord, LLMError, UsageTracker
from agentic_research.models import SearchQuery, SearchResult
from agentic_research.retrieval.fetcher import FetchResult, FetchStats
from agentic_research.schemas import (
    AnalysisOut,
    ClaimOut,
    CoverageOut,
    EntailmentOut,
    EvidenceOut,
    ExtractionOut,
    FollowupOut,
    FollowupsOut,
    PlanOut,
    QueriesOut,
    QueryOut,
    ReportOut,
    SectionOut,
    SubQuestionOut,
)
from agentic_research.search.base import SearchResponse

PAGE_TEXT = (
    "Fraud detection datasets are severely imbalanced, with positive cases often "
    "well under one percent of all recorded transactions. Resampling methods such "
    "as SMOTE generate synthetic minority examples, while cost-sensitive learning "
    "assigns a higher penalty to missed fraud. Gradient boosted trees remain a "
    "strong baseline in production systems, and precision-recall curves are a more "
    "informative evaluation than ROC AUC under heavy imbalance."
)


class FakeRoleModel:
    """Returns a canned, schema-valid object for each schema type."""

    def __init__(self, role: ModelRole, owner: FakeRouter) -> None:
        self.role = role
        self._owner = owner

    async def structured(self, schema: type, system: str, user: str, **_: Any) -> Any:
        self._owner.calls.append((self.role, schema.__name__))
        name = schema.__name__
        if name in self._owner.fail_schemas:
            self._record(name, ok=False)
            raise LLMError(f"induced failure for {name}")
        await asyncio.sleep(0)  # yield, so concurrency is genuinely exercised
        self._record(name, ok=True)
        builder = self._owner.responses.get(name)
        if builder is not None:
            return builder(user)
        return _DEFAULTS[name](user)

    def _record(self, schema_name: str, *, ok: bool) -> None:
        """Feed the real UsageTracker.

        Without this the fake router reports zero calls and zero tokens, and
        every metrics assertion silently passes against an empty tracker.
        """
        self._owner.tracker.record(
            LLMCallRecord(
                role=self.role,
                provider=self._owner._spec.provider,
                model=self._owner._spec.model,
                schema=schema_name,
                latency_s=0.01,
                input_tokens=100,
                output_tokens=40,
                ok=ok,
            )
        )


def _default_analysis(_: str) -> AnalysisOut:
    return AnalysisOut(
        normalized_query="Compare approaches for fraud detection on imbalanced data",
        intent="compare technical approaches",
        entities=["SMOTE", "cost-sensitive learning"],
        constraints=[],
        output_format="comparison",
        time_sensitive=False,
        recency_horizon_months=None,
        requires_web_research=True,
    )


def _default_plan(_: str) -> PlanOut:
    return PlanOut(
        strategy_note="cover sampling, algorithms and evaluation",
        sub_questions=[
            SubQuestionOut(text="Which resampling methods work?", rationale="r1", priority=1),
            SubQuestionOut(text="Which algorithms handle imbalance?", rationale="r2", priority=1),
            SubQuestionOut(text="How should models be evaluated?", rationale="r3", priority=2),
        ],
    )


def _default_queries(user: str) -> QueriesOut:
    # Vary by round so follow-up rounds do not collide with round-1 queries.
    suffix = "advanced" if "SQ4" in user or "follow" in user.lower() else "basic"
    return QueriesOut(
        queries=[
            QueryOut(sub_question_id="SQ1", text=f"smote imbalanced fraud {suffix}"),
            QueryOut(sub_question_id="SQ2", text=f"gradient boosting fraud {suffix}"),
            QueryOut(sub_question_id="SQ3", text=f"precision recall imbalance {suffix}"),
        ]
    )


def _default_extraction(user: str) -> ExtractionOut:
    """Spread findings across every sub-question listed in the prompt.

    A fake that always tagged the first sub-question would leave the others
    permanently uncovered and make the loop look like it never converges.
    """
    listed = re.findall(r"\b(SQ\d+):", user) or ["SQ1"]
    first = listed[0]
    second = listed[1 % len(listed)]
    return ExtractionOut(
        evidence=[
            EvidenceOut(
                sub_question_id=first,
                claim="Fraud datasets are severely imbalanced.",
                # Copied verbatim from PAGE_TEXT so quote verification passes.
                quote="Fraud detection datasets are severely imbalanced, with positive "
                "cases often well under one percent of all recorded transactions",
                stance="supports",
                relevance=0.9,
            ),
            EvidenceOut(
                sub_question_id=second,
                claim="Precision-recall curves beat ROC AUC here.",
                quote="precision-recall curves are a more informative evaluation than "
                "ROC AUC under heavy imbalance",
                stance="supports",
                relevance=0.8,
            ),
        ]
    )


def _default_coverage(_: str) -> CoverageOut:
    return CoverageOut(
        weak_sub_question_ids=[], contradictions=[], missing_angles=[], reasoning="looks fine"
    )


def _default_followups(_: str) -> FollowupsOut:
    return FollowupsOut(followups=[FollowupOut(text="What are the deployment costs?", gap="cost")])


def _default_report(user: str) -> ReportOut:
    """Reference whatever evidence ids the package actually offered.

    A fake that hard-coded ids would keep passing while real resolution broke,
    which is exactly the failure the evidence-first design exists to catch.
    """
    offered = re.findall(r"^- (S\d+-e\d+)", user, flags=re.MULTILINE)
    primary = offered[:1]
    secondary = offered[1:2] or primary
    return ReportOut(
        title="Fraud detection on imbalanced data",
        summary_claims=[
            ClaimOut(
                text="Resampling and cost-sensitive learning are the main approaches",
                evidence_ids=primary,
                kind="factual",
            )
        ],
        sections=[
            SectionOut(
                heading="Approaches",
                claims=[
                    ClaimOut(
                        text="Datasets are severely imbalanced, under one percent positive",
                        evidence_ids=primary,
                        kind="factual",
                    ),
                    ClaimOut(
                        text="This section compares the approaches.",
                        evidence_ids=[],
                        kind="framing",
                    ),
                ],
            )
        ],
        key_findings=[
            ClaimOut(
                text="Precision-recall is more informative than ROC AUC",
                evidence_ids=secondary,
                kind="factual",
            )
        ],
        contradictions=[],
        limitations=[],
    )


def _default_entailment(_: str) -> EntailmentOut:
    return EntailmentOut(verdict="supported", reason="the quote states it")


_DEFAULTS = {
    "AnalysisOut": _default_analysis,
    "PlanOut": _default_plan,
    "QueriesOut": _default_queries,
    "ExtractionOut": _default_extraction,
    "CoverageOut": _default_coverage,
    "FollowupsOut": _default_followups,
    "ReportOut": _default_report,
    "EntailmentOut": _default_entailment,
}


class FakeRouter:
    """Drop-in replacement for ModelRouter."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.calls: list[tuple[ModelRole, str]] = []
        self.responses: dict[str, Any] = {}
        self.fail_schemas: set[str] = set()
        self.tracker = UsageTracker(max_calls=10_000)
        self._spec = ModelSpec(provider=Provider.OLLAMA, model="fake:1b")

    def get(self, role: ModelRole) -> FakeRoleModel:
        return FakeRoleModel(role, self)

    def assignments(self) -> dict[ModelRole, ModelSpec]:
        return dict.fromkeys(ModelRole, self._spec)

    def describe(self) -> dict[str, str]:
        return {role.value: str(self._spec) for role in ModelRole}

    async def preflight(self) -> list[str]:
        return []

    def count(self, schema_name: str) -> int:
        return sum(1 for _, name in self.calls if name == schema_name)


@dataclass
class FakeSearchService:
    """Returns a fixed set of URLs, with a controllable overlap between queries."""

    results_per_query: int = 3
    overlap: bool = True
    fail_queries: set[str] = field(default_factory=set)
    concurrent_peak: int = 0
    _live: int = 0
    stats: Any = None

    def __post_init__(self) -> None:
        from agentic_research.search.service import SearchStats

        self.stats = SearchStats()

    async def run_query(self, query: SearchQuery) -> SearchResponse:
        self._live += 1
        self.concurrent_peak = max(self.concurrent_peak, self._live)
        await asyncio.sleep(0.01)
        self._live -= 1

        if query.id in self.fail_queries:
            self.stats.failures += 1
            return SearchResponse(
                query_id=query.id,
                query_text=query.text,
                provider="fake",
                results=[],
                error="induced search failure",
            )

        self.stats.calls += 1
        self.stats.credits += 1.0
        results = []
        for i in range(self.results_per_query):
            # With overlap on, every query returns the same shared URL first,
            # which is what makes deduplication observable.
            url = (
                f"https://shared.example.com/page-{i}"
                if self.overlap
                else f"https://{query.id.lower()}.example.com/page-{i}"
            )
            results.append(
                SearchResult(
                    url=url,
                    title=f"Result {i} for {query.text[:20]}",
                    snippet="snippet",
                    score=0.9 - i * 0.1,
                    provider="fake",
                    query_id=query.id,
                    query_text=query.text,
                )
            )
        self.stats.results += len(results)
        return SearchResponse(
            query_id=query.id, query_text=query.text, provider="fake", results=results
        )


@dataclass
class FakeFetcher:
    """Serves a fixed page body, optionally failing specific URLs."""

    fail_urls: set[str] = field(default_factory=set)
    empty_urls: set[str] = field(default_factory=set)
    text: str = PAGE_TEXT
    stats: FetchStats = field(default_factory=FetchStats)
    concurrent_peak: int = 0
    _live: int = 0

    async def fetch(self, url: str) -> FetchResult:
        from agentic_research.models import FetchStatus

        self._live += 1
        self.concurrent_peak = max(self.concurrent_peak, self._live)
        await asyncio.sleep(0.01)
        self._live -= 1
        self.stats.attempted += 1

        if url in self.fail_urls:
            return FetchResult(url=url, status=FetchStatus.HTTP_ERROR, error="induced 500")
        if url in self.empty_urls:
            return FetchResult(url=url, status=FetchStatus.EMPTY, error="no text")
        self.stats.succeeded += 1
        # Vary the body slightly per URL so content-hash dedup does not merge
        # genuinely different pages in tests that do not want that.
        return FetchResult(
            url=url, final_url=url, status=FetchStatus.OK, text=f"{self.text} Source marker: {url}."
        )
