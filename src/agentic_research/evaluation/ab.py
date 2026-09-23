"""Controlled model comparison over a frozen evidence corpus.

Running the whole graph twice with two different models does not compare the
models: search results move between runs, so the two reports rest on
different evidence and the difference in quality is confounded with the
difference in sources.

This module freezes one run's retrieval output -- sub-questions, sources and
evidence, exactly as gathered -- and replays only the downstream stages
(synthesis, citation resolution, verification) against it. Both arms then
see byte-identical input and the remaining difference is attributable to the
model.

Verification is exhaustive here, not sampled: the comparison is the point,
and a sampled figure would add variance that has nothing to do with the
models being compared.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_research.config import ModelRole, ModelSpec, Settings
from agentic_research.environment import capture as capture_environment
from agentic_research.evaluation.evaluators import Metric, evaluate_run
from agentic_research.graph.nodes.reporting import synthesize_report, verify_citations
from agentic_research.graph.state import ResearchState, RunContext, initial_state
from agentic_research.llm.base import UsageTracker
from agentic_research.llm.router import ModelRouter
from agentic_research.models import (
    CoverageAssessment,
    EvidenceItem,
    SearchQuery,
    SourceDocument,
    SubQuestion,
)
from agentic_research.observability import get_logger
from agentic_research.retrieval.fetcher import PageFetcher
from agentic_research.runner import RunResult

log = get_logger(__name__)


@dataclass
class EvidenceCorpus:
    """One run's retrieval output, frozen for replay."""

    question: str
    sub_questions: list[SubQuestion]
    sources: list[SourceDocument]
    evidence: list[EvidenceItem]
    completed_queries: list[SearchQuery] = field(default_factory=list)
    coverage: CoverageAssessment | None = None
    captured_at: str = ""

    @classmethod
    def from_result(cls, question: str, result: RunResult) -> EvidenceCorpus:
        state = result.state
        return cls(
            question=question,
            sub_questions=list(state.get("sub_questions", [])),
            sources=list(state.get("sources", [])),
            evidence=list(state.get("evidence", [])),
            completed_queries=list(state.get("completed_queries", [])),
            coverage=state.get("coverage"),
            captured_at=datetime.now(UTC).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "captured_at": self.captured_at,
            "sub_questions": [q.model_dump(mode="json") for q in self.sub_questions],
            "sources": [s.model_dump(mode="json") for s in self.sources],
            "evidence": [e.model_dump(mode="json") for e in self.evidence],
            "completed_queries": [q.model_dump(mode="json") for q in self.completed_queries],
            "coverage": self.coverage.model_dump(mode="json") if self.coverage else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvidenceCorpus:
        coverage = payload.get("coverage")
        return cls(
            question=payload["question"],
            captured_at=payload.get("captured_at", ""),
            sub_questions=[SubQuestion.model_validate(q) for q in payload["sub_questions"]],
            sources=[SourceDocument.model_validate(s) for s in payload["sources"]],
            evidence=[EvidenceItem.model_validate(e) for e in payload["evidence"]],
            completed_queries=[
                SearchQuery.model_validate(q) for q in payload.get("completed_queries", [])
            ],
            coverage=CoverageAssessment.model_validate(coverage) if coverage else None,
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> EvidenceCorpus:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def summary(self) -> str:
        citable = sum(1 for e in self.evidence if e.is_citable)
        return (
            f"{len(self.sub_questions)} sub-questions, {len(self.sources)} sources, "
            f"{len(self.evidence)} evidence items ({citable} citable)"
        )


@dataclass
class ArmResult:
    """One model configuration's output over the shared corpus."""

    label: str
    models: dict[str, str]
    ok: bool
    markdown: str = ""
    metrics: list[Metric] = field(default_factory=list)
    duration_s: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    known_cost_usd: float = 0.0
    cost_is_complete: bool = True
    error: str = ""
    verification: dict[str, Any] = field(default_factory=dict)

    def value(self, name: str) -> float | None:
        return next((m.value for m in self.metrics if m.name == name), None)


async def run_arm(
    label: str,
    corpus: EvidenceCorpus,
    settings: Settings,
    *,
    overrides: dict[ModelRole, str] | None = None,
) -> ArmResult:
    """Synthesise and verify once over the frozen corpus.

    Only the downstream stages run: no search, no fetching, no extraction.
    A PageFetcher is constructed because RunContext requires one, but
    nothing in this path calls it.
    """
    applied = dict(overrides or {})
    if applied:
        field_names = {
            ModelRole.PLANNER: "planner_model",
            ModelRole.RESEARCHER: "researcher_model",
            ModelRole.CRITIC: "critic_model",
            ModelRole.SYNTHESIZER: "synthesizer_model",
            ModelRole.VERIFIER: "verifier_model",
        }
        settings = settings.model_copy(
            update={field_names[role]: spec for role, spec in applied.items()}
        )

    router = ModelRouter(settings, UsageTracker(settings.max_llm_calls, settings.cloud_budget))
    state: ResearchState = initial_state("ab", corpus.question, exhaustive_verification=True)
    state.update(
        {
            "sub_questions": corpus.sub_questions,
            "sources": corpus.sources,
            "evidence": corpus.evidence,
            "completed_queries": corpus.completed_queries,
            "coverage": corpus.coverage,
        }
    )

    started = time.perf_counter()
    async with PageFetcher(settings) as fetcher:
        from agentic_research.search.service import SearchStats

        class _UnusedSearch:
            """The replay path issues no searches; this makes that explicit."""

            stats = SearchStats()

            async def run_query(self, query: SearchQuery) -> Any:  # pragma: no cover
                raise AssertionError("replay must not search")

        context = RunContext(
            settings=settings,
            router=router,
            search=_UnusedSearch(),  # type: ignore[arg-type]
            fetcher=fetcher,
            budget=settings.budget,
            run_id=f"ab-{label}",
        )

        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(ResearchState, context_schema=RunContext)
        graph.add_node("synthesize", synthesize_report)
        graph.add_node("verify_citations", verify_citations)
        graph.add_edge(START, "synthesize")
        graph.add_edge("synthesize", "verify_citations")
        graph.add_edge("verify_citations", END)

        try:
            final = await graph.compile().ainvoke(state, context=context)
        except Exception as exc:
            log.warning("ab_arm_failed", label=label, error=str(exc)[:200])
            return ArmResult(
                label=label,
                models=router.describe(),
                ok=False,
                duration_s=round(time.perf_counter() - started, 2),
                error=str(exc)[:300],
            )

    duration = time.perf_counter() - started
    totals = router.tracker.totals()
    verification = final.get("verification") or {}

    from agentic_research.report import render_markdown

    report = final.get("report")
    markdown = ""
    if report is not None:
        from agentic_research.models import CitationVerification

        parsed = CitationVerification.model_validate(verification) if verification else None
        markdown = render_markdown(report, corpus.sources, parsed, evidence=corpus.evidence)

    return ArmResult(
        label=label,
        models=router.describe(),
        ok=True,
        markdown=markdown,
        metrics=evaluate_run(final, {}),
        duration_s=round(duration, 2),
        llm_calls=totals.calls,
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        known_cost_usd=totals.known_cost_usd,
        cost_is_complete=totals.cost_is_complete,
        verification=verification,
    )


@dataclass
class Comparison:
    question: str
    corpus_summary: str
    environment: dict[str, Any]
    arms: list[ArmResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "corpus": self.corpus_summary,
            "environment": self.environment,
            "note": (
                "Both arms synthesised from a byte-identical frozen evidence "
                "corpus, so differences are attributable to the model rather "
                "than to changing search results. Verification was exhaustive."
            ),
            "arms": [
                {
                    "label": a.label,
                    "models": a.models,
                    "ok": a.ok,
                    "error": a.error,
                    "duration_s": a.duration_s,
                    "llm_calls": a.llm_calls,
                    "input_tokens": a.input_tokens,
                    "output_tokens": a.output_tokens,
                    "known_cost_usd": a.known_cost_usd,
                    "cost_is_complete": a.cost_is_complete,
                    "metrics": {m.name: m.value for m in a.metrics},
                    "details": {m.name: m.detail for m in a.metrics},
                    "support_breakdown": a.verification.get("support_breakdown", {}),
                }
                for a in self.arms
            ],
        }


async def compare(
    corpus: EvidenceCorpus,
    settings: Settings,
    arms: dict[str, dict[ModelRole, str]],
) -> Comparison:
    """Run each configuration over the same corpus, sequentially.

    Sequential because the arms contend for the same local GPU and the same
    rate limits; running them concurrently would measure queueing.
    """
    comparison = Comparison(
        question=corpus.question,
        corpus_summary=corpus.summary(),
        environment=capture_environment(settings),
    )
    for label, overrides in arms.items():
        log.info("ab_arm_started", label=label)
        comparison.arms.append(await run_arm(label, corpus, settings, overrides=overrides))
    return comparison


def all_roles(spec: str) -> dict[ModelRole, str]:
    """Point every role at one model, for a clean single-variable arm."""
    ModelSpec.parse(spec)  # fail fast on a typo rather than mid-run
    return dict.fromkeys(ModelRole, spec)
