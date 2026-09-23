"""Benchmark question set and evaluation harness.

The questions are chosen to need decomposition and multiple sources. A
question answerable from one page measures nothing about a research engine.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_research.config import Settings
from agentic_research.evaluation.evaluators import Metric, evaluate_run
from agentic_research.observability import get_logger
from agentic_research.runner import run_research

log = get_logger(__name__)


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    question: str
    why: str
    """What this question is meant to stress."""


BENCHMARK: tuple[BenchmarkQuestion, ...] = (
    BenchmarkQuestion(
        id="B1",
        question=(
            "Compare modern approaches for detecting fraud in highly imbalanced "
            "transaction datasets, including their evaluation methodology."
        ),
        why="requires decomposition across sampling, algorithms and evaluation",
    ),
    BenchmarkQuestion(
        id="B2",
        question=(
            "Are locally hosted open-weight language models viable for enterprise "
            "document analysis, considering cost, privacy and capability tradeoffs?"
        ),
        why="multi-dimensional tradeoff question with commercially biased sources",
    ),
    BenchmarkQuestion(
        id="B3",
        question=(
            "What are the practical differences between vector databases and "
            "traditional search engines for retrieval-augmented generation?"
        ),
        why="vendor-heavy topic; tests source diversity and contradiction handling",
    ),
    BenchmarkQuestion(
        id="B4",
        question=(
            "How effective are current techniques for detecting AI-generated text, "
            "and what are their documented failure modes?"
        ),
        why="genuine disagreement in the literature; tests contradiction preservation",
    ),
    BenchmarkQuestion(
        id="B5",
        question=(
            "What are the operational tradeoffs between Kubernetes and serverless "
            "platforms for machine learning inference workloads?"
        ),
        why="comparison with strong marketing noise around it",
    ),
)


@dataclass
class QuestionResult:
    question_id: str
    question: str
    ok: bool
    metrics: list[Metric] = field(default_factory=list)
    run_metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def value(self, name: str) -> float | None:
        return next((m.value for m in self.metrics if m.name == name), None)


@dataclass
class BenchmarkReport:
    started_at: str
    mode: str
    results: list[QuestionResult] = field(default_factory=list)

    def aggregate(self) -> dict[str, float | None]:
        """Mean of each metric across questions that produced a value."""
        names = [m.name for r in self.results if r.ok for m in r.metrics]
        summary: dict[str, float | None] = {}
        for name in dict.fromkeys(names):
            values = [v for r in self.results if r.ok and (v := r.value(name)) is not None]
            summary[name] = round(statistics.mean(values), 4) if values else None
        return summary

    def totals(self) -> dict[str, Any]:
        ok = [r for r in self.results if r.ok]
        return {
            "questions": len(self.results),
            "succeeded": len(ok),
            "failed": len(self.results) - len(ok),
            "total_llm_calls": sum(r.run_metrics.get("llm_calls", 0) for r in ok),
            "total_sources": sum(r.run_metrics.get("unique_sources", 0) for r in ok),
            "total_cost_usd": round(sum(r.run_metrics.get("known_cost_usd", 0.0) for r in ok), 4),
            "mean_duration_s": round(
                statistics.mean([r.run_metrics.get("duration_s", 0.0) for r in ok]), 1
            )
            if ok
            else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "mode": self.mode,
            "totals": self.totals(),
            "aggregate": self.aggregate(),
            "per_question": [
                {
                    "id": r.question_id,
                    "question": r.question,
                    "ok": r.ok,
                    "error": r.error,
                    "metrics": {m.name: m.value for m in r.metrics},
                    "details": {m.name: m.detail for m in r.metrics},
                    "run": r.run_metrics,
                }
                for r in self.results
            ],
        }


async def run_benchmark(
    settings: Settings,
    questions: tuple[BenchmarkQuestion, ...] = BENCHMARK,
    *,
    limit: int | None = None,
    on_question: Any = None,
) -> BenchmarkReport:
    """Run the benchmark and score every run.

    Questions run sequentially rather than concurrently. They contend for the
    same rate limits and, in local mode, the same GPU, so running them in
    parallel would measure queueing rather than the engine.
    """
    selected = questions[:limit] if limit else questions
    report = BenchmarkReport(started_at=datetime.now(UTC).isoformat(), mode=settings.llm_mode.value)

    for question in selected:
        if on_question is not None:
            on_question(question)
        log.info("benchmark_question_started", id=question.id)
        try:
            # Benchmarks check every eligible claim. Sampling here and
            # publishing the result as a quality figure would misrepresent it.
            result = await run_research(question.question, settings, exhaustive_verification=True)
        except Exception as exc:
            log.warning("benchmark_question_failed", id=question.id, error=str(exc)[:200])
            report.results.append(
                QuestionResult(question.id, question.question, ok=False, error=str(exc)[:300])
            )
            continue

        run_metrics = result.metrics.model_dump(mode="json")
        report.results.append(
            QuestionResult(
                question_id=question.id,
                question=question.question,
                ok=True,
                metrics=evaluate_run(result.state, run_metrics),
                run_metrics=run_metrics,
            )
        )
    return report


def write_report(report: BenchmarkReport, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = directory / f"benchmark-{report.mode}-{stamp}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return path
