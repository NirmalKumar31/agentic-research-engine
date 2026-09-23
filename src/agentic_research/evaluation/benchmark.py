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
    """One benchmark question and the property it is meant to stress."""

    id: str
    question: str
    why: str
    category: str


# Twelve questions spanning the categories that stress different parts of
# the engine. Deliberately inexpensive: each is answerable from a handful
# of sources, because the point is to exercise the pipeline rather than to
# produce a literature review.
#
# Running all of them against a paid model costs real money. `evaluate -n N`
# takes the first N, and the categories are ordered so a small N still
# covers a spread.
BENCHMARK: tuple[BenchmarkQuestion, ...] = (
    BenchmarkQuestion(
        id="B01",
        category="technical comparison",
        question=(
            "Compare modern approaches for detecting fraud in highly imbalanced "
            "transaction datasets, including their evaluation methodology."
        ),
        why="requires decomposition across sampling, algorithms and evaluation",
    ),
    BenchmarkQuestion(
        id="B02",
        category="multi-dimensional decision",
        question=(
            "Are locally hosted open-weight language models viable for enterprise "
            "document analysis, considering cost, privacy and capability tradeoffs?"
        ),
        why="several independent dimensions; commercially biased sources",
    ),
    BenchmarkQuestion(
        id="B03",
        category="contradictory sources",
        question=(
            "How effective are current techniques for detecting AI-generated text, "
            "and what are their documented failure modes?"
        ),
        why="genuine disagreement in the literature; tests contradiction capture",
    ),
    BenchmarkQuestion(
        id="B04",
        category="quantitative claims",
        question=(
            "What measured latency and throughput differences are reported between "
            "INT8 and FP16 inference for transformer models on GPUs?"
        ),
        why="numeric claims; quote fidelity matters most where figures are quoted",
    ),
    BenchmarkQuestion(
        id="B05",
        category="academic / PDF",
        question=(
            "What do published papers report about the sample efficiency of "
            "parameter-efficient fine-tuning compared with full fine-tuning?"
        ),
        why="arXiv-heavy, so it exercises PDF extraction and page citations",
    ),
    BenchmarkQuestion(
        id="B06",
        category="time sensitive",
        question=(
            "What changed in the EU AI Act obligations that took effect most "
            "recently, and who do they apply to?"
        ),
        why="recency matters; stale sources should be visibly penalised",
    ),
    BenchmarkQuestion(
        id="B07",
        category="primary-source heavy",
        question=(
            "What does the official Kubernetes documentation specify about pod "
            "eviction behaviour under memory pressure?"
        ),
        why="a correct answer should cite first-party docs, not blog summaries",
    ),
    BenchmarkQuestion(
        id="B08",
        category="sparse evidence",
        question=(
            "What is documented about failure rates of automated citation "
            "verification systems in production deployments?"
        ),
        why="little good evidence exists; the run should admit that, not invent it",
    ),
    BenchmarkQuestion(
        id="B09",
        category="technical comparison",
        question=(
            "What are the practical differences between vector databases and "
            "traditional search engines for retrieval-augmented generation?"
        ),
        why="vendor-heavy topic; tests source diversity",
    ),
    BenchmarkQuestion(
        id="B10",
        category="multi-dimensional decision",
        question=(
            "What are the operational tradeoffs between Kubernetes and serverless "
            "platforms for machine learning inference workloads?"
        ),
        why="comparison with heavy marketing noise around it",
    ),
    BenchmarkQuestion(
        id="B11",
        category="quantitative claims",
        question=(
            "What reduction in labelling effort do published active learning "
            "studies report for text classification tasks?"
        ),
        why="reported percentages vary widely; tests whether spread is preserved",
    ),
    BenchmarkQuestion(
        id="B12",
        category="contradictory sources",
        question=(
            "Do published results support or contradict the claim that retrieval "
            "augmentation reduces hallucination in language models?"
        ),
        why="the literature genuinely splits; a single-sided answer is a failure",
    ),
)

CATEGORIES = tuple(dict.fromkeys(q.category for q in BENCHMARK))


@dataclass
class QuestionResult:
    question_id: str
    question: str
    ok: bool
    category: str = ""
    metrics: list[Metric] = field(default_factory=list)
    run_metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def value(self, name: str) -> float | None:
        return next((m.value for m in self.metrics if m.name == name), None)


@dataclass
class BenchmarkReport:
    started_at: str
    mode: str
    environment: dict[str, Any] = field(default_factory=dict)
    results: list[QuestionResult] = field(default_factory=list)

    def aggregate(self) -> dict[str, float | None]:
        """Mean of each metric across questions that produced a value."""
        names = [m.name for r in self.results if r.ok for m in r.metrics]
        summary: dict[str, float | None] = {}
        for name in dict.fromkeys(names):
            values = [v for r in self.results if r.ok and (v := r.value(name)) is not None]
            summary[name] = round(statistics.mean(values), 4) if values else None
        return summary

    def by_category(self) -> dict[str, dict[str, float | None]]:
        """Mean of each metric within each category.

        A single aggregate hides that, say, quote fidelity is fine on
        technical comparisons and poor on quantitative claims.
        """
        out: dict[str, dict[str, float | None]] = {}
        for category in dict.fromkeys(r.category for r in self.results if r.ok):
            rows = [r for r in self.results if r.ok and r.category == category]
            names = dict.fromkeys(m.name for r in rows for m in r.metrics)
            out[category] = {}
            for name in names:
                values = [v for r in rows if (v := r.value(name)) is not None]
                out[category][name] = round(statistics.mean(values), 4) if values else None
        return out

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
            "environment": self.environment,
            "totals": self.totals(),
            "aggregate": self.aggregate(),
            "by_category": self.by_category(),
            "per_question": [
                {
                    "id": r.question_id,
                    "category": r.category,
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
                QuestionResult(
                    question.id,
                    question.question,
                    ok=False,
                    category=question.category,
                    error=str(exc)[:300],
                )
            )
            continue

        run_metrics = result.metrics.model_dump(mode="json")
        report.results.append(
            QuestionResult(
                question_id=question.id,
                question=question.question,
                ok=True,
                category=question.category,
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
