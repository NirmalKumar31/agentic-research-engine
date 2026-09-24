"""Cross-attribution experiment: what narrowing extraction actually costs.

78% of evidence in the measured run is ``cross_attributed`` -- it answers a
sub-question whose queries never retrieved that source. The cause is
structural: :func:`~agentic_research.graph.routing.dispatch_extraction` shows
every source every open sub-question, so the extractor is invited to find
something for a question that page was never fetched for.

Narrowing extraction shows each source only the sub-questions it was
retrieved for. That would drive cross-attribution to zero **by construction**,
which is why "does it work?" is the wrong question -- it cannot fail. The
question this module exists to answer is what it *costs*: how much citable
evidence disappears, how many sub-questions lose coverage, and how many
sources stop earning their fetch.

Three strategies, measured over one frozen corpus so every arm reads
byte-identical source text:

======================  =========================================================
``all_open`` (A)        Every open sub-question. The production default.
``retrieved_only`` (B)  Only sub-questions a query retrieved this source for.
``adjacent`` (C)        B, plus up to *k* lexically nearest other questions.
======================  =========================================================

Each arm calls the real :func:`~agentic_research.graph.nodes.research.extract_worker`
through a real graph. Nothing is reimplemented: a hand-rolled extraction loop
would measure the loop, not the strategy.

**Nothing here changes production behaviour.** Strategy A remains the default
until there is a measurement to argue against it.
"""

from __future__ import annotations

import re
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agentic_research.config import ModelRole, Provider, Settings
from agentic_research.environment import capture as capture_environment
from agentic_research.evaluation.ab import EvidenceCorpus
from agentic_research.evaluation.evaluators import evidence_coverage
from agentic_research.graph.nodes.research import extract_worker
from agentic_research.graph.state import (
    ExtractTask,
    ResearchState,
    RunContext,
    initial_state,
)
from agentic_research.llm.base import UsageTracker
from agentic_research.llm.router import ModelRouter
from agentic_research.models import (
    EvidenceItem,
    QuoteMatch,
    SearchQuery,
    SourceDocument,
    SubQuestion,
)
from agentic_research.observability import get_logger
from agentic_research.retrieval.fetcher import PageFetcher
from agentic_research.search.service import SearchStats

log = get_logger(__name__)


class Strategy(StrEnum):
    """Which sub-questions an extractor is shown for a given source."""

    ALL_OPEN = "all_open"
    """Every open sub-question, regardless of what retrieved the source."""
    RETRIEVED_ONLY = "retrieved_only"
    """Only the sub-questions this source was genuinely retrieved for."""
    ADJACENT = "adjacent"
    """Retrieved sub-questions plus a bounded set of lexically nearest others."""

    @property
    def letter(self) -> str:
        return {"all_open": "A", "retrieved_only": "B", "adjacent": "C"}[self.value]

    @property
    def description(self) -> str:
        return {
            "all_open": "every open sub-question (production default)",
            "retrieved_only": "only sub-questions whose queries retrieved this source",
            "adjacent": "retrieved sub-questions plus the k nearest others",
        }[self.value]


class CloudSpendRefused(Exception):
    """The experiment would have spent cloud credits without being asked to.

    One pass is one extraction call per source. Three strategies over three
    repeats multiplies that by nine before anyone notices, and an experiment
    that quietly bills is not one you can run casually. Opting in is
    explicit.
    """


class UnusableCorpusError(Exception):
    """The corpus cannot support this experiment.

    Raised rather than logged, for the same reason
    :class:`~agentic_research.evaluation.ab.DegradedCorpusError` is: a corpus
    with no source text produces three arms of zeros that read like a
    finding.
    """


# ---------------------------------------------------------------------------
# Adjacency (strategy C)
# ---------------------------------------------------------------------------
#
# Deliberately lexical and deterministic. An embedding or an LLM judge would
# make the adjacency set itself a moving part, and then a difference between
# arms could be the adjacency model rather than the strategy. Crude and
# reproducible beats clever and unpinned here.

_WORD = re.compile(r"[a-z0-9]+")

# Kept as prose and split at import rather than written as a list literal:
# the words are easier to review and amend in this form, and the cost is one
# split at module load.
_STOPWORD_TEXT = """
    the a an and or but for nor so yet of to in on at by with from into over
    under about as is are was were be been being do does did have has had
    what which who whom whose when where why how much many more most other
    such can could should would may might will shall than that this these
    those their there they them its it not no any all some each both between
    during before after above below out off again further then once
"""

_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def _terms(text: str) -> set[str]:
    """Content words of a sub-question, for overlap scoring."""
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _overlap(left: set[str], right: set[str]) -> float:
    """Jaccard similarity. Zero when either side has no content words."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def sub_questions_for(
    strategy: Strategy,
    source: SourceDocument,
    sub_questions: Sequence[SubQuestion],
    *,
    adjacent_k: int = 2,
) -> list[SubQuestion]:
    """The sub-question set a given strategy puts in front of one source.

    Corpus order is preserved in every arm. Order is not cosmetic: it is the
    order the extractor reads them in, and the worker attributes an
    out-of-range id to the *first* entry, so reordering would introduce a
    difference that has nothing to do with the strategy being tested.
    """
    if strategy is Strategy.ALL_OPEN:
        return list(sub_questions)

    retrieved = {d.sub_question_id for d in source.discovered_by}
    chosen = {q.id for q in sub_questions if q.id in retrieved}

    if strategy is Strategy.ADJACENT and adjacent_k > 0:
        retrieved_terms = [_terms(q.text) for q in sub_questions if q.id in chosen]
        scored: list[tuple[float, str]] = []
        for question in sub_questions:
            if question.id in chosen:
                continue
            terms = _terms(question.text)
            best = max((_overlap(terms, other) for other in retrieved_terms), default=0.0)
            if best > 0.0:
                scored.append((best, question.id))
        # Sorted by id as the tiebreak so an arm is reproducible when two
        # questions score identically.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        chosen.update(qid for _, qid in scored[:adjacent_k])

    return [q for q in sub_questions if q.id in chosen]


# ---------------------------------------------------------------------------
# Corpus eligibility
# ---------------------------------------------------------------------------


def eligible_sources(corpus: EvidenceCorpus) -> list[SourceDocument]:
    """Sources every arm can read.

    A source with no ``discovered_by`` entry has no retrieved sub-question,
    so strategies B and C would have nothing to show it. Rather than let one
    arm silently fall back to A for that source -- which would make the arms
    incomparable -- such sources are dropped from **every** arm and counted.
    """
    return [s for s in corpus.sources if s.is_usable and not s.duplicate_of and s.discovered_by]


def validate_corpus_for_extraction(corpus: EvidenceCorpus) -> list[str]:
    """Problems that would make the comparison meaningless.

    Distinct from :meth:`EvidenceCorpus.validate_for_replay`, which cares
    whether the *frozen evidence* can be cited. This experiment discards the
    frozen evidence and re-extracts, so what matters instead is whether the
    sources still carry text and whether the strategies can actually differ.
    """
    problems: list[str] = []

    without_text = [s.id for s in corpus.sources if not s.text.strip()]
    if without_text:
        problems.append(
            f"{len(without_text)} source(s) have no text ({', '.join(without_text[:5])}). "
            "Extraction would read nothing. Build the corpus with "
            "`agentic-research freeze`, which keeps source text."
        )

    usable = eligible_sources(corpus)
    if not usable:
        problems.append("no usable source carries a discovery path; every arm would be empty")

    if len(corpus.sub_questions) < 2:
        problems.append(
            f"{len(corpus.sub_questions)} sub-question(s): narrowing cannot differ from "
            "showing everything, so the comparison has nothing to measure"
        )

    if usable and all(
        len({d.sub_question_id for d in s.discovered_by}) == len(corpus.sub_questions)
        for s in usable
    ):
        problems.append(
            "every source was retrieved for every sub-question, so strategy B is "
            "identical to strategy A on this corpus"
        )

    return problems


# ---------------------------------------------------------------------------
# One pass
# ---------------------------------------------------------------------------


@dataclass
class PassResult:
    """One strategy, run once over the corpus."""

    strategy: Strategy
    repeat: int
    ok: bool
    evidence: list[EvidenceItem] = field(default_factory=list)
    shown_per_source: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0
    llm_calls: int = 0
    provider_requests: int = 0
    failed_provider_requests: int = 0
    rate_limit_refusals: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    known_cost_usd: float = 0.0
    cost_is_complete: bool = True
    extraction_failures: int = 0
    error: str = ""


def _begin(state: ResearchState) -> ResearchState:
    """Entry node.

    A conditional edge needs a concrete source node, and this mirrors how the
    production graph dispatches extraction from ``register_sources``.
    """
    return {}


def _collect(state: ResearchState) -> ResearchState:
    """Fan-in barrier. The ``evidence`` reducer has already done the work."""
    return {}


def _extraction_graph(
    dispatch: Callable[[ResearchState], list[Send] | str],
) -> Any:
    """A two-stage graph: fan out to the real extractor, then wait."""
    graph = StateGraph(ResearchState, context_schema=RunContext)
    graph.add_node("begin", _begin)
    graph.add_node("extract_worker", extract_worker, input_schema=ExtractTask)
    graph.add_node("collect", _collect, defer=True)
    graph.add_edge(START, "begin")
    # The fallback destination is declared for the same reason the production
    # graph declares one: an empty Send list ends the graph silently.
    graph.add_conditional_edges("begin", dispatch, ["extract_worker", "collect"])
    graph.add_edge("extract_worker", "collect")
    graph.add_edge("collect", END)
    return graph.compile()


async def run_pass(
    strategy: Strategy,
    corpus: EvidenceCorpus,
    settings: Settings,
    *,
    repeat: int = 1,
    adjacent_k: int = 2,
) -> PassResult:
    """Extract from every eligible source once, under one strategy.

    Budget accounting is per pass, so each arm's cost is attributable rather
    than pooled.
    """
    sources = eligible_sources(corpus)
    shown = {
        source.id: sub_questions_for(strategy, source, corpus.sub_questions, adjacent_k=adjacent_k)
        for source in sources
    }

    def dispatch(state: ResearchState) -> list[Send] | str:
        sends = [
            Send("extract_worker", {"source": source, "sub_questions": shown[source.id]})
            for source in sources
            if shown[source.id]
        ]
        return sends or "collect"

    router = ModelRouter(
        settings,
        UsageTracker(
            settings.max_llm_calls,
            settings.cloud_budget,
            max_provider_requests=settings.max_provider_requests,
        ),
    )

    state = initial_state(f"attr-{strategy.value}-{repeat}", corpus.question)
    state.update({"sub_questions": list(corpus.sub_questions), "sources": list(sources)})

    started = time.perf_counter()
    async with PageFetcher(settings) as fetcher:

        class _UnusedSearch:
            """Extraction issues no searches; this makes that explicit."""

            stats = SearchStats()

            async def run_query(self, query: SearchQuery) -> Any:  # pragma: no cover
                raise AssertionError("the attribution experiment must not search")

        context = RunContext(
            settings=settings,
            router=router,
            search=_UnusedSearch(),  # type: ignore[arg-type]
            fetcher=fetcher,
            budget=settings.budget,
            run_id=f"attr-{strategy.value}-{repeat}",
        )

        try:
            final = await _extraction_graph(dispatch).ainvoke(state, context=context)
        except Exception as exc:
            log.warning(
                "attribution_pass_failed",
                strategy=strategy.value,
                repeat=repeat,
                error=str(exc)[:200],
            )
            return PassResult(
                strategy=strategy,
                repeat=repeat,
                ok=False,
                shown_per_source={sid: len(qs) for sid, qs in shown.items()},
                duration_s=round(time.perf_counter() - started, 2),
                error=str(exc)[:300],
            )

    totals = router.tracker.totals()
    counters = final.get("counters", {}) or {}
    return PassResult(
        strategy=strategy,
        repeat=repeat,
        ok=True,
        evidence=list(final.get("evidence", []) or []),
        shown_per_source={sid: len(qs) for sid, qs in shown.items()},
        duration_s=round(time.perf_counter() - started, 2),
        llm_calls=totals.calls,
        provider_requests=totals.provider_requests,
        failed_provider_requests=totals.failed_provider_requests,
        rate_limit_refusals=totals.rate_limit_refusals,
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        known_cost_usd=totals.known_cost_usd,
        cost_is_complete=totals.cost_is_complete,
        extraction_failures=int(counters.get("extraction_failed", 0))
        + int(counters.get("extraction_skipped", 0)),
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure(result: PassResult, corpus: EvidenceCorpus) -> dict[str, float | int | bool | None]:
    """Turn one pass into the numbers the experiment is about.

    Rates are over the pass's own evidence, so they are comparable between
    arms that produced different amounts of it. Absolute counts are reported
    alongside, because a rate on three items is not a measurement.
    """
    sources = eligible_sources(corpus)
    evidence = result.evidence
    citable = [e for e in evidence if e.is_citable]

    state = {
        "sub_questions": list(corpus.sub_questions),
        "sources": sources,
        "evidence": evidence,
    }
    coverage = evidence_coverage(state)

    answered = {e.sub_question_id for e in citable}
    earning = {e.source_id for e in citable}
    shown = list(result.shown_per_source.values())

    total = len(evidence)
    return {
        # -- what the extractor produced ---------------------------------
        "evidence_items": total,
        "citable_items": len(citable),
        "quote_fidelity": round(len(citable) / total, 4) if total else None,
        "quote_drift": (
            round(sum(1 for e in evidence if e.quote_match is QuoteMatch.FUZZY) / total, 4)
            if total
            else None
        ),
        # -- the thing being traded away ---------------------------------
        "cross_attribution_rate": (
            round(sum(1 for e in evidence if e.cross_attributed) / total, 4) if total else None
        ),
        "citable_cross_attribution_rate": (
            round(sum(1 for e in citable if e.cross_attributed) / len(citable), 4)
            if citable
            else None
        ),
        # -- the thing being traded for ----------------------------------
        "evidence_coverage": coverage.value,
        "sub_questions_with_citable_evidence": (
            round(len(answered) / len(corpus.sub_questions), 4) if corpus.sub_questions else None
        ),
        "source_utilisation": round(len(earning) / len(sources), 4) if sources else None,
        # -- what it cost -------------------------------------------------
        "mean_sub_questions_shown": round(statistics.fmean(shown), 2) if shown else 0.0,
        "extraction_calls": result.llm_calls,
        "provider_requests": result.provider_requests,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "known_cost_usd": round(result.known_cost_usd, 6),
        "cost_is_complete": result.cost_is_complete,
        "duration_s": result.duration_s,
        # A pass throttled by 429s produces less evidence for a reason that
        # has nothing to do with its strategy, so it has to be visible or the
        # arm looks simply worse.
        "extraction_failures": result.extraction_failures,
        "failed_provider_requests": result.failed_provider_requests,
        "rate_limit_refusals": result.rate_limit_refusals,
    }


def invariant_breaches(result: PassResult) -> list[str]:
    """Checks that must hold if the harness itself is correct.

    Strategy B shows a source only the sub-questions it was retrieved for, so
    every item it produces has a discovery path and cross-attribution must be
    exactly zero. If it is not, the harness is measuring something other than
    what it claims and the numbers should not be read.
    """
    breaches: list[str] = []
    if result.strategy is Strategy.RETRIEVED_ONLY:
        stray = [e.id for e in result.evidence if e.cross_attributed]
        if stray:
            breaches.append(
                f"retrieved_only produced {len(stray)} cross-attributed item(s) "
                f"({', '.join(stray[:5])}); this is impossible by construction, so "
                "the sub-question sets or the discovery refs are wrong"
            )
    return breaches


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


@dataclass
class Experiment:
    question: str
    corpus_summary: str
    environment: dict[str, Any]
    strategies: list[Strategy]
    repeats: int
    adjacent_k: int
    eligible_sources: int
    excluded_sources: list[str] = field(default_factory=list)
    passes: list[PassResult] = field(default_factory=list)
    measurements: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    breaches: list[str] = field(default_factory=list)

    def aggregate(self, strategy: Strategy) -> dict[str, Any]:
        """Mean, min and max across repeats, for each numeric measurement.

        ``n`` is reported alongside because a metric is ``None`` when its
        denominator was empty, and a mean over two of three repeats is not
        the same claim as a mean over three.
        """
        rows = [r for r in self.measurements.get(strategy.value, []) if not r.get("failed")]
        if not rows:
            return {"repeats": 0, "all_passes_failed": True}

        summary: dict[str, Any] = {"repeats": len(rows)}
        for key in [k for k in rows[0] if k != "repeat" and all(k in r for r in rows)]:
            raw = [r[key] for r in rows]
            numeric = [v for v in raw if isinstance(v, int | float) and not isinstance(v, bool)]
            if not numeric:
                summary[key] = raw[0]
                continue
            summary[key] = {
                "mean": round(statistics.fmean(numeric), 4),
                "min": min(numeric),
                "max": max(numeric),
                "n": len(numeric),
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": "cross_attribution",
            "question": self.question,
            "corpus": self.corpus_summary,
            "captured_at": datetime.now(UTC).isoformat(),
            "environment": self.environment,
            "design": {
                "eligible_sources": self.eligible_sources,
                "excluded_sources": self.excluded_sources,
                "repeats": self.repeats,
                "adjacent_k": self.adjacent_k,
                "strategies": {
                    s.value: {"letter": s.letter, "description": s.description}
                    for s in self.strategies
                },
            },
            "note": (
                "Every arm extracted from the same frozen sources with the same "
                "extract_worker; only the sub-question set shown to it differs. "
                "retrieved_only reaches 0% cross-attribution by construction, not "
                "by working -- the measurement of interest is what it costs in "
                "citable evidence, coverage and source utilisation."
            ),
            "invariant_breaches": self.breaches,
            "per_repeat": dict(sorted(self.measurements.items())),
            "aggregate": {s.value: self.aggregate(s) for s in self.strategies},
        }


async def run_experiment(
    corpus: EvidenceCorpus,
    settings: Settings,
    *,
    strategies: Sequence[Strategy] | None = None,
    repeats: int = 1,
    adjacent_k: int = 2,
    allow_cloud: bool = False,
) -> Experiment:
    """Run each strategy over the same corpus, sequentially.

    Sequential for the same reason :func:`~agentic_research.evaluation.ab.compare`
    is: the arms contend for one local model and one rate limit, so running
    them together would measure queueing.

    Refuses to spend cloud credits unless ``allow_cloud=True`` is passed
    deliberately. The refusal lives here rather than only in the CLI so a
    Python caller cannot bypass it.
    """
    chosen = list(strategies) if strategies else list(Strategy)

    problems = validate_corpus_for_extraction(corpus)
    if problems:
        raise UnusableCorpusError(
            "corpus cannot support the cross-attribution experiment: " + "; ".join(problems)
        )

    spec = ModelRouter(settings).assignments()[ModelRole.RESEARCHER]
    if spec.provider is not Provider.OLLAMA and not allow_cloud:
        sources = len(eligible_sources(corpus))
        raise CloudSpendRefused(
            f"the researcher role resolves to {spec}, and this experiment would issue "
            f"about {sources * len(chosen) * repeats} paid extraction call(s) "
            f"({sources} sources x {len(chosen)} strategies x {repeats} repeat(s)). "
            "Re-run with LLM_MODE=local, or pass allow_cloud=True to accept the spend."
        )

    usable = eligible_sources(corpus)
    excluded = [
        s.id for s in corpus.sources if s.is_usable and not s.duplicate_of and not s.discovered_by
    ]
    if excluded:
        log.info("attribution_sources_excluded", ids=excluded, reason="no discovery path")

    experiment = Experiment(
        question=corpus.question,
        corpus_summary=corpus.summary(),
        environment=capture_environment(settings),
        strategies=chosen,
        repeats=repeats,
        adjacent_k=adjacent_k,
        eligible_sources=len(usable),
        excluded_sources=excluded,
    )

    for strategy in chosen:
        rows: list[dict[str, Any]] = []
        for repeat in range(1, repeats + 1):
            log.info("attribution_pass_started", strategy=strategy.value, repeat=repeat)
            result = await run_pass(
                strategy, corpus, settings, repeat=repeat, adjacent_k=adjacent_k
            )
            experiment.passes.append(result)
            experiment.breaches.extend(invariant_breaches(result))
            if result.ok:
                rows.append({"repeat": repeat, **measure(result, corpus)})
            else:
                rows.append({"repeat": repeat, "failed": True, "error": result.error})
        experiment.measurements[strategy.value] = rows

    return experiment
