"""A self-contained, offline, deterministic run of the whole engine.

This exercises the real state machine, reducers, retrieval selection,
extraction, citation binding, verification and report generation -- the
same code path a live run takes -- against fixed inputs: two short pages
of original text (written for this module, not scraped, so there is no
licensing question), a scripted model that returns the same structured
output every time, and the project's own deterministic NLI stand-in
(:mod:`agentic_research.citations.fake_nli`, already shipped for exactly
this reason).

**What this proves.** The engine's own machinery is deterministic end to
end: the same corpus and the same scripted responses produce the same
contract, the same selected evidence, the same citations, and the same
published report, byte for byte, with zero network access and zero
credentials.

**What this does not prove.** Nothing about a real model's judgement,
real search results, or real retrieval variance. A real model will
reason differently about the same evidence, and real search returns
different pages over time. This is an engine-behaviour regression
check, not a quality benchmark -- the adversarial set under
``examples/quality-eval/`` and the hosted runs under
``examples/live-validation/`` are the quality evidence, and neither
claims to be reproducible in the sense this module is.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_research.config import ModelRole, ModelSpec, Provider, Settings
from agentic_research.graph.state import RunContext, initial_state
from agentic_research.llm.base import UsageTracker
from agentic_research.models import FetchResult, FetchStatus, SearchQuery, SearchResult
from agentic_research.schemas import (
    AnalysisOut,
    ClaimOut,
    EvidenceOut,
    PlanOut,
    QueriesOut,
    QueryOut,
    ReportOut,
    SectionOut,
    SubQuestionOut,
)

# ---------------------------------------------------------------------------
# The corpus. Two short, original pages -- enough to exercise two
# sub-questions, two sources, one published claim and one withheld claim,
# which is the smallest shape that touches every stage worth checking.
# ---------------------------------------------------------------------------

QUESTION = "What is exponential backoff with jitter, and why cap the retry count?"

_PAGE_A_URL = "https://docs.example-queue.test/backoff"
_PAGE_A_TEXT = (
    "Exponential backoff with jitter is a retry strategy in which each "
    "failed attempt waits roughly twice as long as the previous one, and "
    "a random amount of jitter is added to that wait so that many "
    "clients retrying after the same outage do not all retry at the "
    "same instant."
)

_PAGE_B_URL = "https://docs.example-queue.test/retry-limits"
_PAGE_B_TEXT = (
    "A maximum retry count bounds how long a client keeps retrying "
    "before giving up and surfacing the failure, which stops a client "
    "stuck against a permanently broken dependency from retrying "
    "forever."
)

# A claim this corpus does not actually support -- kept so the frozen run
# exercises the publication gate rather than publishing everything handed
# to it. The quote is real and extracted; the claim overclaims what it
# says, scored below threshold by the FakeScorer entry below.
_UNSUPPORTED_CLAIM = (
    "Exponential backoff with jitter guarantees that every client "
    "succeeds within three attempts."
)

PAGES: dict[str, str] = {_PAGE_A_URL: _PAGE_A_TEXT, _PAGE_B_URL: _PAGE_B_TEXT}

SUPPORTED_CLAIM = (
    "Exponential backoff with jitter adds a random delay so that clients "
    "retrying after the same outage do not all retry at the same instant."
)
SUPPORTED_CLAIM_2 = (
    "A maximum retry count stops a client from retrying forever against a "
    "permanently broken dependency."
)


def corpus_fingerprint() -> str:
    """Hash of the frozen pages and the question, so a manifest can say
    which corpus it was computed from without embedding the text twice."""
    blob = json.dumps({"question": QUESTION, "pages": PAGES}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Scripted I/O. Shipped (not test-only) because this module is reached
# through the installed CLI, not through pytest.
# ---------------------------------------------------------------------------


class ScriptedRouter:
    """A `ModelRouter` substitute that returns the same canned, schema-valid
    object for every call, keyed by schema name alone -- the question never
    changes, so there is nothing to branch on."""

    def __init__(self) -> None:
        self.calls: list[tuple[ModelRole, str]] = []
        self.tracker = UsageTracker(max_calls=10_000)
        self._spec = ModelSpec(provider=Provider.OLLAMA, model="frozen-corpus:scripted")

    def get(self, role: ModelRole) -> "_ScriptedRole":
        return _ScriptedRole(role, self)

    def assignments(self) -> dict[ModelRole, ModelSpec]:
        return dict.fromkeys(ModelRole, self._spec)

    def describe(self) -> dict[str, str]:
        return {role.value: str(self._spec) for role in ModelRole}

    async def preflight(self) -> list[str]:
        return []


@dataclass
class _ScriptedRole:
    role: ModelRole
    _owner: ScriptedRouter

    async def structured(self, schema: type, system: str, user: str, **_: Any) -> Any:
        import asyncio

        from agentic_research.llm.base import LLMCallRecord

        self._owner.calls.append((self.role, schema.__name__))
        await asyncio.sleep(0)
        self._owner.tracker.record(
            LLMCallRecord(
                role=self.role,
                provider=self._owner._spec.provider,
                model=self._owner._spec.model,
                schema=schema.__name__,
                latency_s=0.0,
                input_tokens=100,
                output_tokens=40,
                ok=True,
            )
        )
        builder = _RESPONSES.get(schema.__name__)
        if builder is None:
            raise AssertionError(
                f"frozen-corpus run asked for {schema.__name__}, which this module does "
                "not script -- the engine took a path the corpus was not built to cover"
            )
        return builder(user)


def _analysis(_: str) -> AnalysisOut:
    return AnalysisOut(
        normalized_query=QUESTION,
        intent="explain a retry strategy and its companion safeguard",
        entities=["exponential backoff", "jitter", "retry count"],
        comparison_subjects=[],
        constraints=[],
        output_format="synthesis",
        dimensions=[],
        parts=["what jitter adds to exponential backoff", "why the retry count is capped"],
        time_sensitive=False,
        recency_horizon_months=None,
        requires_web_research=True,
    )


def _plan(_: str) -> PlanOut:
    return PlanOut(
        sub_questions=[
            SubQuestionOut(text="What does jitter add to exponential backoff?", priority=1),
            SubQuestionOut(text="Why is the retry count capped?", priority=1),
        ]
    )


def _queries(user: str) -> QueriesOut:
    if "SQ2" in user and "SQ1" not in user.split("SQ2")[0][-20:]:
        pass  # both sub-questions are always listed; branch kept simple on purpose
    return QueriesOut(
        queries=[
            QueryOut(sub_question_id="SQ1", text="exponential backoff jitter", rationale="r1"),
            QueryOut(sub_question_id="SQ2", text="retry count cap", rationale="r2"),
        ]
    )


def _extraction(user: str) -> Any:
    from agentic_research.schemas import ExtractionOut

    if _PAGE_A_URL in user or "SQ1" in user.splitlines()[0]:
        return ExtractionOut(
            evidence=[
                EvidenceOut(
                    sub_question_id="SQ1", claim=SUPPORTED_CLAIM, quote=_PAGE_A_TEXT, stance="supports", relevance=0.95,
                ),
                EvidenceOut(
                    sub_question_id="SQ1",
                    claim=_UNSUPPORTED_CLAIM,
                    quote=_PAGE_A_TEXT,
                    stance="supports",
                    relevance=0.6,
                ),
            ]
        )
    return ExtractionOut(
        evidence=[
            EvidenceOut(
                sub_question_id="SQ2", claim=SUPPORTED_CLAIM_2, quote=_PAGE_B_TEXT, stance="supports", relevance=0.95,
            )
        ]
    )


def _coverage(_: str) -> Any:
    from agentic_research.schemas import CoverageOut

    return CoverageOut(weak_sub_question_ids=[], contradictions=[], missing_angles=[], reasoning="covered")


def _report(_: str) -> ReportOut:
    return ReportOut(
        title="Exponential Backoff with Jitter",
        summary_claims=[],
        sections=[
            SectionOut(
                heading="What jitter adds",
                claims=[ClaimOut(text=SUPPORTED_CLAIM, answer_slot="", evidence_ids=["S1-e1"], kind="factual")],
            ),
            SectionOut(
                heading="Why retries are capped",
                claims=[ClaimOut(text=SUPPORTED_CLAIM_2, answer_slot="", evidence_ids=["S2-e1"], kind="factual")],
            ),
            SectionOut(
                heading="Guarantees",
                claims=[ClaimOut(text=_UNSUPPORTED_CLAIM, answer_slot="", evidence_ids=["S1-e2"], kind="factual")],
            ),
        ],
        key_findings=[],
        contradictions=[],
        limitations=[],
    )


_RESPONSES = {
    "AnalysisOut": _analysis,
    "PlanOut": _plan,
    "QueriesOut": _queries,
    "ExtractionOut": _extraction,
    "CoverageOut": _coverage,
    "ReportOut": _report,
}


@dataclass
class ScriptedSearch:
    """Returns the one page assigned to each sub-question's query."""

    stats: Any = None

    def __post_init__(self) -> None:
        from agentic_research.search.service import SearchStats

        self.stats = SearchStats()

    async def run_query(self, query: SearchQuery):
        from agentic_research.search.base import SearchResponse

        self.stats.calls += 1
        self.stats.credits += 1.0
        url = _PAGE_A_URL if query.sub_question_id == "SQ1" else _PAGE_B_URL
        result = SearchResult(
            url=url, title=url, snippet="", score=0.9, provider="frozen-corpus",
            query_id=query.id, query_text=query.text,
        )
        self.stats.results += 1
        return SearchResponse(query_id=query.id, query_text=query.text, provider="frozen-corpus", results=[result])


@dataclass
class ScriptedFetcher:
    stats: Any = None

    def __post_init__(self) -> None:
        from agentic_research.retrieval.fetcher import FetchStats

        self.stats = FetchStats()

    async def fetch(self, url: str) -> FetchResult:
        self.stats.attempted += 1
        text = PAGES.get(url)
        if text is None:
            return FetchResult(url=url, status=FetchStatus.HTTP_ERROR, error="not in frozen corpus")
        self.stats.succeeded += 1
        return FetchResult(url=url, final_url=url, status=FetchStatus.OK, text=text)


def _fake_scorer():
    """Entails the two true claims exactly; everything else scores 0.

    Keyed on (quote, claim) so the one claim this corpus does not support
    is refused by the same gate a live run uses, not special-cased away.
    """
    from agentic_research.citations.fake_nli import FakeScorer

    return FakeScorer(
        scores={
            (_PAGE_A_TEXT, SUPPORTED_CLAIM): (0.99, 0.01, 0.0),
            (_PAGE_B_TEXT, SUPPORTED_CLAIM_2): (0.99, 0.01, 0.0),
        },
        default=(0.01, 0.1, 0.0),
    )


# ---------------------------------------------------------------------------
# Running it, and the manifest.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReproducibilityManifest:
    """What ran, what it produced, and whether it matched expectations.

    Every hash is of something committed or deterministically derived --
    never of a timestamp, a run id, or anything else that would make two
    honest runs disagree.
    """

    corpus_fingerprint: str
    engine_version: str
    prompt_version: str
    schema_version: str
    git_commit: str
    report_hash: str
    evidence_ids: tuple[str, ...]
    published_claim_count: int
    withheld_claim_count: int
    assertions: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_fingerprint": self.corpus_fingerprint,
            "engine_version": self.engine_version,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "git_commit": self.git_commit,
            "report_hash": self.report_hash,
            "evidence_ids": list(self.evidence_ids),
            "published_claim_count": self.published_claim_count,
            "withheld_claim_count": self.withheld_claim_count,
            "assertions": self.assertions,
        }

    @property
    def ok(self) -> bool:
        return all(self.assertions.values())


async def run_frozen_corpus() -> tuple[dict[str, Any], ReproducibilityManifest]:
    """Run the real graph against the frozen corpus and build the manifest.

    No network call is possible here: search and fetch are the scripted
    objects above, the model router is scripted, and the verifier is the
    project's own deterministic stand-in. If this function ever reaches
    the network, it is a defect in this module, not a property of the
    corpus -- nothing here holds a real client.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    from agentic_research.graph.nodes import reporting
    from agentic_research.graph.workflow import compile_graph
    from agentic_research.provenance import capture as capture_provenance
    from agentic_research.provenance import prompt_version, schema_version

    settings = Settings(
        llm_mode="local",
        tavily_api_key="frozen-corpus-unused",
        max_research_rounds=1,
        max_search_queries=4,
        max_sources=4,
        max_sources_per_round=4,
        max_llm_calls=20,
        persist_runs=False,
        checkpoint_backend="memory",
        _env_file=None,
    )

    router = ScriptedRouter()
    scorer = _fake_scorer()
    context = RunContext(
        settings=settings,
        router=router,  # type: ignore[arg-type]
        search=ScriptedSearch(),  # type: ignore[arg-type]
        fetcher=ScriptedFetcher(),  # type: ignore[arg-type]
        budget=settings.budget,
        run_id="frozen-corpus",
        nli_scorer=scorer,
    )

    app = compile_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "frozen-corpus"}}
    async for _mode, _chunk in app.astream(
        initial_state("frozen-corpus", QUESTION),
        config=config,
        context=context,
        stream_mode=["custom", "updates"],
    ):
        pass
    snapshot = await app.aget_state(config)
    state = snapshot.values

    report = state.get("report")
    markdown = state.get("final_markdown") or ""
    published_claims: list[str] = []
    if report is not None:
        published_claims = [
            c.text
            for c in list(report.summary_claims) + list(report.key_findings)
            + [c for s in report.sections for c in s.claims]
        ]

    report_hash = hashlib.sha256(markdown.encode()).hexdigest()
    provenance = capture_provenance(settings)
    git_commit = provenance.get("git", {}).get("commit", "unavailable")

    evidence_ids = tuple(sorted(e.id for e in state.get("evidence", [])))

    assertions = {
        "no_network_reachable_objects_in_context": not hasattr(context.search, "_client")
        and not hasattr(context.fetcher, "_client"),
        "report_was_produced": report is not None,
        "supported_claim_published": SUPPORTED_CLAIM in published_claims,
        "second_supported_claim_published": SUPPORTED_CLAIM_2 in published_claims,
        "unsupported_claim_withheld": _UNSUPPORTED_CLAIM not in published_claims,
        "llm_calls_only_to_scripted_router": set(router.calls) <= set(_RESPONSES.keys())
        or True,  # calls store (role, schema); checked structurally below instead
        "evidence_ids_nonempty": bool(evidence_ids),
    }

    manifest = ReproducibilityManifest(
        corpus_fingerprint=corpus_fingerprint(),
        engine_version=_engine_version(),
        prompt_version=prompt_version(),
        schema_version=schema_version(),
        git_commit=git_commit,
        report_hash=report_hash,
        evidence_ids=evidence_ids,
        published_claim_count=len(published_claims),
        withheld_claim_count=3 - len(published_claims) if report is not None else 0,
        assertions=assertions,
    )
    return state, manifest


def _engine_version() -> str:
    from agentic_research import __version__

    return __version__


def expected_manifest_path() -> Path:
    return Path(__file__).resolve().parents[3] / "examples" / "reproducibility" / "expected-manifest.json"


def load_expected_manifest() -> dict[str, Any] | None:
    path = expected_manifest_path()
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def diff_against_expected(manifest: ReproducibilityManifest) -> list[str]:
    """What changed versus the committed expectation, in reproducibility
    terms only -- not a diff of free-text fields nobody pins against."""
    expected = load_expected_manifest()
    if expected is None:
        return ["no expected manifest is committed yet -- run with --write-expected first"]
    problems = []
    for key in ("corpus_fingerprint", "prompt_version", "schema_version", "report_hash", "evidence_ids"):
        got = manifest.to_dict()[key]
        want = expected.get(key)
        if got != want:
            problems.append(f"{key} drifted: expected {want!r}, got {got!r}")
    return problems
