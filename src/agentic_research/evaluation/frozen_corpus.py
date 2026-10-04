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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from agentic_research.citations.fake_nli import FakeScorer
from agentic_research.config import LLMMode, ModelRole, ModelSpec, Provider, Settings
from agentic_research.graph.state import RunContext, initial_state
from agentic_research.llm.base import UsageTracker
from agentic_research.models import FetchStatus, SearchQuery, SearchResult
from agentic_research.retrieval.fetcher import FetchResult
from agentic_research.schemas import (
    AnalysisOut,
    ClaimOut,
    EvidenceOut,
    PlanOut,
    QueriesOut,
    QueryOut,
    RelevanceOut,
    RelevanceVerdictOut,
    ReportOut,
    SectionOut,
    SubQuestionOut,
)
from agentic_research.search.base import SearchResponse

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
    "Exponential backoff with jitter guarantees that every client succeeds within three attempts."
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

    def get(self, role: ModelRole) -> _ScriptedRole:
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
                    sub_question_id="SQ1",
                    claim=SUPPORTED_CLAIM,
                    quote=_PAGE_A_TEXT,
                    stance="supports",
                    relevance=0.95,
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
                sub_question_id="SQ2",
                claim=SUPPORTED_CLAIM_2,
                quote=_PAGE_B_TEXT,
                stance="supports",
                relevance=0.95,
            )
        ]
    )


def _coverage(_: str) -> Any:
    from agentic_research.schemas import CoverageOut

    return CoverageOut(
        weak_sub_question_ids=[], contradictions=[], missing_angles=[], reasoning="covered"
    )


def _report(_: str) -> ReportOut:
    return ReportOut(
        title="Exponential Backoff with Jitter",
        summary_claims=[
            ClaimOut(
                text=SUPPORTED_CLAIM, answer_slot="part_1", evidence_ids=["S1-e1"], kind="factual"
            )
        ],
        sections=[
            SectionOut(
                heading="Why retries are capped",
                claims=[
                    ClaimOut(
                        text=SUPPORTED_CLAIM_2,
                        answer_slot="part_2",
                        evidence_ids=["S2-e1"],
                        kind="factual",
                    )
                ],
            ),
            SectionOut(
                heading="Guarantees",
                claims=[
                    ClaimOut(
                        text=_UNSUPPORTED_CLAIM,
                        answer_slot="part_1",
                        evidence_ids=["S1-e2"],
                        kind="factual",
                    )
                ],
            ),
        ],
        key_findings=[],
        contradictions=[],
        limitations=[],
    )


def _relevance(user: str) -> RelevanceOut:
    """Judge every candidate claim relevant to its own sub-question.

    The two true claims in this corpus genuinely answer what they are
    filed under; marking them relevant is not a thumb on the scale, it
    is what a correct judge would say. The overclaiming third claim is
    withheld on entailment regardless of this verdict -- relevance is a
    separate gate from support, and this corpus does not need relevance
    to do the refusing.
    """
    indices = [int(m) for m in re.findall(r"^(\d+)\. ", user, flags=re.MULTILINE)]
    return RelevanceOut(
        verdicts=[
            RelevanceVerdictOut(
                claim_index=i, answers_question=True, reason="answers its sub-question"
            )
            for i in indices
        ]
    )


_RESPONSES = {
    "AnalysisOut": _analysis,
    "PlanOut": _plan,
    "QueriesOut": _queries,
    "ExtractionOut": _extraction,
    "CoverageOut": _coverage,
    "ReportOut": _report,
    "RelevanceOut": _relevance,
}


@dataclass
class ScriptedSearch:
    """Returns the one page assigned to each sub-question's query."""

    stats: Any = None

    def __post_init__(self) -> None:
        from agentic_research.search.service import SearchStats

        self.stats = SearchStats()

    async def run_query(self, query: SearchQuery) -> SearchResponse:
        self.stats.calls += 1
        self.stats.credits += 1.0
        url = _PAGE_A_URL if query.sub_question_id == "SQ1" else _PAGE_B_URL
        result = SearchResult(
            url=url,
            title=url,
            snippet="",
            score=0.9,
            provider="frozen-corpus",
            query_id=query.id,
            query_text=query.text,
        )
        self.stats.results += 1
        return SearchResponse(
            query_id=query.id, query_text=query.text, provider="frozen-corpus", results=[result]
        )


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


def _fake_scorer() -> FakeScorer:
    """Entails the two true claims exactly; everything else scores 0.

    Keyed on (quote, claim) so the one claim this corpus does not support
    is refused by the same gate a live run uses, not special-cased away.
    """
    return FakeScorer(
        scores={
            (_PAGE_A_TEXT, SUPPORTED_CLAIM): (0.99, 0.01, 0.0),
            (_PAGE_B_TEXT, SUPPORTED_CLAIM_2): (0.99, 0.01, 0.0),
        },
        # A valid probability distribution that stays below threshold --
        # NLIScores rejects a triple that does not sum to ~1.0, which the
        # first version of this default did not and failed as "malformed
        # scores" rather than as a withheld claim.
        default=(0.01, 0.95, 0.04),
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
    from pydantic import SecretStr

    from agentic_research.graph.workflow import compile_graph
    from agentic_research.provenance import capture as capture_provenance
    from agentic_research.provenance import prompt_version, schema_version

    settings = Settings(
        llm_mode=LLMMode.LOCAL,
        tavily_api_key=SecretStr("frozen-corpus-unused"),
        max_research_rounds=1,
        max_search_queries=4,
        max_sources=4,
        max_sources_per_round=4,
        max_llm_calls=20,
        persist_runs=False,
        checkpoint_backend="memory",
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
    config: RunnableConfig = {"configurable": {"thread_id": "frozen-corpus"}}
    async for _mode, _chunk in app.astream(
        initial_state("frozen-corpus", QUESTION),
        config=config,
        context=context,
        # Must be a list, not a tuple -- see runner.py's own comment on
        # the same call: LangGraph switches on isinstance(..., list) to
        # decide whether to yield (mode, chunk) pairs.
        stream_mode=["custom", "updates"],
    ):
        pass
    snapshot = await app.aget_state(config)
    state = snapshot.values

    report = state.get("report")
    published_claims: list[str] = []
    if report is not None:
        published_claims = [
            c.text
            for c in list(report.summary_claims)
            + list(report.key_findings)
            + [c for s in report.sections for c in s.claims]
        ]

    # Hashed from structured content, deliberately not from
    # `final_markdown`. The rendered report ends with "Generated by
    # Agentic Research Engine on <timestamp> UTC." -- correct behaviour
    # for a real report, and it means the rendered markdown can never be
    # byte-identical across two runs made in different seconds, however
    # deterministic the engine itself is. The first version of this
    # manifest hashed the markdown and failed its own reproducibility
    # check for exactly that reason, on its first real invocation.
    content = {
        "title": report.title if report is not None else None,
        "summary_claims": [c.text for c in report.summary_claims] if report is not None else [],
        "key_findings": [c.text for c in report.key_findings] if report is not None else [],
        "sections": [
            {"heading": sec.heading, "claims": [c.text for c in sec.claims]}
            for sec in (report.sections if report is not None else [])
        ],
        "limitations": list(report.limitations) if report is not None else [],
    }
    report_hash = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    provenance = capture_provenance(settings)
    git_commit = provenance.get("git", {}).get("commit", "unavailable")

    evidence_ids = tuple(sorted(e.id for e in state.get("evidence", [])))

    schemas_requested = {schema for _role, schema in router.calls}
    assertions = {
        # Proves the run used the scripted I/O objects, not merely that
        # it *could have* -- an isinstance check on what RunContext
        # actually holds, rather than probing for an attribute a real
        # client happens to have.
        "search_and_fetch_are_the_scripted_objects": isinstance(context.search, ScriptedSearch)
        and isinstance(context.fetcher, ScriptedFetcher),
        "report_was_produced": report is not None,
        "supported_claim_published": SUPPORTED_CLAIM in published_claims,
        "second_supported_claim_published": SUPPORTED_CLAIM_2 in published_claims,
        "unsupported_claim_withheld": _UNSUPPORTED_CLAIM not in published_claims,
        # The router was actually exercised (not bypassed by a cached
        # path) and only ever asked for schemas this module scripts --
        # guaranteed by `_ScriptedRole.structured` raising on anything
        # else, so this is a sanity check on that guarantee holding.
        "router_was_used_and_only_for_scripted_schemas": bool(router.calls)
        and schemas_requested <= set(_RESPONSES.keys()),
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
    """Where the committed expectation lives.

    Packaged as data under ``evaluation/data/``, not under the
    repository's top-level ``examples/`` -- a relative path climbed from
    ``__file__`` resolves inside a source checkout and nowhere inside an
    installed wheel, where this file still has to be found: the CI job
    that installs the wheel into a bare /tmp venv runs this command, and
    there is no ``examples/`` directory there at all. Resolved the same
    way ``recorded_runs`` already is.
    """
    from importlib.resources import files

    return Path(str(files("agentic_research.evaluation") / "data" / "expected-manifest.json"))


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
    for key in (
        "corpus_fingerprint",
        "prompt_version",
        "schema_version",
        "report_hash",
        "evidence_ids",
    ):
        got = manifest.to_dict()[key]
        want = expected.get(key)
        if got != want:
            problems.append(f"{key} drifted: expected {want!r}, got {got!r}")
    return problems
