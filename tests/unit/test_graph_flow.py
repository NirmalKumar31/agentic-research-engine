"""End-to-end graph behaviour with every external dependency faked.

These are the tests that matter most: they exercise the real state machine,
the real reducers, real deduplication and real citation verification, while
substituting only the three I/O boundaries.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from agentic_research.config import Settings
from agentic_research.graph.state import RunContext, initial_state
from agentic_research.graph.workflow import compile_graph
from fakes import FakeFetcher, FakeRouter, FakeSearchService


def make_context(settings: Settings, **overrides) -> RunContext:
    return RunContext(
        settings=settings,
        router=overrides.get("router") or FakeRouter(),
        search=overrides.get("search") or FakeSearchService(),
        fetcher=overrides.get("fetcher") or FakeFetcher(),
        budget=settings.budget,
        run_id="test-run",
    )


async def run_graph(settings: Settings, context: RunContext, query: str = "test question"):
    app = compile_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    events: list[dict] = []
    async for mode, chunk in app.astream(
        initial_state("test-run", query),
        config=config,
        context=context,
        stream_mode=["custom", "updates"],
    ):
        if mode == "custom":
            events.append(chunk)
    snapshot = await app.aget_state(config)
    return snapshot.values, events


class TestHappyPath:
    async def test_full_run_reaches_a_verified_report(self, settings: Settings) -> None:
        context = make_context(settings)
        state, _events = await run_graph(settings, context)

        assert state["report"] is not None
        assert state["final_markdown"].startswith("# ")
        assert state["plan"] is not None
        # 3 from the initial plan plus 1 follow-up added by round 2.
        assert len(state["sub_questions"]) == 4
        assert state["round_number"] == 2
        assert state["completed_queries"], "queries should have been issued"
        assert state["sources"], "sources should have been gathered"
        assert state["evidence"], "evidence should have been extracted"

        verification = state["verification"]
        assert verification is not None
        # Every evidence reference the synthesiser made resolved, so nothing
        # was dropped and no error-level issue was raised.
        assert verification["resolvable_evidence_refs"] == verification["total_evidence_refs"]
        assert verification["resolvable_citations"] == verification["total_citations"]
        assert not any(i["severity"] == "error" for i in verification["issues"])

    async def test_graph_terminates_and_does_not_loop_after_finalize(
        self, settings: Settings
    ) -> None:
        """The drawn graph shows a spurious finalize edge; execution must not
        follow it. Guards against the run continuing past the report."""
        context = make_context(settings)
        _state, events = await run_graph(settings, context)
        completed = [e for e in events if e.get("event") == "completed"]
        assert len(completed) == 1, "finalize must run exactly once"

    async def test_progress_events_are_emitted_in_order(self, settings: Settings) -> None:
        context = make_context(settings)
        _, events = await run_graph(settings, context)
        names = [e["event"] for e in events]
        for expected in (
            "analyzing_query",
            "plan_generated",
            "queries_generated",
            "search_completed",
            "sources_deduplicated",
            "sources_registered",
            "evidence_extracted",
            "coverage_evaluated",
            "synthesized",
            "citations_verified",
            "completed",
        ):
            assert expected in names, f"missing progress event: {expected}"
        assert names.index("plan_generated") < names.index("queries_generated")
        assert names.index("synthesized") < names.index("citations_verified")

    async def test_provenance_chain_is_intact(self, settings: Settings) -> None:
        """Every evidence item must trace back to a real source and a real
        sub-question. This is the property the whole design exists to hold."""
        context = make_context(settings)
        state, _ = await run_graph(settings, context)

        source_ids = {s.id for s in state["sources"]}
        sub_question_ids = {q.id for q in state["sub_questions"]}
        query_ids = {q.id for q in state["completed_queries"]}

        assert state["evidence"]
        for item in state["evidence"]:
            assert item.source_id in source_ids
            assert item.sub_question_id in sub_question_ids
            if item.discovery is not None:
                # A recorded discovery path must be real on both halves.
                assert item.discovery.query_id in query_ids
                assert item.discovery.sub_question_id == item.sub_question_id
                assert not item.cross_attributed
            else:
                # No path is represented honestly, not with a borrowed query.
                assert item.cross_attributed
                assert item.query_id == ""


class TestDeduplication:
    async def test_shared_urls_are_fetched_once(self, settings: Settings) -> None:
        """Three queries returning the same three URLs must produce three
        fetches, not nine. This is the central efficiency claim.

        Pinned to one round so the measurement is not confounded by the
        follow-up round issuing further queries."""
        settings.max_research_rounds = 1
        search = FakeSearchService(results_per_query=3, overlap=True)
        fetcher = FakeFetcher()
        context = make_context(settings, search=search, fetcher=fetcher)
        state, _ = await run_graph(settings, context)

        assert search.stats.calls == 3, "three queries expected"
        assert search.stats.results == 9, "nine raw results expected"
        assert fetcher.stats.attempted == 3, (
            f"expected 3 fetches for 3 unique URLs, got {fetcher.stats.attempted}"
        )
        assert state["counters"]["duplicate_urls"] == 6

    async def test_distinct_urls_are_all_fetched(self, settings: Settings) -> None:
        settings.max_research_rounds = 1
        settings.max_sources_per_round = 10  # above the 6 distinct URLs produced
        search = FakeSearchService(results_per_query=2, overlap=False)
        fetcher = FakeFetcher()
        context = make_context(settings, search=search, fetcher=fetcher)
        await run_graph(settings, context)
        assert fetcher.stats.attempted == 6

    async def test_per_round_source_cap_is_enforced(self, settings: Settings) -> None:
        """Budgets are enforced before the spend, not reported after it."""
        settings.max_research_rounds = 1
        settings.max_sources_per_round = 4
        search = FakeSearchService(results_per_query=3, overlap=False)  # 9 distinct URLs
        fetcher = FakeFetcher()
        context = make_context(settings, search=search, fetcher=fetcher)
        state, _ = await run_graph(settings, context)
        assert fetcher.stats.attempted == 4
        assert state["counters"]["dropped_over_budget"] == 5

    async def test_one_extraction_call_per_source(self, settings: Settings) -> None:
        """Extraction is per source, not per (source x sub-question)."""
        router = FakeRouter()
        context = make_context(settings, router=router)
        state, _ = await run_graph(settings, context)
        usable = [s for s in state["sources"] if s.is_usable]
        assert router.count("ExtractionOut") == len(usable)


class TestConcurrency:
    async def test_searches_run_concurrently(self, settings: Settings) -> None:
        search = FakeSearchService()
        context = make_context(settings, search=search)
        await run_graph(settings, context)
        assert search.concurrent_peak > 1, "searches ran serially"

    async def test_fetches_run_concurrently(self, settings: Settings) -> None:
        settings.max_research_rounds = 1
        search = FakeSearchService(results_per_query=4, overlap=False)
        fetcher = FakeFetcher()
        context = make_context(settings, search=search, fetcher=fetcher)
        await run_graph(settings, context)
        assert fetcher.concurrent_peak > 1, "fetches ran serially"


class TestPartialFailure:
    async def test_one_failed_search_does_not_end_the_round(self, settings: Settings) -> None:
        search = FakeSearchService(fail_queries={"Q1"})
        context = make_context(settings, search=search)
        state, _ = await run_graph(settings, context)

        assert state["report"] is not None, "run must survive a failed search"
        assert state["sources"], "the other queries' sources should survive"
        assert any(e.stage == "search" for e in state["errors"])

    async def test_one_failed_fetch_does_not_end_the_round(self, settings: Settings) -> None:
        fetcher = FakeFetcher(fail_urls={"https://shared.example.com/page-0"})
        context = make_context(settings, fetcher=fetcher)
        state, _ = await run_graph(settings, context)

        assert state["report"] is not None
        failed = [s for s in state["sources"] if not s.is_usable]
        assert len(failed) == 1
        assert any(s.is_usable for s in state["sources"])

    async def test_all_searches_failing_still_produces_a_report(self, settings: Settings) -> None:
        search = FakeSearchService(fail_queries={"Q1", "Q2", "Q3"})
        context = make_context(settings, search=search)
        state, _ = await run_graph(settings, context)
        assert state["report"] is not None
        assert state["final_markdown"]

    async def test_synthesis_failure_falls_back_to_evidence_listing(
        self, settings: Settings
    ) -> None:
        router = FakeRouter()
        router.fail_schemas = {"ReportOut"}
        context = make_context(settings, router=router)
        state, _ = await run_graph(settings, context)

        assert state["report"] is not None
        assert "Evidence summary" in state["report"].title
        assert state["report"].key_findings, "verified evidence should still be listed"

    async def test_planning_failure_degrades_to_single_dimension(self, settings: Settings) -> None:
        router = FakeRouter()
        router.fail_schemas = {"PlanOut"}
        settings.max_research_rounds = 1
        context = make_context(settings, router=router)
        state, _ = await run_graph(settings, context)
        assert len(state["sub_questions"]) == 1
        assert state["report"] is not None

    async def test_extraction_failure_is_recorded_not_fatal(self, settings: Settings) -> None:
        router = FakeRouter()
        router.fail_schemas = {"ExtractionOut"}
        context = make_context(settings, router=router)
        state, _ = await run_graph(settings, context)
        assert state["evidence"] == []
        assert state["report"] is not None
        assert any(e.stage == "extract" for e in state["errors"])


class TestLoopTermination:
    """The engine must stop. These are the tests that prove it."""

    async def test_never_exceeds_max_rounds_even_if_critic_always_wants_more(
        self, settings: Settings
    ) -> None:
        """Termination must not depend on a model choosing to be satisfied."""
        from agentic_research.schemas import CoverageOut, FollowupOut, FollowupsOut

        router = FakeRouter()
        router.responses["CoverageOut"] = lambda _: CoverageOut(
            weak_sub_question_ids=["SQ1", "SQ2", "SQ3"],
            contradictions=[],
            missing_angles=["always something missing", "and another thing"],
            reasoning="never satisfied",
        )
        counter = {"n": 0}

        def endless_followups(_: str) -> FollowupsOut:
            counter["n"] += 1
            return FollowupsOut(
                followups=[FollowupOut(text=f"follow-up {counter['n']}", gap="gap")]
            )

        router.responses["FollowupsOut"] = endless_followups
        settings.max_research_rounds = 3
        settings.max_search_queries = 100
        settings.max_sources = 100

        context = make_context(settings, router=router)
        state, _ = await run_graph(settings, context)

        assert state["round_number"] == 3
        assert state["report"] is not None
        assert len(state["coverage_history"]) == 3

    async def test_stops_early_when_coverage_is_sufficient(self, settings: Settings) -> None:
        settings.max_research_rounds = 5
        # Two distinct sources per sub-question satisfies the coverage rule.
        search = FakeSearchService(results_per_query=3, overlap=False)
        context = make_context(settings, search=search)
        state, _ = await run_graph(settings, context)

        assert state["round_number"] < 5, "should not have used every round"
        assert state["coverage"].sufficient

    async def test_query_budget_stops_the_loop(self, settings: Settings) -> None:
        settings.max_research_rounds = 10
        settings.max_search_queries = 4
        context = make_context(settings)
        state, _ = await run_graph(settings, context)

        assert len(state["completed_queries"]) <= 4
        assert state["report"] is not None

    async def test_source_budget_stops_the_loop(self, settings: Settings) -> None:
        settings.max_research_rounds = 10
        settings.max_sources = 3
        search = FakeSearchService(results_per_query=3, overlap=False)
        context = make_context(settings, search=search)
        state, _ = await run_graph(settings, context)

        assert len(state["sources"]) <= 3
        assert state["report"] is not None

    async def test_llm_call_budget_is_enforced(self, settings: Settings) -> None:
        """The ceiling holds even though extraction fans out in parallel."""
        from agentic_research.llm.base import UsageTracker
        from fakes import FakeRoleModel

        router = FakeRouter()
        router.tracker = UsageTracker(max_calls=6)
        unpatched = FakeRoleModel.structured

        async def counted(self, schema, system, user, **kw):
            # Route the fake through the real reservation path so the budget
            # is exercised rather than simulated.
            await self._owner.tracker.reserve()
            return await unpatched(self, schema, system, user, **kw)

        FakeRoleModel.structured = counted
        try:
            context = make_context(settings, router=router)
            state, _ = await run_graph(settings, context)
        finally:
            FakeRoleModel.structured = unpatched

        assert len(router.calls) <= 6
        # A run that hit the ceiling must still produce something.
        assert state["report"] is not None

    async def test_empty_search_round_routes_forward_rather_than_ending(
        self, settings: Settings
    ) -> None:
        """A dispatcher returning no work must fall through to the next stage.
        An empty Send list silently ends the graph instead."""
        search = FakeSearchService(results_per_query=0)
        context = make_context(settings, search=search)
        state, events = await run_graph(settings, context)

        assert state["report"] is not None, "graph must not end at an empty round"
        assert any(e["event"] == "completed" for e in events)


class TestWorkersNeverRaise:
    """LangGraph's node-level error_handler does not fire for Send-dispatched
    nodes, so an exception escaping a worker aborts every sibling in the same
    super-step. These tests pin the contract that workers swallow their own
    failures. A real Ollama read timeout escaped this way once."""

    async def test_search_worker_survives_an_unexpected_exception(self, settings: Settings) -> None:
        class ExplodingSearch(FakeSearchService):
            async def run_query(self, query):
                raise RuntimeError("transport blew up")

        context = make_context(settings, search=ExplodingSearch())
        state, _ = await run_graph(settings, context)
        assert state["report"] is not None
        assert any(e.stage == "search" for e in state["errors"])

    async def test_fetch_worker_survives_an_unexpected_exception(self, settings: Settings) -> None:
        class ExplodingFetcher(FakeFetcher):
            async def fetch(self, url: str):
                raise RuntimeError("socket exploded")

        context = make_context(settings, fetcher=ExplodingFetcher())
        state, _ = await run_graph(settings, context)
        assert state["report"] is not None
        assert any(e.stage == "fetch" for e in state["errors"])

    async def test_extract_worker_survives_a_non_llm_exception(self, settings: Settings) -> None:
        """Regression: an httpx.ReadTimeout is not an LLMError, so a worker
        catching only LLMError let it escape and kill the super-step."""
        import httpx

        from fakes import FakeRoleModel

        unpatched = FakeRoleModel.structured

        async def timeout_on_extraction(self, schema, system, user, **kw):
            if schema.__name__ == "ExtractionOut":
                raise httpx.ReadTimeout("local model timed out")
            return await unpatched(self, schema, system, user, **kw)

        FakeRoleModel.structured = timeout_on_extraction
        try:
            context = make_context(settings)
            state, _ = await run_graph(settings, context)
        finally:
            FakeRoleModel.structured = unpatched

        assert state["report"] is not None, "a model timeout must not end the run"
        assert any(e.stage == "extract" for e in state["errors"])


class TestGraphStructure:
    """The declared graph, and the diagram we publish of it."""

    def test_declared_edges_match_the_intended_topology(self) -> None:
        from agentic_research.graph.workflow import build_graph

        builder = build_graph()
        plain = set(builder.edges)
        assert ("verify_citations", "finalize") in plain
        assert ("finalize", "__end__") in plain
        assert ("search_worker", "dedupe_sources") in plain
        assert ("fetch_worker", "register_sources") in plain
        assert ("extract_worker", "assess_coverage") in plain
        # finalize is terminal: nothing may route out of it except END.
        assert [t for s, t in plain if s == "finalize"] == ["__end__"]

    def test_every_dispatcher_declares_a_non_send_fallback(self) -> None:
        """A conditional edge returning an empty Send list silently ends the
        graph, so each dispatcher must also be able to route to a real node."""
        from agentic_research.graph.workflow import build_graph

        fallbacks = {
            "generate_queries": "assess_coverage",
            "dedupe_sources": "register_sources",
            "register_sources": "assess_coverage",
        }
        branches = build_graph().branches
        for source, fallback in fallbacks.items():
            ends = next(iter(branches[source].values())).ends or {}
            assert fallback in ends.values(), f"{source} cannot fall through"

    def test_rendered_diagram_matches_the_executable_graph(self) -> None:
        """We render Mermaid ourselves because LangGraph 1.2.x's
        draw_mermaid() drops finalize->END and invents conditional edges that
        were never declared. This pins our renderer to the real structure."""
        from agentic_research.graph.workflow import build_graph, render_mermaid

        builder = build_graph()
        diagram = render_mermaid()

        for source, target in builder.edges:
            src = "START" if source == "__start__" else source
            tgt = "END" if target == "__end__" else target
            assert f"    {src} --> {tgt}" in diagram, f"missing {src}->{tgt}"

        for source, branches in builder.branches.items():
            for branch in branches.values():
                for target in (branch.ends or {}).values():
                    assert f"{source} -." in diagram
                    assert target in diagram

        assert "finalize --> END" in diagram
        assert "finalize -.-> register_sources" not in diagram
