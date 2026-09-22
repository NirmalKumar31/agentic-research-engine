"""Search, retrieval and evidence extraction.

This module holds the parallel stages. Three of its functions are fan-out
workers, dispatched with ``Send`` and running concurrently; the others are
barriers that run once all workers in a stage have settled.

One rule governs every worker here: **it must not raise.** LangGraph's
node-level ``error_handler`` does not fire for ``Send``-dispatched tasks
(verified against 1.2.x), so an exception escaping a worker aborts the entire
super-step and loses its siblings' completed work. Workers therefore catch
their own failures and record them into the ``errors`` channel, and the run
continues with whatever succeeded.
"""

from __future__ import annotations

from langgraph.types import Overwrite

from agentic_research.config import ModelRole
from agentic_research.evidence.dedup import (
    dedupe_by_content,
    dedupe_search_results,
)
from agentic_research.evidence.quality import classify_source, score_source
from agentic_research.evidence.store import verify_quote
from agentic_research.graph.nodes.common import ctx, emit, error_from, stage
from agentic_research.graph.prompts import EXTRACTOR_SYSTEM, extractor_user
from agentic_research.graph.state import (
    ExtractTask,
    FetchTask,
    ResearchState,
    SearchTask,
)
from agentic_research.llm.base import BudgetExceededError, LLMError
from agentic_research.models import (
    EvidenceItem,
    FetchStatus,
    SourceDocument,
    Stance,
)
from agentic_research.observability import get_logger
from agentic_research.retrieval.parser import markdown_to_text, truncate, word_count
from agentic_research.schemas import ExtractionOut

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — parallel search
# ---------------------------------------------------------------------------


async def search_worker(state: SearchTask) -> ResearchState:
    """Run one search query. One instance per query, all concurrent.

    The parameter is named ``state`` rather than ``payload`` because
    LangGraph's node protocol declares it that way and structural matching is
    name-sensitive. What arrives here is the Send payload, not graph state.

    Concurrency is bounded inside ``SearchService`` by a semaphore rather than
    by limiting the fan-out, so the graph can dispatch every query at once and
    still hold the provider to a fixed number of simultaneous requests.
    """
    query = state["query"]
    context = ctx()

    response = await context.search.run_query(query)
    if not response.ok:
        emit("search_failed", query_id=query.id, query=query.text)
        return {
            "round_results": [],
            "errors": [
                error_from(
                    "search",
                    RuntimeError(response.error or "search failed"),
                    f"{query.id}: {query.text}",
                )
            ],
            "counters": {"searches_failed": 1},
        }

    emit(
        "search_completed",
        query_id=query.id,
        query=query.text,
        results=len(response.results),
    )
    return {
        "round_results": response.results,
        "counters": {"searches_ok": 1, "search_results": len(response.results)},
    }


# ---------------------------------------------------------------------------
# Barrier — deduplicate before spending any fetches
# ---------------------------------------------------------------------------


async def dedupe_sources(state: ResearchState) -> ResearchState:
    """Collapse this round's results into unique fetch candidates.

    This barrier is the reason the pipeline fans out per stage rather than
    running search-fetch-extract inside one worker. A worker cannot see its
    siblings' results, so a page found by four sub-questions would be fetched
    and LLM-processed four times. Deduplicating here makes it cost one of each.

    Source ids are also assigned here: this node is single-writer, so ids are
    deterministic and free of races.
    """
    results = state.get("round_results", [])
    existing_sources = state.get("sources", [])
    completed = state.get("completed_queries", [])
    context = ctx()

    with stage("dedupe_sources") as timing:
        query_to_sub_question = {q.id: q.sub_question_id for q in completed}
        known = {s.canonical_url for s in existing_sources}
        candidates, dedup_stats = dedupe_search_results(
            results, query_to_sub_question, known_canonical_urls=known
        )

        # Best-first, so the per-round and total source caps keep the most
        # promising pages rather than an arbitrary prefix.
        candidates.sort(key=lambda c: c.best_score or 0.0, reverse=True)

        room_total = max(0, context.budget.max_sources - len(existing_sources))
        limit = min(context.budget.max_sources_per_round, room_total)
        selected = candidates[:limit]
        dropped = len(candidates) - len(selected)

        next_index = len(existing_sources) + 1
        stubs: list[SourceDocument] = []
        for offset, candidate in enumerate(selected):
            source_type = classify_source(candidate.url, candidate.domain)
            stubs.append(
                SourceDocument(
                    id=f"S{next_index + offset}",
                    url=candidate.url,
                    canonical_url=candidate.canonical_url,
                    title=candidate.title or candidate.url,
                    domain=candidate.domain,
                    source_type=source_type,
                    published_date=candidate.published_date,  # type: ignore[arg-type]
                    search_score=candidate.best_score,
                    found_by_queries=list(candidate.found_by_queries),
                    answers_sub_questions=list(candidate.sub_question_ids),
                )
            )
        timing["candidates"] = len(candidates)
        timing["selected"] = len(selected)

    log.info(
        "sources_deduplicated",
        results_in=dedup_stats.results_in,
        unique=dedup_stats.unique_out,
        fetches_avoided=dedup_stats.fetches_avoided,
        selected=len(selected),
        dropped_over_budget=dropped,
    )
    emit(
        "sources_deduplicated",
        results=dedup_stats.results_in,
        unique=dedup_stats.unique_out,
        avoided=dedup_stats.fetches_avoided,
        selected=len(selected),
    )
    return {
        "sources": stubs,
        "fetch_candidates": selected,
        # Clear the fan-in channel so the next round starts empty. Without
        # Overwrite the operator.add reducer would keep every round's results
        # and re-deduplicate them on every pass.
        "round_results": Overwrite(value=[]),  # type: ignore[typeddict-item]
        "stage_timings": [timing],
        "counters": {
            "duplicate_urls": dedup_stats.duplicate_urls,
            "already_known": dedup_stats.already_known,
            "fetches_avoided": dedup_stats.fetches_avoided,
            "dropped_over_budget": dropped,
        },
    }


# ---------------------------------------------------------------------------
# Stage 2 — parallel fetch
# ---------------------------------------------------------------------------


async def fetch_worker(state: FetchTask) -> ResearchState:
    """Retrieve one source's text. One instance per unique URL.

    Skips the HTTP request entirely when the search provider already returned
    the page body. Tavily can return page content with the search result, and
    re-downloading a page we have been handed is a waste of time and of the
    origin server's bandwidth.
    """
    source = state["source"]
    candidate = state["candidate"]
    settings = ctx().settings

    if candidate.provider_content and len(candidate.provider_content.strip()) > 200:
        text = truncate(markdown_to_text(candidate.provider_content), settings.max_extract_chars)
        if text.strip():
            return _finalise_source(source, text, FetchStatus.PROVIDER_CONTENT, reused=True)

    result = await ctx().fetcher.fetch(source.url)
    if not result.ok:
        emit("source_failed", source_id=source.id, status=result.status.value)
        return {
            "sources": [
                source.model_copy(
                    update={"fetch_status": result.status, "fetch_error": result.error}
                )
            ],
            "counters": {"fetch_failed": 1},
        }

    text = truncate(result.text, settings.max_extract_chars)
    emit("source_retrieved", source_id=source.id, title=source.title[:70])
    return _finalise_source(source, text, FetchStatus.OK, reused=False)


def _finalise_source(
    source: SourceDocument, text: str, status: FetchStatus, *, reused: bool
) -> ResearchState:
    words = word_count(text)
    return {
        "sources": [
            source.model_copy(
                update={
                    "text": text,
                    "fetch_status": status,
                    "content_hash": SourceDocument.hash_text(text),
                    "word_count": words,
                }
            )
        ],
        "counters": {
            "fetched_ok": 1,
            "provider_content_reused": 1 if reused else 0,
            "words_retrieved": words,
        },
    }


# ---------------------------------------------------------------------------
# Barrier — score sources and choose what is worth an extraction call
# ---------------------------------------------------------------------------


async def register_sources(state: ResearchState) -> ResearchState:
    """Content-deduplicate, quality-score, and select sources for extraction.

    Extraction is one LLM call per source, so this is where the largest share
    of a run's token spend is committed. Selecting by quality rather than by
    arrival order is the difference between reading the standards document and
    reading whatever happened to be fetched first.
    """
    sources = state.get("sources", [])
    analysis = state.get("analysis")
    horizon = analysis.recency_horizon_months if analysis else None

    with stage("register_sources") as timing:
        deduped, content_duplicates = dedupe_by_content(sources)

        scored: list[SourceDocument] = []
        for source in deduped:
            if not source.is_usable or source.duplicate_of:
                scored.append(source)
                continue
            score, reasons = score_source(
                url=source.url,
                domain=source.domain,
                source_type=source.source_type,
                search_score=source.search_score,
                word_count=source.word_count,
                published=source.published_date,
                recency_horizon_months=horizon,
            )
            scored.append(
                source.model_copy(update={"quality_score": score, "quality_reasons": reasons})
            )

        # Only sources with usable text and no prior evidence need a call.
        already_extracted = {e.source_id for e in state.get("evidence", [])}
        ready = [
            s
            for s in scored
            if s.is_usable and not s.duplicate_of and s.id not in already_extracted
        ]
        ready.sort(key=lambda s: s.quality_score, reverse=True)
        pending = [s.id for s in ready]
        timing["extract_candidates"] = len(pending)

    usable = sum(1 for s in scored if s.is_usable and not s.duplicate_of)
    log.info(
        "sources_registered",
        total=len(scored),
        usable=usable,
        content_duplicates=content_duplicates,
        queued_for_extraction=len(pending),
    )
    emit(
        "sources_registered", usable=usable, duplicates=content_duplicates, extracting=len(pending)
    )
    return {
        "sources": scored,
        "pending_extraction": pending,
        "stage_timings": [timing],
        "counters": {"content_duplicates": content_duplicates},
    }


# ---------------------------------------------------------------------------
# Stage 3 — parallel evidence extraction
# ---------------------------------------------------------------------------


async def extract_worker(state: ExtractTask) -> ResearchState:
    """Extract evidence from one source against all open sub-questions.

    One call per source, not one per (source, sub-question) pair: the model
    reads the page once and tags each finding with the question it answers.
    That bounds extraction calls at the number of sources instead of their
    product with the number of sub-questions.

    Every quote is checked against the source text here, inside the worker,
    while the text is in hand. An unverifiable quote is kept but flagged, so
    the failure is visible in the report rather than silently dropped.
    """
    source = state["source"]
    sub_questions = state["sub_questions"]

    block = "\n".join(f"{q.id}: {q.text}" for q in sub_questions)
    valid_ids = {q.id for q in sub_questions}

    try:
        out = (
            await ctx()
            .router.get(ModelRole.RESEARCHER)
            .structured(
                ExtractionOut,
                EXTRACTOR_SYSTEM,
                extractor_user(block, source.title, source.text),
            )
        )
    except BudgetExceededError as exc:
        # Expected at the ceiling, not an anomaly: stop extracting quietly.
        log.info("extraction_skipped_budget", source_id=source.id)
        return {
            "evidence": [],
            "errors": [error_from("extract", exc, source.id)],
            "counters": {"extraction_skipped": 1},
        }
    except LLMError as exc:
        log.warning("extraction_failed", source_id=source.id, error=str(exc)[:200])
        return {
            "evidence": [],
            "errors": [error_from("extract", exc, f"{source.id} {source.url}")],
            "counters": {"extraction_failed": 1},
        }

    items: list[EvidenceItem] = []
    unverified = 0
    for index, finding in enumerate(out.evidence, start=1):
        if not finding.claim.strip() or not finding.quote.strip():
            continue
        # A model that tags a finding with an unknown sub-question id would
        # corrupt provenance; attribute it to the first open question instead.
        sub_question_id = (
            finding.sub_question_id if finding.sub_question_id in valid_ids else sub_questions[0].id
        )
        verified = verify_quote(finding.quote, source.text)
        if not verified:
            unverified += 1
        items.append(
            EvidenceItem(
                # Unique without coordination: exactly one worker owns S<n>.
                id=f"{source.id}-e{index}",
                source_id=source.id,
                sub_question_id=sub_question_id,
                query_id=source.found_by_queries[0] if source.found_by_queries else "",
                claim=finding.claim.strip(),
                quote=finding.quote.strip(),
                stance=Stance(finding.stance),
                relevance=min(max(finding.relevance, 0.0), 1.0),
                quote_verified=verified,
            )
        )

    log.debug(
        "evidence_extracted",
        source_id=source.id,
        items=len(items),
        unverified_quotes=unverified,
    )
    emit("evidence_extracted", source_id=source.id, items=len(items))
    return {
        "evidence": items,
        "counters": {
            "evidence_items": len(items),
            "unverified_quotes": unverified,
            "sources_extracted": 1,
        },
    }
