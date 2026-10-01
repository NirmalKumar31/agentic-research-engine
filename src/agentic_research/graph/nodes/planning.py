"""Planning nodes: understand the question, decompose it, write queries.

All three run once per round at most and are single-writer, which is why they
are the only place identifiers get assigned.
"""

from __future__ import annotations

from agentic_research.answer_contract import (
    AnswerContract,
    QuestionType,
    build_contract,
    type_from_output_format,
    unusable,
)
from agentic_research.config import ModelRole
from agentic_research.graph.nodes.common import ctx, emit, error_from, numbered, stage
from agentic_research.graph.prompts import (
    ANALYST_SYSTEM,
    FOLLOWUP_SYSTEM,
    PLANNER_SYSTEM,
    QUERY_WRITER_SYSTEM,
    analyst_user,
    followup_user,
    planner_user,
    query_writer_user,
)
from agentic_research.graph.state import ResearchState
from agentic_research.models import (
    OutputFormat,
    QueryAnalysis,
    ResearchPlan,
    SearchQuery,
    SubQuestion,
)
from agentic_research.observability import get_logger
from agentic_research.question_form import (
    reconcile_comparison_subjects,
    shape_from_wording,
)
from agentic_research.schemas import AnalysisOut, FollowupsOut, PlanOut, QueriesOut

log = get_logger(__name__)


async def analyze_query(state: ResearchState) -> ResearchState:
    """Turn the raw question into a structured brief.

    Degrades rather than fails: if the model cannot produce a valid analysis,
    the run continues with a minimal one built from the raw question. Losing
    intent detection costs some quality; aborting costs the user the run.
    """
    query = state["original_query"]
    emit("analyzing_query", query=query)

    with stage("analyze_query") as timing:
        try:
            out = (
                await ctx()
                .router.get(ModelRole.PLANNER)
                .structured(AnalysisOut, ANALYST_SYSTEM, analyst_user(query))
            )
            analysis = QueryAnalysis(
                original_query=query,
                normalized_query=out.normalized_query or query,
                intent=out.intent,
                entities=out.entities[:12],
                comparison_subjects=out.comparison_subjects[:6],
                dimensions=out.dimensions[:8],
                parts=out.parts[:8],
                constraints=out.constraints[:8],
                output_format=OutputFormat(out.output_format),
                time_sensitive=out.time_sensitive,
                recency_horizon_months=out.recency_horizon_months if out.time_sensitive else None,
                requires_web_research=out.requires_web_research,
            )
            errors = []
        except Exception as exc:
            # Any failure, not only LLMError: the fallback below is the
            # whole point, and one that fires for a single exception
            # type is a promise the node does not keep.
            log.warning(
                "analysis_failed_using_fallback",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            analysis = QueryAnalysis(
                original_query=query, normalized_query=query, intent="unclassified"
            )
            errors = [error_from("analyze_query", exc, "fell back to raw question")]

    log.info(
        "query_analyzed",
        intent=analysis.intent[:80],
        format=analysis.output_format.value,
        time_sensitive=analysis.time_sensitive,
    )
    emit("query_analyzed", intent=analysis.intent, format=analysis.output_format.value)

    contract = contract_from_analysis(analysis)
    if contract.usable:
        log.info(
            "answer_contract",
            type=str(contract.question_type),
            slots=[s.name for s in contract.required_slots],
        )
        emit(
            "answer_contract",
            question_type=str(contract.question_type),
            required=[s.name for s in contract.core_slots],
        )
    else:
        # Not fatal. The run continues and publishes nothing on
        # relevance grounds, and the report says the question could not
        # be turned into checkable requirements -- which is more useful
        # than a confident answer to a question nobody pinned down.
        log.warning("answer_contract_unusable", reason=contract.unusable_reason)
        emit("answer_contract_unusable", reason=contract.unusable_reason)

    return {
        "analysis": analysis,
        "contract": contract,
        "stage_timings": [timing],
        "errors": errors,
    }


def contract_from_analysis(analysis: QueryAnalysis) -> AnswerContract:
    """Derive the answer contract from the structured analysis.

    The question type comes from the analysis stage's own
    ``output_format`` rather than being classified a second time: two
    classifications that can disagree are worse than one.
    """
    question_type = type_from_output_format(analysis.output_format.value)
    if question_type is None:
        return unusable(
            analysis.normalized_query,
            f"no answer shape is defined for {analysis.output_format.value!r}",
        )

    # The user's own wording outranks the model's reading of it, but
    # only where the wording is explicit. Two near-identical phrasings
    # of "the main causes of hallucination" were classified `list` and
    # `definition`, and "the context window size of GPT-4 Turbo" became
    # a definition -- a figure checked against a slot asking what the
    # subject is. The original query is used, not the normalisation:
    # the normalisation is itself model output and can smooth away the
    # form being read.
    wording = shape_from_wording(analysis.original_query or analysis.normalized_query)
    shape_source = "model"

    # A named multi-part decomposition is explicit information the
    # wording reader does not have. "What is RAG, and how much does it
    # cost to run?" contains an explicit numeric form, so the reader
    # returns `numeric` and cannot see the question has two halves.
    # Overriding a model that correctly identified both would make the
    # answer worse, which is the opposite of the point.
    named_parts = [part for part in analysis.parts if part.strip()]
    if question_type is QuestionType.SYNTHESIS and named_parts:
        wording = None
    if wording is not None and wording is not question_type:
        question_type, shape_source = wording, "corrected-from-wording"
    elif wording is not None:
        shape_source = "wording"

    # Sides read from the wording. An empty result means the wording
    # carried no explicit comparison, and the analyst's entities stand
    # -- refusing a run over a failed parse would be worse than the
    # defect being fixed.
    sides = reconcile_comparison_subjects(
        analysis.original_query or analysis.normalized_query,
        list(analysis.comparison_subjects) or list(analysis.entities),
    )
    # A comparison needs its subjects. The analysis names entities; the
    # contract refuses when there are fewer than two, rather than
    # accepting a contrast nothing could fill.
    # A multi-part question whose parts were not named cannot be turned
    # into per-part slots, and an unusable contract refuses *every*
    # claim. Falling back to the shape this question would have been
    # given before `synthesis` was reachable is strictly no worse than
    # that, and keeps a model that picks the label without filling the
    # field from silencing the whole run.
    if question_type is QuestionType.SYNTHESIS and not [p for p in analysis.parts if p.strip()]:
        question_type = QuestionType.DEFINITION

    return build_contract(
        analysis.normalized_query,
        question_type,
        entities=analysis.entities,
        comparison_subjects=sides,
        shape_source=shape_source,
        dimensions=analysis.dimensions,
        constraints=analysis.constraints,
        parts=analysis.parts,
    )


def _analysis_block(analysis: QueryAnalysis) -> str:
    lines = [f"Question: {analysis.normalized_query}", f"Intent: {analysis.intent}"]
    if analysis.entities:
        lines.append("Key entities: " + ", ".join(analysis.entities))
    if analysis.constraints:
        lines.append("Constraints: " + "; ".join(analysis.constraints))
    if analysis.time_sensitive:
        lines.append(
            f"Time sensitive: prefer sources from the last "
            f"{analysis.recency_horizon_months or 24} months"
        )
    lines.append(f"Expected answer shape: {analysis.output_format.value}")
    return "\n".join(lines)


async def plan_research(state: ResearchState) -> ResearchState:
    """Decompose the question into sub-questions.

    Falls back to treating the original question as a single dimension. That
    is a worse plan, but it still produces a researchable run.
    """
    analysis = state.get("analysis") or QueryAnalysis(
        original_query=state["original_query"],
        normalized_query=state["original_query"],
        intent="unclassified",
    )
    emit("planning")

    with stage("plan_research") as timing:
        errors = []
        try:
            out = (
                await ctx()
                .router.get(ModelRole.PLANNER)
                .structured(PlanOut, PLANNER_SYSTEM, planner_user(_analysis_block(analysis)))
            )
            # Capped by what the source budget can actually cover.
            #
            # Coverage needs two distinct sources per sub-question, so a
            # run reading six pages can cover at most three. A live run
            # planned six dimensions for "what types of vector index",
            # covered two, and reported "only limited evidence was
            # found" four times -- which reads as a retrieval failure
            # and was arithmetic.
            affordable = max(2, ctx().budget.max_sources // 2)
            raw = out.sub_questions[:affordable]
        except Exception as exc:
            log.warning(
                "planning_failed_using_single_dimension",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            raw = []
            errors = [error_from("plan_research", exc, "single-dimension fallback")]

        ids = numbered("SQ", 1, len(raw))
        sub_questions = [
            SubQuestion(
                id=sq_id,
                text=item.text,
                priority=min(max(item.priority, 1), 3),
                round_introduced=1,
            )
            for sq_id, item in zip(ids, raw, strict=True)
        ]
        if not sub_questions:
            sub_questions = [
                SubQuestion(
                    id="SQ1",
                    text=analysis.normalized_query,
                    rationale="fallback: the question itself",
                    priority=1,
                )
            ]

        plan = ResearchPlan(analysis=analysis, sub_questions=sub_questions)

    log.info("plan_generated", sub_questions=len(sub_questions))
    emit(
        "plan_generated",
        count=len(sub_questions),
        questions=[q.text for q in sub_questions],
    )
    return {
        "plan": plan,
        "sub_questions": sub_questions,
        "stage_timings": [timing],
        "errors": errors,
    }


def _sub_question_block(sub_questions: list[SubQuestion]) -> str:
    return "\n".join(f"{q.id}: {q.text}" for q in sub_questions)


def _median_word_count(texts: list[str]) -> float:
    """Query length, logged because it is the measurable symptom.

    The queries that failed averaged eleven words of stacked jargon.
    A number in the log makes the regression visible without having to
    read six query strings and judge them by eye.
    """
    if not texts:
        return 0.0
    counts = sorted(len(t.split()) for t in texts)
    middle = len(counts) // 2
    if len(counts) % 2:
        return float(counts[middle])
    return (counts[middle - 1] + counts[middle]) / 2


async def generate_queries(state: ResearchState) -> ResearchState:
    """Write this round's search queries.

    Enforces the query budget here rather than at dispatch, because this is
    where the count is known and can be trimmed before anything is spent.
    """
    context = ctx()
    round_number = state.get("round_number", 0) + 1
    completed = state.get("completed_queries", [])
    all_sub_questions = state.get("sub_questions", [])

    # In a follow-up round only the newly added sub-questions need queries;
    # the earlier ones have already been searched.
    targets = (
        [q for q in all_sub_questions if q.round_introduced == round_number]
        if round_number > 1
        else all_sub_questions
    )
    if not targets:
        targets = all_sub_questions

    remaining_budget = context.budget.max_search_queries - len(completed)
    if remaining_budget <= 0:
        log.warning("query_budget_exhausted", issued=len(completed))
        return {
            "pending_queries": [],
            "round_number": round_number,
            "stop_reason": "query budget exhausted",
        }

    emit("generating_queries", round=round_number, sub_questions=len(targets))
    with stage("generate_queries") as timing:
        errors = []
        try:
            contract = state.get("contract")
            analysis = state.get("analysis")
            out = await context.router.get(ModelRole.RESEARCHER).structured(
                QueriesOut,
                QUERY_WRITER_SYSTEM,
                query_writer_user(
                    _sub_question_block(targets),
                    [q.text for q in completed],
                    # The user's own wording, and the shape the answer
                    # must take. A causes question and a comparison do
                    # not want the same query style, and sending one
                    # style for every shape is how a broad explanatory
                    # question was searched as a literature review.
                    question=(
                        getattr(analysis, "original_query", "") or getattr(contract, "question", "")
                    ),
                    answer_shape=(
                        str(contract.question_type)
                        if contract is not None and contract.usable
                        else ""
                    ),
                ),
            )
            # The rationale travels with the query. Flattening to
            # `(id, text)` here is how six earlier values were computed
            # and then dropped before anything could read them --
            # `SearchQuery.rationale` has existed unpopulated all along.
            proposed = [(q.sub_question_id, q.text, q.rationale) for q in out.queries]
        except Exception as exc:
            log.warning(
                "query_generation_failed_using_text",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            # The sub-question text is a serviceable query on its own.
            proposed = [
                (q.id, q.text, "fallback: the sub-question text, used verbatim") for q in targets
            ]
            errors = [error_from("generate_queries", exc, "using sub-question text")]

        valid_ids = {q.id for q in targets}
        seen = {q.text.strip().lower() for q in completed}
        next_index = len(completed) + 1

        # Group the proposals by the sub-question each serves, keeping the
        # model's order within a group.
        #
        # This replaces taking the proposals in the order they arrived and
        # stopping at the budget, which let the model starve sub-questions
        # it had written nothing for. The live run on 94368bb7 issued six
        # queries across five sub-questions -- two each for SQ1, SQ2 and
        # SQ3 and *none* for SQ4 or SQ5 -- so no candidate in the pool was
        # attributed to either and neither was ever searched. The prompt
        # permits "one or two queries per sub-question" and nothing
        # enforced that every sub-question got one first.
        #
        # The retrieval manifest made it visible; before that, the run
        # reported those two as "only limited evidence was found", which
        # reads as a retrieval outcome rather than as a query that was
        # never sent.
        by_sub_question: dict[str, list[tuple[str, str]]] = {q.id: [] for q in targets}
        for sub_question_id, text, rationale in proposed:
            text = text.strip()
            key = text.lower()
            if not text or key in seen:
                continue
            # A model that invents a sub-question id would break provenance,
            # so unknown ids are reassigned rather than trusted.
            owner = sub_question_id if sub_question_id in valid_ids else targets[0].id
            seen.add(key)
            by_sub_question[owner].append((text, rationale.strip()))

        # A sub-question the model wrote nothing for gets a query from its
        # own text. Long for a search query, and the prompt discourages
        # that -- but a sub-question that is never searched cannot be
        # answered at all, which is the worse of the two.
        for target in targets:
            if by_sub_question[target.id]:
                continue
            key = target.text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                by_sub_question[target.id].append(
                    (target.text.strip(), "fallback: no query was written for this sub-question")
                )

        # Breadth before depth: every sub-question's first query precedes
        # any sub-question's second, so trimming to the budget drops
        # second queries rather than whole sub-questions. Ordered by
        # planner priority, so a budget too small to cover everything
        # spends itself on the essential parts.
        ranked = sorted(targets, key=lambda q: (q.priority, q.id))
        ordered: list[tuple[str, str, str]] = []
        depth = 0
        while True:
            added = False
            for target in ranked:
                group = by_sub_question[target.id]
                if depth < len(group):
                    ordered.append((target.id, *group[depth]))
                    added = True
            if not added:
                break
            depth += 1

        queries = [
            SearchQuery(
                id=f"Q{next_index + offset}",
                sub_question_id=sub_question_id,
                text=text,
                round_number=round_number,
                rationale=rationale,
            )
            for offset, (sub_question_id, text, rationale) in enumerate(ordered[:remaining_budget])
        ]

        unsearched = [t.id for t in ranked if not any(q.sub_question_id == t.id for q in queries)]
        if unsearched:
            # Budget genuinely too small to cover every sub-question.
            # Logged rather than hidden: a gap whose cause is "no query
            # was issued" needs a different fix from one whose cause is
            # "the search found nothing".
            log.warning(
                "sub_questions_unsearched",
                ids=unsearched,
                budget=remaining_budget,
                sub_questions=len(targets),
            )

        if not queries:
            queries = [
                SearchQuery(
                    id=f"Q{next_index + i}",
                    sub_question_id=q.id,
                    text=q.text,
                    round_number=round_number,
                )
                for i, q in enumerate(targets[:remaining_budget])
            ]

    log.info(
        "queries_generated",
        round=round_number,
        count=len(queries),
        # Query, the sub-question it serves, and why it was phrased that
        # way. A query alone cannot be judged: the overfitting run's
        # queries looked plausible until you saw which part of the
        # answer each was supposed to supply.
        queries=[
            {"id": q.id, "text": q.text, "for": q.sub_question_id, "why": q.rationale}
            for q in queries
        ],
        median_query_words=_median_word_count([q.text for q in queries]),
    )
    emit(
        "queries_generated",
        round=round_number,
        count=len(queries),
        queries=[q.text for q in queries],
    )
    return {
        "pending_queries": queries,
        "completed_queries": queries,
        "round_number": round_number,
        "stage_timings": [timing],
        "errors": errors,
    }


async def generate_followups(state: ResearchState) -> ResearchState:
    """Turn coverage gaps into new sub-questions for another round."""
    coverage = state.get("coverage")
    existing = state.get("sub_questions", [])
    round_number = state.get("round_number", 1)
    gaps = []
    if coverage:
        gaps = list(coverage.missing) + [f"weak evidence for {sq_id}" for sq_id in coverage.weak]
    if not gaps:
        return {"stop_reason": "no actionable gaps"}

    emit("generating_followups", gaps=len(gaps))
    with stage("generate_followups") as timing:
        errors = []
        try:
            out = (
                await ctx()
                .router.get(ModelRole.PLANNER)
                .structured(
                    FollowupsOut,
                    FOLLOWUP_SYSTEM,
                    followup_user(_sub_question_block(existing), "\n".join(f"- {g}" for g in gaps)),
                )
            )
            items = out.followups[:4]
        except Exception as exc:
            log.warning(
                "followup_generation_failed",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            items = []
            errors = [error_from("generate_followups", exc)]

        start = len(existing) + 1
        followups = [
            SubQuestion(
                id=f"SQ{start + i}",
                text=item.text,
                rationale=f"follow-up: {item.gap}",
                priority=1,
                round_introduced=round_number + 1,
                is_followup=True,
                parent_gap=item.gap,
            )
            for i, item in enumerate(items)
            if item.text.strip()
        ]

    if not followups:
        return {"stop_reason": "no follow-up questions produced", "errors": errors}

    log.info("followups_generated", count=len(followups), round=round_number + 1)
    emit("followups_generated", count=len(followups), questions=[f.text for f in followups])
    return {"sub_questions": followups, "stage_timings": [timing], "errors": errors}
