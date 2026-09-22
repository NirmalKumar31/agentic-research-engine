"""Streamlit front end.

Intentionally thin. It drives the same ``stream_research`` generator the CLI
uses and renders the same event stream, so there is no second copy of the
research logic here and nothing in the engine knows a browser exists.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import asyncio
from typing import Any

import streamlit as st

from agentic_research.config import LLMMode, Settings, get_settings
from agentic_research.llm.router import ModelRouter
from agentic_research.runner import RunResult, stream_research

st.set_page_config(page_title="Agentic Research Engine", page_icon="+", layout="wide")


def load_settings(**overrides: Any) -> Settings:
    get_settings.cache_clear()
    base = get_settings()
    applied = {k: v for k, v in overrides.items() if v is not None}
    return Settings(**{**base.model_dump(), **applied}) if applied else base


async def execute(question: str, settings: Settings, status: Any, log: Any) -> RunResult | None:
    """Consume the run's event stream and reflect it in the UI."""
    lines: list[str] = []
    result: RunResult | None = None

    async for event in stream_research(question, settings):
        name = event.get("event", "")
        if name == "result":
            result = event["result"]
            continue

        label = _describe(event)
        if label:
            status.update(label=label)
            lines.append(label)
            log.code("\n".join(lines[-14:]), language="text")
    return result


def _describe(event: dict[str, Any]) -> str:
    name = event.get("event", "")
    match name:
        case "analyzing_query":
            return "Analysing the question..."
        case "plan_generated":
            return f"Planned {event['count']} sub-questions"
        case "queries_generated":
            return f"Round {event['round']}: searching {event['count']} queries"
        case "search_completed":
            return f"Searched: {event['query'][:60]} ({event['results']} results)"
        case "search_failed":
            return f"Search failed: {event.get('query', '')[:60]}"
        case "sources_deduplicated":
            return (
                f"{event['results']} results -> {event['unique']} unique "
                f"({event['avoided']} duplicate fetches avoided)"
            )
        case "source_retrieved":
            return f"Retrieved [{event['source_id']}] {event['title']}"
        case "sources_registered":
            return f"{event['usable']} usable sources; extracting from {event['extracting']}"
        case "coverage_evaluated":
            verdict = "sufficient" if event["sufficient"] else "gaps remain"
            return f"Coverage {event['ratio']:.0%} - {verdict}"
        case "followups_generated":
            return f"Following up on {event['count']} gap(s)"
        case "synthesizing":
            return f"Synthesising from {event['evidence']} evidence items..."
        case "citations_verified":
            return f"Verified {event['valid']}/{event['total']} citations"
        case "completed":
            return f"Complete ({event['stop_reason']})"
        case "warning":
            return f"Warning: {event['message']}"
        case _:
            return ""


# --- sidebar ---------------------------------------------------------------

st.sidebar.title("Configuration")
mode = st.sidebar.selectbox(
    "Mode",
    [m.value for m in LLMMode],
    index=2,
    help="local runs entirely on Ollama and costs nothing per token.",
)
max_rounds = st.sidebar.slider("Max research rounds", 1, 5, 2)
max_sources = st.sidebar.slider("Max sources", 5, 40, 15)

try:
    preview = load_settings(llm_mode=mode, max_research_rounds=max_rounds, max_sources=max_sources)
    st.sidebar.caption("Model routing")
    for role, spec in sorted(ModelRouter(preview).describe().items()):
        st.sidebar.text(f"{role:12} {spec}")
    config_error = None
except Exception as exc:  # noqa: BLE001 - surfaced in the UI
    preview = None
    config_error = str(exc)

# --- main ------------------------------------------------------------------

st.title("Agentic Research Engine")
st.caption(
    "Decomposes a question, researches it in parallel, tracks every claim back "
    "to a retrieved source, and verifies its own citations."
)

if config_error:
    st.error(f"Configuration problem:\n\n{config_error}")
    st.stop()

question = st.text_area(
    "Research question",
    placeholder="Compare modern approaches for detecting fraud in highly imbalanced "
    "transaction datasets.",
    height=90,
)

if st.button("Run research", type="primary", disabled=not question.strip()):
    assert preview is not None
    with st.status("Starting...", expanded=True) as status:
        log = st.empty()
        try:
            result = asyncio.run(execute(question.strip(), preview, status, log))
        except Exception as exc:  # noqa: BLE001 - shown rather than traced to a terminal
            status.update(label="Failed", state="error")
            st.error(str(exc))
            result = None
        else:
            status.update(label="Complete", state="complete", expanded=False)

    if result is not None:
        metrics = result.metrics
        row = st.columns(5)
        row[0].metric("Rounds", metrics.research_rounds)
        row[1].metric("Sources", f"{metrics.usable_sources}/{metrics.unique_sources}")
        row[2].metric("Evidence", metrics.evidence_items)
        row[3].metric(
            "Citations valid",
            "n/a"
            if metrics.citation_validity_rate is None
            else f"{metrics.citation_validity_rate:.0%}",
        )
        row[4].metric("Cost", metrics.cost_display)

        report_tab, sources_tab, evidence_tab, metrics_tab = st.tabs(
            ["Report", "Sources", "Evidence", "Metrics"]
        )
        with report_tab:
            st.markdown(result.markdown)
            st.download_button("Download report.md", result.markdown, "report.md")
        with sources_tab:
            for source in result.state.get("sources", []):
                status_note = "" if source.is_usable else f" — unusable ({source.fetch_status})"
                st.markdown(
                    f"**[{source.id}]** [{source.title}]({source.url})  \n"
                    f"{source.domain} · {source.source_type.value} · "
                    f"quality {source.quality_score:.2f}{status_note}"
                )
        with evidence_tab:
            for item in result.state.get("evidence", []):
                flag = "" if item.quote_verified else " — quote not found in source"
                st.markdown(
                    f"**[{item.source_id}]** ({item.stance.value}, "
                    f"confidence {item.confidence:.2f}){flag}  \n"
                    f"{item.claim}  \n> {item.quote}"
                )
        with metrics_tab:
            st.json(metrics.model_dump(mode="json"))
