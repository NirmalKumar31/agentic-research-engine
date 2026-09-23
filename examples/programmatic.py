"""Driving the engine from Python.

Two entry points, depending on whether you want progress:

    run_research(...)     -> RunResult, once it is done
    stream_research(...)  -> async iterator of progress events

Run with:  python examples/programmatic.py
"""

from __future__ import annotations

import asyncio

from agentic_research.config import LLMMode, Settings
from agentic_research.runner import run_research, stream_research

QUESTION = (
    "What evaluation metrics are appropriate for fraud detection models "
    "trained on highly imbalanced data?"
)


def cheap_local_settings() -> Settings:
    """A deliberately small run: one round, few sources, local models."""
    return Settings(
        llm_mode=LLMMode.LOCAL,
        max_research_rounds=1,
        max_sources=6,
        max_sources_per_round=4,
        # Ollama serves one model largely serially; more concurrency here
        # produces queueing rather than throughput.
        max_parallel_local_llm_calls=1,
        llm_timeout_seconds=300,
    )


async def simple() -> None:
    """Wait for the finished result."""
    result = await run_research(QUESTION, cheap_local_settings())

    print(result.markdown)
    print(
        f"\nsources={result.metrics.unique_sources} "
        f"evidence={result.metrics.evidence_items} "
        f"evidence_integrity={result.metrics.evidence_integrity_rate} "
        f"cost={result.metrics.cost_display}"
    )

    # Every claim can be traced back to the text it came from.
    for item in result.state["evidence"][:3]:
        source = next(s for s in result.state["sources"] if s.id == item.source_id)
        print(f"\n[{item.source_id}] {item.claim}")
        print(f'  quote: "{item.quote[:120]}..."  verified={item.quote_verified}')
        print(f"  from:  {source.url}")


async def with_progress() -> None:
    """Consume progress events as the graph runs."""
    async for event in stream_research(QUESTION, cheap_local_settings()):
        if event.get("event") == "result":
            print(f"\ndone in {event['result'].metrics.duration_s:.1f}s")
        else:
            print(f"  {event.get('event')}")


if __name__ == "__main__":
    asyncio.run(simple())
