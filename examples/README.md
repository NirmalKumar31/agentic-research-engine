# Examples

## Questions worth asking it

The engine is built for questions that need decomposition and several sources.
"What is the capital of France?" exercises nothing.

```bash
agentic-research research "Compare modern approaches for detecting fraud in \
highly imbalanced transaction datasets, including their evaluation methodology."

agentic-research research "Are locally hosted open-weight language models viable \
for enterprise document analysis, considering cost, privacy and capability tradeoffs?"

agentic-research research "What are the practical differences between vector \
databases and traditional search engines for retrieval-augmented generation?"
```

These are the first three of the five benchmark questions in
`src/agentic_research/evaluation/benchmark.py`, chosen because each one has
genuinely separable dimensions and at least one commercially biased corner.

## Driving the engine from Python

`examples/programmatic.py` shows the two entry points: `run_research` for a
finished result, and `stream_research` when you want progress events.

## Cheapest possible run

```bash
LLM_MODE=local MAX_RESEARCH_ROUNDS=1 MAX_SOURCES=5 \
  agentic-research research "..." --quiet
```

Local models cost nothing per token; only search credits are spent.
