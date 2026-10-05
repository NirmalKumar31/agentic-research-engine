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

## What's in this directory

| Path | What it is |
|---|---|
| `programmatic.py` | The two entry points above, runnable directly |
| `benchmark/` | `questions.json` -- the five benchmark questions `evaluate`/`compare` run against |
| `quality-eval/` | The frozen adversarial quality set: cases written before a release, so the baseline is honest about what the pipeline did, not what it was later made to do |
| `release-audit/` | Candidate-level validation of the publication gate at the unit it actually publishes -- the atomic generated claim |
| `verifier-calibration/` | Development calibration cases that shaped the verifier designs -- partly fitted, explicitly not an independent benchmark |
| `attribution-experiment/` | What it costs to give every piece of evidence a query-level lineage |
| `live-validation/` | Hosted live-research runs, each committed with its raw stream and a written review -- see [`live-validation/README.md`](live-validation/README.md) |
| `archive/` | Superseded evidence kept for the record, not for citing as current -- `invalid-cloud-comparison/` is referenced from [`docs/LIMITATIONS.md`](../docs/LIMITATIONS.md) as a known-invalid measurement, not deleted so the invalidity stays checkable |
