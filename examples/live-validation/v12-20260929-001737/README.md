# Hosted acceptance — v1.2.0 release candidate

One live run through the deployed HTTP/SSE endpoint, captured byte for
byte, with everything else in this directory derived from those bytes
offline.

**This is deployment acceptance. It is not a research-quality
evaluation.** One run on one question says the pipeline works end to
end under the public budgets. It says nothing about how well the
engine answers questions in general, and it would say nothing even if
the answer had been excellent.

| | |
| --- | --- |
| Service | `agentic-research-engine-rc` (temporary; deleted after acceptance) |
| Commit | `6cf9999c` |
| Version | 1.2.0 |
| Question | How does a large language model differ from a neural network? |
| Duration | 165.4s engine, inside the 240s ceiling |
| Claims | 7 generated, 7 checked, **3 published**, 4 withheld |
| Cost | $0.009466 OpenAI, 6 Tavily credits, Hugging Face uptime not measured |
| Provider requests | 14 of a 30 ceiling |

The public demo stayed on v1.1.1 throughout and was unaffected. It
served its own five-run daily allowance from a separate key in the
same shared store, separated by `DEMO_QUOTA_NAMESPACE`.

## What is in here

| File | What it is |
| --- | --- |
| `stream.raw.sse` | The capture. Bytes as the service sent them, unmodified. |
| `stream.index.jsonl` | Offset, length and arrival time of each chunk. Timing without touching the stream. |
| `events.json` | Every event except the result, parsed. |
| `report.md` | The published report. |
| `contract.json` | The answer contract the question was decomposed into, before retrieval. |
| `propositions.json` | Per-assertion entailment for any claim that was decomposed. **Empty for this run** — see the manual review. |
| `relevance-decisions.json` | Which claims answer the question, and by which gate. |
| `repairs.json` | Every bounded rewrite attempt, accepted and refused. |
| `citations.json` | Each published claim resolved to its quote and source URL. |
| `withheld-reasons.json` | Every claim that did not publish, with untruncated text. |
| `metrics.json` | The engine's metrics plus a derived reconciliation. |
| `provider-usage.json` | Calls, tokens and cost, separated out. |
| `sources.json` | Every source retrieved, cited or not. |
| `environment.json` | The deployment, the limits, the verifier pin, the costs. |
| `manual-review.md` | What a person concluded from reading it. |

## Verifying it

```
cd examples/live-validation/v12-20260929-001737
shasum -a 256 -c checksums.sha256
```

Every derived file can be rebuilt from the capture:

```
python examples/live-validation/tools/acceptance.py build \
  --run-dir examples/live-validation/v12-20260929-001737
```

The rebuild is deterministic, so a file that no longer matches its
checksum was edited rather than derived.

## Reading the numbers

**Source quality is a heuristic.** The `quality` figures in
`sources.json` and `report.md` are computed from domain type and
structure. They are not a measure of truth, and a 0.95 does not mean
a source is right. The highest-scoring source in this run was
retrieved and never cited.

**Cost is per provider, never totalled.** OpenAI token cost is the
engine's arithmetic over provider-reported counts. Tavily was a
free-tier key. Hugging Face bills per hour of endpoint uptime rather
than per request, so no per-run figure exists. A single number across
all three would be invented.

**`known_cost_usd` is marked incomplete** because a response reported
a token category for which this project records no rate. The charge
uses the full input rate, which cannot understate it.
