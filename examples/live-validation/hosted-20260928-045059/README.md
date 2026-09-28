# Hosted acceptance — 2026-09-28

One authorized live run through the **deployed** HTTP/SSE endpoint,
under the public deployment's own limits, captured byte-for-byte.

**This is deployment acceptance. It is not a research-quality
evaluation.** The run published nothing, which is recorded here as it
happened rather than repeated until it looked better.

## Headline

| | |
| --- | --- |
| Deployed commit | `6e34f908` (v1.1.0) |
| Duration | **144.4s** against the 240s ceiling |
| Verifier at dispatch | **`scaledToZero`** — a genuine cold start |
| Sources retrieved / cited | 6 / **0** |
| Evidence items | 34, all quotes exact (34/34) |
| Claims generated / checked | 6 / 6 |
| **Published** | **0** |
| **Withheld** | **6** (2 partially supported, 4 unsupported) |
| Key findings | 0 |
| OpenAI cost | **$0.009472** (incomplete — see below) |
| Reserved upper bound | $0.03107 — held, 3.3× actual |
| Tavily | 6 credits |

## Why nothing published

Three claims fell below the 0.98 entailment threshold — one of them at
**0.976**, missing by 0.004. Three failed deterministic guards
(attribution, atomicity, numeric) — rejected for how they were phrased,
not for lacking support.

`manual-review.md` argues about this properly. The short version: the
gate behaved as designed, and this run is the clearest demonstration
yet of what that design costs a visitor.

## Why the cost figure says "incomplete"

`unpriced_calls` is 0, so every model had a price. The flag fired
because a response reported a token category — cached input or cache
write — for which this project records no rate. The charge uses the
full input rate, which cannot understate it, and the run declines to
call its own cost complete.

This is the first run under that instrumentation, and it caught
something on the first try. Before this change the same run would have
reported its cost as complete.

The figure covers **OpenAI only**. Tavily credits are counted (6) but
not priced; Hugging Face bills per hour of endpoint uptime, not per
request, and was not measured. No total run cost is stated.

## Contents

| File | What it holds |
| --- | --- |
| `stream.raw.sse` | The complete SSE stream, byte-for-byte, untransformed |
| `stream.index.jsonl` | Arrival timing per chunk, kept separate so the stream is not rewritten |
| `metrics.json` | Derived counts, every category reconciled |
| `withheld-reasons.json` | Why each claim was withheld, verbatim |
| `report.md` | The generated report as produced |
| `sources.json` | The 6 retrieved sources |
| `environment.json` | Commit, limits, models, verifier pin, cost scope |
| `manual-review.md` | Review of the outcome |
| `checksums.sha256` | Integrity of the above |

The previous CLI capture truncated each line at 150 characters and
destroyed the result payload, which the service does not persist and so
could not be recovered. This capture writes raw bytes to disk and
records timing separately.

## Verifying

```bash
cd examples/live-validation/hosted-20260928-045059
shasum -a 256 -c checksums.sha256
```

## Honest reading

The deployment works: public limits enforced, cold start recovered,
fail-closed publication, exact quotes, reservation sufficient.

The output does not demonstrate research quality, and one run could not
demonstrate it either way. **The recorded replay runs remain the better
demonstration of what this engine produces.** The live path shows the
verification machinery refusing to publish, which is the honest thing
to show and a thin thing to read.
