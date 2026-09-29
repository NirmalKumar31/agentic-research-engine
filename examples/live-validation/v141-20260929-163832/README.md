# Hosted verification — v1.4.1, live research restored

One live run through the deployed demo, captured byte for byte.

The first hosted run on v1.4.0 died at the coverage critique and
produced no report. This one exists to answer two questions: does live
research complete on v1.4.1, and did the source-quality fixes change
which pages the engine reads.

**Deployment verification, not a research-quality evaluation.**

| | |
| --- | --- |
| Service | `agentic-research-engine-live` — the public demo |
| Commit | `c8f9d14f` |
| Version | 1.4.1 |
| Duration | 113.0s engine, inside the 240s ceiling |
| Claims | 3 generated, 3 checked, **0 published** |
| Errors recorded | **0** |
| Cost | $0.009305 OpenAI, 6 Tavily credits |
| Provider requests | 13 of a 30 ceiling |

## What it verified

**Live research completes.** Clean run to `completed`, zero recorded
errors. Nothing degraded, so the widened failure handlers added in
v1.4.1 were not exercised — this shows the pipeline works, not that
the resilience fix does.

**Selection reads better pages.** An arXiv source at quality 0.98,
classified academic, was chosen and fetched, and yielded six citable
quotes. A pre-fix local run on this question selected six blogs, best
quality 0.57.

## What it did not verify

**Whether the better source gets cited.** Nothing published, so no
source was cited and the question the selection fix was meant to
answer is still open on the hosted path.

**The resilience fix.** Nothing failed, so nothing degraded.

## What it found

Two of the three generated claims were about the *evidence* rather
than about the subject:

- *"The evidence describes a neural network by its interconnected-node
  computational structure…"* — refused, atomicity
- *"The LLM architectures discussed in the study are based on…"* —
  refused, entailment **0.031**

The third was judged irrelevant. So the report published nothing, and
two-thirds of its budget went on claims the verifier cannot accept:
the premise is the quote, and the quote does not say what the evidence
describes — it just says the thing.

The synthesiser prompt was teaching this. Two of its three worked
examples for splitting a compound claim began *"The source reports"*.
Measured on the pinned checkpoint, adding a frame the quote lacks
costs up to 0.48 entailment. Fixed after this capture, so **this run
does not verify that fix**.

## Nothing was published, so look at a replay instead

This run published zero claims. As a demonstration of what the engine
produces it is not the one to read — the three recorded runs served by
the demo's **replay** path show complete reports with resolvable
citations, and cost nothing to explore.

Read this artifact for what it is: evidence that live research
completes on v1.4.1, that selection now reaches an academic source,
and that the synthesiser wrote two claims about the evidence instead
of about the subject.

## Verifying it

```
cd examples/live-validation/v141-20260929-163832
shasum -a 256 -c checksums.sha256

python examples/live-validation/tools/acceptance.py build \
  --run-dir examples/live-validation/v141-20260929-163832
```

## Reading the numbers

**Source quality is a heuristic** from domain type and structure, not
a measure of truth. **Cost is per provider, never totalled**: Tavily
was a free-tier key, and Hugging Face bills per hour of endpoint
uptime rather than per request.
