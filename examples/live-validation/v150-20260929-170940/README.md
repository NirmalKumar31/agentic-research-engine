# Hosted verification — v1.5.0, the meta-claim fix

One live run through the deployed demo, captured byte for byte.

**Deployment verification, not a research-quality evaluation.**

| | |
| --- | --- |
| Service | `agentic-research-engine-live` — the public demo |
| Commit | `636e9e86` |
| Version | 1.5.0 |
| Duration | 171.3s engine, inside the 240s ceiling |
| Claims | 3 generated, **0 published** |
| Errors recorded | 0 |
| Cost | $0.008139 OpenAI, 6 Tavily credits |
| Provider requests | 13 of a 30 ceiling |

## What it verified

**Claims about the evidence are gone.**

| Run | Claims about the evidence |
| --- | --- |
| v1.2.0 | 1, at entailment 0.007 |
| v1.4.1 | **2 of 3**, at 0.031 and 0.115 |
| v1.5.0 (this run) | **0** |

The synthesiser prompt had been teaching it — two of its three worked
examples began *"The source reports"*. It no longer does, and the
model stopped.

## Nothing was published, so look at a replay instead

Zero claims reached the reader. The three recorded runs served by the
demo's **replay** path show complete reports with resolvable
citations, and cost nothing to explore. This artifact is evidence
about one fix, not a demonstration of output.

## What it found

All three claims declared the **optional** `dimension` slot. None
filled `direct_contrast`, the slot a comparison is required to fill.
The relevance gate refused two of them for describing one subject
instead of contrasting them — correctly.

The prompt listed all three slots identically. The contract knew
which was core; the call site flattened the slots to
`(name, description)` pairs and dropped the flag on the way to the
model. Sixth instance in this project of a value computed and then
not passed to the thing that needed it.

Fixed after this capture, so **this run does not verify that fix**.

## Verifying it

```
cd examples/live-validation/v150-20260929-170940
shasum -a 256 -c checksums.sha256

python examples/live-validation/tools/acceptance.py build \
  --run-dir examples/live-validation/v150-20260929-170940
```

## Reading the numbers

**Source quality is a heuristic** from domain type and structure, not
a measure of truth. **Cost is per provider, never totalled**: Tavily
was a free-tier key, and Hugging Face bills per hour of endpoint
uptime rather than per request.
