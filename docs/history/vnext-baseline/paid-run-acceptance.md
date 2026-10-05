# Paid-run acceptance plan

Written **before** the run, so the criteria cannot be chosen to fit the
result. One run, no automatic retry.

## Prerequisite that is not mine to satisfy

The deployed service runs **`514cddc9`** (main). This branch is
**`ecc05db7`**. A paid run against the demo today exercises the *old*
code and would tell us nothing about any fix in this branch.

So the run requires the branch to be deployed first, and deploying is
the owner's action. Until then there is nothing worth spending on.

## The run

Question, verbatim — the preserved baseline:

```
What are the main causes of overfitting in machine learning?
```

Command, from the repository root:

```
python examples/live-validation/tools/acceptance.py capture \
  --base https://agentic-research-engine-live.onrender.com \
  --question "What are the main causes of overfitting in machine learning?" \
  --out examples/live-validation/question-shapes/overfitting-<UTC-timestamp>
```

Then, offline and repeatable against the same bytes:

```
python examples/live-validation/tools/acceptance.py build \
  --run-dir examples/live-validation/question-shapes/overfitting-<UTC-timestamp>
```

The capture writes raw stream bytes to disk as they arrive and records
timing in a separate index, because the first hosted capture was done by
hand, truncated each line at 150 characters, and destroyed the only copy
of its own result.

## Maximum exposure

The **configured hard ceilings**, not the ~$0.008 estimate from previous
runs. The estimate is what similar runs drew; the ceiling is what this
one could:

| Ceiling | Value |
| --- | --- |
| `MAX_CLOUD_COST_USD` | **$0.05** |
| `MAX_PROVIDER_REQUESTS` | 30 |
| `MAX_CLOUD_CALLS` | 20 |
| `MAX_CLOUD_INPUT_TOKENS` | 120,000 |
| `MAX_CLOUD_OUTPUT_TOKENS` | 20,000 |
| `MAX_SEARCH_CREDITS` | 8 |
| `MAX_RUNTIME_SECONDS` | 240 |

Worst case: **$0.05 of OpenAI budget and 8 Tavily search credits**, plus
one of the 24 daily run admissions. Tavily credits are not bounded by
the OpenAI project limit.

## Pass criteria

Every one is checkable from the preserved artifact.

**Queries**
- [ ] preserve the question's own vocabulary; "overfitting" and
      "machine learning" appear
- [ ] median length within three to eight words
- [ ] no query stacks specialist terminology the question never used
- [ ] each query records its sub-question and a rationale

**Selection**
- [ ] sources allocated across answer slots, not concentrated on one
- [ ] a candidate serving several sub-questions is credited to all
- [ ] no social or newsletter source displaces an accountable source
      that the provider rated within 0.35 of it
- [ ] every selected source and every drop reason recorded
- [ ] fetch failure distinguishable from selection and from relevance
      failure

**Coverage and claims**
- [ ] evidence counted toward a slot is admissible, not merely
      quote-exact
- [ ] every gap carries a cause
- [ ] every published claim is supported and relevant
- [ ] **unsupported publications: 0**
- [ ] **false "answered" conclusion: 0**

**Lifecycle and budget**
- [ ] **duplicate `started` events: 0**
- [ ] runtime, provider requests, search credits, LLM calls and cost all
      within the ceilings above
- [ ] artifact contains no credentials, private paths, internal
      endpoints or secret-bearing headers (gitleaks over the directory
      before it is committed)

## Comparison against the preserved baseline

Baseline: the hosted overfitting run on `b16ad010` — 6 sources (a tweet,
a LinkedIn-style blog, a newsletter, two double-descent papers), 1 of 5
covered, **0 published of 1 generated**, 16 evidence items.

Compare: sources and source classes, coverage by slot, gap causes,
published and withheld claims, runtime, provider/search/LLM usage, cost.

## What counts as failure

**Not** a zero-publication result on its own. If the available web
evidence genuinely does not answer the question, publishing nothing and
saying so is the design working.

Failure is:

- selecting avoidably poor sources when better ones were in the pool;
- misclassifying coverage — counting evidence toward a slot it does not
  address, or reporting a gap whose cause is wrong;
- claiming the question was answered when it was not;
- any unsupported claim published;
- a duplicate lifecycle event;
- exceeding any ceiling.

## What this run does *not* validate

It exercises query generation, Tavily candidate quality, selection,
topical coverage admission, gap causes and the SSE lifecycle.

It does **not** exercise the structured-comparison work — relationship
kinds, named-axis pairs, the contrast table — because the question is
not a comparison. If this run passes, comparison behaviour remains
**offline-validated only**, and a separate authorisation would be needed
for a RAG-versus-fine-tuning live run. That is a second paid run and is
not to be added silently to this one.

Nor does it validate the causal split beyond the driver-seeking half:
"what are the main causes of X" is `causal_drivers`, so the yes/no
`causal` contract is exercised only by offline tests.
