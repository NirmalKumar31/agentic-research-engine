# Release evidence — v1.9.0

What this release is verified to do, by what means, and where that
verification stops. Three levels are kept separate throughout, because
conflating them is the failure this project keeps correcting:

- **implemented and CI-verified** — tests, types, lint, mutation testing;
- **paid-live-validated** — one authorised run through the deployed service;
- **structurally but not live-validated** — correct by construction and
  covered by tests, with no live run behind it.

## The paid validation

One run. One attempt. No retry. Authorised explicitly, and the deployed SHA
was verified twice before anything was spent.

| | |
| --- | --- |
| Question | *"What are the main causes of overfitting in machine learning?"* |
| Deployed commit | `94368bb7` |
| Question shape | **`causal_drivers`** (`shape_source: wording`) |
| Duration | **147.5s** of 240 |
| Cost | **$0.008809** of $0.05 |
| Provider requests | **14** / 30 |
| LLM calls | **14** / 20 |
| Input tokens | **25,662** / 120,000 |
| Output tokens | **12,485** / 20,000 |
| Tavily credits | **6** / 8 |
| Candidates | **all 41 preserved** before governed selection |
| Published | **5 of 6** |
| Answered | **true** |
| Coverage | **2 covered, 3 weak, 0 missing** |
| Query median | **5 words** versus baseline 11 |
| Visible `started` events | **1** |
| Attempts | **1**, no retry |
| Leak scan | no credential, private-path or internal-endpoint leaks detected |
| URL sanitisation | query strings, fragments and userinfo absent from all 41 |

Artifact:
`examples/live-validation/question-shapes/overfitting-20260930-221039/` —
17 of 18 files checksummed and verified (`checksums.sha256` cannot cover
itself). The raw SSE stream is the only copy the service kept: it runs with
`PERSIST_RUNS=false`.

The question was chosen because it had **failed**: the same question on
`b16ad010` published 0 claims of 1 generated with 1-of-5 coverage, having
read a tweet, a LinkedIn-style blog, a newsletter and two papers on double
descent.

## The defect the run found

The run passed and simultaneously exposed a real fault, which is the more
useful half of it:

- **the model proposed queries only for SQ1–SQ3** — two each;
- **SQ4 and SQ5 received none**, so no candidate in the pool of 41 was
  attributed to either and neither was ever searched;
- the report described them as *"only limited evidence was found"*, which
  reads as a retrieval outcome rather than as a question nobody asked;
- **`f8f81727` fixes this structurally**: query assembly is breadth-first in
  planner priority order, a sub-question the model omits gets a query from
  its own text, and an unsearched sub-question now reports `no search query
  was issued for this sub-question`;
- **the fix is fully tested and mutation-tested** — 4 of 4 mutants caught:
  arrival-order truncation, missing fallback injection, missing prompt
  instruction, and the false retrieval gap cause;
- **the fix itself has not been validated against another paid live run.**

**The paid run did not exercise the breadth-first fix.** It predates it. The
run is what revealed the defect; the fix landed afterwards and is
structurally verified only.

## Remaining evidence boundaries

Stated plainly, not softened.

- **The paid run exercised `causal_drivers`** and nothing else.
- **Structured comparison remains offline-validated only** — relationship
  kinds, named-axis pairs and the contrast table are covered by tests and
  mutation testing, with no live run behind them.
- **The yes/no `causal` contract remains offline-validated only.**
- **The deterministic assembly guarantee does not prove that the model will
  voluntarily distribute its proposals.** It proves the engine no longer
  depends on it doing so.
- **Injected fallback queries can be longer than the preferred query
  length.** The prompt asks for three to eight words; a fallback is the
  sub-question's own sentence. Kept because an unsearched sub-question
  cannot be answered at all, which is worse.
- **One successful live question proves the path works, not universal
  research quality.** A different question, domain or day may behave
  differently.
- **Open-web pages may refuse fetching.** A selected source can return an
  HTTP error, a timeout, or content the parser cannot use. The manifest
  records which.
- **The one-round public budget may still end with weak coverage.** This run
  ended with 2 covered and 3 weak. A second round is not enabled, and the
  audit in `docs/vnext-baseline/phase5-budget-audit.md` explains why: round
  one exhausts the query and source caps, and no runtime measurement exists
  for a two-round profile.
- **The provider cost ceiling is an application admission control, not a
  billing guarantee.** `MAX_CLOUD_COST_USD` is checked before dispatch
  against the engine's own accounting. It is not a provider-side spending
  limit and does not reconcile against a dashboard charge.
- **Free shared quota storage can reset if the datastore restarts.** The
  daily admission counter lives in a free Key Value instance keyed by UTC
  day. A restart loses the count, which fails in the permissive direction.

## Verification levels, by feature

| Feature | Level |
| --- | --- |
| Answer-shape override and `shape_source` | CI-verified; exercised live (`causal_drivers` from wording) |
| Comparison-subject separation | CI-verified only |
| `causal` / `causal_drivers` split | CI-verified; `causal_drivers` half exercised live |
| Structured comparison pairs, relationship kinds | **CI-verified only** |
| Query generation (length, vocabulary, shape-aware) | CI-verified; exercised live |
| Retrieval manifest | CI-verified; produced live, all 41 candidates |
| Authority-aware selection, slot allocation | CI-verified; exercised live |
| Bounded topicality and the named relevance ladder | CI-verified; exercised live |
| Single `started` event | CI-verified; exercised live |
| Breadth-first query coverage | **CI-verified only** (landed after the run) |
| Explicit gap causes | CI-verified; partially exercised live |

## Paid-call accounting for this release

| | |
| --- | --- |
| Authorised paid runs | **1** |
| Paid runs performed | **1** |
| OpenAI cost | **$0.008809** (engine-calculated; not dashboard-reconciled) |
| Tavily credits | **6** |
| Daily admissions consumed | 1 of 24 |
| Retries | **0** |

No other paid call was made during this release.
