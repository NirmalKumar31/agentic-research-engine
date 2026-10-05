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
  audit in `docs/history/vnext-baseline/phase5-budget-audit.md` explains why: round
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

---

# Addendum — two further authorised paid runs

Authorised explicitly after the release matrix, to move structured
comparison and the yes/no `causal` contract off "offline-validated only".
Both ran against the deployed release code `5af5bc15`, one attempt each, no
retries, and **no code was tuned to improve either recorded outcome**.

## Run 2 — structured comparison

*"How does retrieval-augmented generation differ from fine-tuning?"*

| | |
| --- | --- |
| Shape | `comparison` (`shape_source: wording`) |
| Comparison subjects | **exactly 2** — `retrieval-augmented generation (RAG)`, `fine-tuning` |
| Context entity correctly excluded | `large language models` — in `entities`, **not** a side |
| Named axes proposed by the analyst | 3 |
| Complete comparison pairs | **2**, on named axes |
| `relationship_discharge` | `''` — answered by pairs, not the escape hatch |
| Answered | **true**, `missing_core_slots: []` |
| Published | **7** claims |
| Duration / cost | 176.0s of 240 · $0.008877 of $0.05 |
| Provider requests / LLM calls | 13 / 30 · 13 / 20 |
| Tavily credits | 5 / 8 |
| Candidates preserved | 23 |
| Visible `started` | 1 |

The published claims form genuine contrasts on shared axes — RAG
incorporating new documents in minutes against fine-tuning taking hours to
days; RAG not modifying the model against fine-tuning adjusting weights; RAG
providing a source reference against fine-tuned responses not.

**This is the exact question that failed on `b16ad010`**, where three
correct claims were published and the run reported it had not answered,
because "language models" was treated as a third side.

## Run 3 — yes/no causal test

*"Does label noise cause overfitting in deep neural networks?"*

| | |
| --- | --- |
| Shape | `causal` (`shape_source: wording`) |
| `candidate_drivers` slot present | **no** — the contract does not contain one |
| Answered | **false**, `missing_core_slots: ['causal_evidence']` |
| Published | 1 claim, declaring the optional `effect_direction` |
| Withheld | 4 — atomicity, entailment 0.001 against a 0.98 threshold, and two fail-closed "relevance could not be judged" |
| Duration / cost | 108.2s of 240 · $0.008759 of $0.05 |
| Provider requests / LLM calls | 13 / 30 · 13 / 20 |
| Tavily credits | 5 / 8 |
| Candidates preserved | 37, with 8 social/forum/blog dropped |
| Sources read | proceedings.mlr.press, arxiv.org, semanticscholar.org, dl.acm.org (fetch failed), and two explanatory pages |
| Visible `started` | 1 |

**Not answering is the correct outcome here, and it is what validates the
contract.** The engine found material about memorising noise and early
stopping, and no quote entailed a causal claim at the threshold. It refused
to let a plausible driver or an association discharge `causal_evidence`,
published the one supported claim it had, and said plainly that it had not
answered the question. One fetch failure was recorded as such and remains
distinguishable from a selection or relevance failure.

## The defect run 2 found

The comparison run reported the question answered with two complete pairs —
**and shipped a report with no contrast table in it.**

There are two renderers. `finalize` calls `render_markdown` for
`final_markdown`; `runner._render` calls it for the `markdown` the web
result actually carries. Only `finalize` had been given the pairs. The test
written for that feature asserted the one call site it knew about and passed
throughout.

This is the tenth instance in this repository of a value computed correctly
and never handed to the thing that needed it.

Fixed by sharing one implementation — `comparison.pairs_from_payload` — and
feeding both call sites. Verified by re-rendering **this run's own payload**,
which now produces the table. Two new tests: one enumerates every
`render_markdown` call site from the source and fails when a new one appears
unfed, the other drives both renderers. Both mutations caught.

**The fix postdates both runs and was not exercised by either.** It is
structurally verified only, exactly as the breadth-first query fix is.

## Updated verification levels

| Feature | Level after this addendum |
| --- | --- |
| Structured comparison pairs, named axes, subject separation | **paid-live-validated** (run 2) |
| Yes/no `causal` contract, refusal to substitute drivers | **paid-live-validated** (run 3) |
| Relationship-kind discharge | still CI-verified only — run 2 answered by pairs, so the escape hatch was not exercised |
| Contrast table rendering in both renderers | CI-verified only (fix postdates the runs) |
| Breadth-first query coverage | CI-verified only (fix postdates run 1) |

## Paid accounting after the addendum

| | |
| --- | --- |
| Paid runs this release | **3** |
| OpenAI cost | $0.008809 + $0.008877 + $0.008759 = **$0.026445** |
| Tavily credits | 6 + 5 + 5 = **16** |
| Daily admissions | 3 of 24 |
| Retries | **0** |

All three within every configured ceiling. No leak detected in any artifact;
all candidate URLs free of query strings, fragments and userinfo; exactly one
visible `started` event in each.

## A limitation found while choosing run 3's question

The causal-test pattern fires only on explicit causal verbs — *cause*,
*lead to*, *result in*, *produce*, *drive*, *responsible for*,
*contribute to*. So *"Does dropout **reduce** overfitting?"* reads as no
explicit shape and falls back to the model's label. Recorded, not fixed:
widening the verb set during release closure is the kind of change this
release is closing.
