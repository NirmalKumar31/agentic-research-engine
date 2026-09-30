# Hosted validation — vNext quality release

One live run through the deployed demo, captured byte for byte. The single
paid validation authorised for this release.

| | |
| --- | --- |
| Service | `agentic-research-engine-live` — the public demo |
| Commit | `94368bb7` (verified twice before spending) |
| Version | 1.8.0 |
| Question | *"What are the main causes of overfitting in machine learning?"* |
| Duration | 147.5s engine, inside the 240s ceiling |
| Claims | 6 generated, **5 published** |
| Coverage | 2 of 5 sub-questions covered, 3 weak, 0 missing |
| Answered | **yes** |
| Errors recorded | 0 |
| Cost | $0.008809 OpenAI of a $0.05 ceiling, 6 Tavily credits of 8 |
| Provider requests | 14 of 30 |

This re-asks the exact question that failed on `b16ad010`, to establish
empirically whether corrected queries and selection make Tavily adequate for
a broad explanatory question. It was not chosen to flatter the release.

## Against the baseline

| | baseline (`b16ad010`) | this run (`94368bb7`) |
| --- | --- | --- |
| Answer shape | `definition` | **`causal_drivers`** (`shape_source: wording`) |
| Median query length | 11 words | **5 words** |
| First query | `parametric knowledge long tail facts factual recall…` | **`causes of overfitting in machine learning`** |
| Sources read | tweet, LinkedIn-style blog, newsletter, 2 double-descent papers | **2× developers.google.com, geeksforgeeks ×2, lenovo, towardsai** |
| Candidates recorded | 6 of 44 (38 lost permanently) | **all 41** |
| Coverage | 1 of 5 | **2 of 5 covered, 3 weak** |
| Published | **0** of 1 generated | **5** of 6 generated |
| Reported as answered | no | **yes** |
| Visible `started` events | 2 | **1** |

## What published

Five claims, all declaring `candidate_drivers`, each supported by one of its
own cited quotes:

1. a model excessively complex relative to the size and diversity of its
   training data;
2. imbalanced training data;
3. absence of regularization, leaving models free to fit training data too
   closely;
4. training for too many epochs;
5. too many features magnifying noise.

That is a recognisable answer to the question asked.

## What it found — a defect, not a success

**The query writer starved two sub-questions.** Six queries were issued
across five sub-questions: two each for SQ1, SQ2 and SQ3, and **none for SQ4
or SQ5**. No candidate in the pool is attributed to either, so neither was
ever searched. The prompt permits "one or two queries per sub-question"
against a six-query budget without requiring that every sub-question be
covered first.

This is the same defect class as one sub-question consuming the entire source
budget — which this release fixed — but one stage earlier, in query
generation. It was **invisible before the retrieval manifest existed**, and
the manifest found it on its first live run. It is recorded here and not
fixed; fixing it is a separate change with its own validation.

The run's own limitations name the consequence honestly: three sub-questions
report "only limited evidence was found", and the two unsearched ones are
among them.

## What this run does not validate

It is a `causal_drivers` question. It exercises **neither** the structured
comparison work (relationship kinds, named-axis pairs, the contrast table)
**nor** the yes/no `causal` contract. Both remain **offline-validated only**
and would need separately authorised runs.

## Reproducing the derivation

Every file except this one, `manual-review.md` and `environment.json` is
derived from `stream.raw.sse` by:

```
python examples/live-validation/tools/acceptance.py build --run-dir <this directory>
```

The stream is the only copy the service kept: it runs with
`PERSIST_RUNS=false`.
