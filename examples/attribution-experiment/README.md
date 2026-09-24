# Cross-attribution experiment

What it costs to give every piece of evidence a query-level lineage.

Under the production default, 79.6% of evidence on this corpus is
`cross_attributed`: it answers a sub-question whose queries never retrieved
that source. The cause is structural — `dispatch_extraction` shows every
source every open sub-question. Narrowing that to only the sub-questions a
source was retrieved for drives cross-attribution to zero by construction,
so the question is not whether narrowing works but what it costs.

Run 2026-09-23 against commit `597924aa`, clean tree, `qwen3:4b`
(`359d7dd4bcdab3d8`, Q4_K_M) through Ollama. **$0.00** — no cloud model.

| File | What it is |
|---|---|
| `attribution-qwen3-4b.json` | Raw artifact: 9 passes, per-repeat rows, aggregates, full provenance block |

## Reproducibility

The original frozen corpus is not distributed, so this is an archived
measured run rather than a bit-reproducible experiment. The artifact
preserves the engine commit, model digest, prompt and schema hashes and
configuration fingerprint needed to reconstruct the conditions.

A *new* experiment can be run on a newly frozen corpus. It will not
reproduce the numbers below — different search results, different sources,
different discovery structure:

```bash
agentic-research freeze "your question" -o evaluations/corpus.json
agentic-research attribution evaluations/corpus.json -r 3
```

## Result, n=3

Mean, with `[min..max]` where repeats differed. All three strategies read
the same 6 frozen sources and differ **only** in the sub-question set the
extractor is shown.

| | A `all_open` | B `retrieved_only` | C `adjacent` (k=2) |
|---|---|---|---|
| Citable evidence | **27.33** `[26..29]` | 17.67 `[17..18]` | 18 |
| **Evidence coverage** | **100%** | **16.7%** | 50.0% |
| Sub-questions answered | **100%** | 50.0% | 66.7% |
| Cross-attributed | 79.6% `[77.8..80.6]` | **0.0%** | 59.1% |
| Quote fidelity | 75.9% `[72.2..80.6]` | **88.3%** `[85.0..90.0]` | 81.8% |
| Sources earning their fetch | 83.3% | 83.3% | 83.3% |
| Mean sub-questions shown | 6 | 1.17 | 3.17 |
| Extraction calls | 6 | 6 | 6 |
| Input tokens | 16,603 | 15,660 | 16,066 |
| Duration | 415s `[357..499]` | 242s `[223..254]` | 224s `[205..244]` |

9/9 passes completed. No invariant breaches, no extraction failures, no
rate-limit refusals, no failed provider requests, no source excluded.

## Why production uses `all_open`

Evidence coverage — at least two exact-match items from at least two
distinct sources — took the same value in all three repeats of every arm:
100%, 16.7%, 50.0%. **No variation was observed across the three repeats.**
That is not the same as establishing that sampling noise is absent; three
draws cannot do that.

The structural argument is separate, and is the defensible one. Each source
in this corpus was retrieved for a mean of **1.17** sub-questions. Under
`retrieved_only`, most sub-questions can therefore only ever see a single
source, so a criterion requiring two distinct sources is mechanically
limited regardless of how the model behaves. B's 16.7% is close to its
structural ceiling on this corpus rather than a tuning failure.

The two strategies optimise competing objectives:

* **`all_open`** optimises evidence discovery and corroboration. More
  sub-questions reach the two-source bar.
* **`retrieved_only`** optimises query-level lineage. Every item can name
  the query that fetched its source.

Given how sources are discovered here, both are not simultaneously
available. Production uses `all_open` because this project prioritises
multi-source coverage over complete query-level lineage. That is a design
choice for this system, not a finding that one strategy is generally
better.

## Cross-attribution is not broken provenance

The name invites the wrong reading. A cross-attributed item still has an
exact chain:

```
Claim → EvidenceItem → verbatim quote (+ page) → SourceDocument
```

What it lacks is a *search query that retrieved that source specifically
for that sub-question*. That is the conditional half of the chain, and it
is recorded as absent rather than filled in with an unrelated query.

## What this does not establish

* **One corpus.** Six sources on one question. The 1.17 sources-per-
  sub-question discovery structure drives the coverage result, and a
  corpus where sources are retrieved by many queries would narrow the gap.
* **Quote fidelity is the soft finding.** A 75.9% → 88.3% difference under
  narrowing is plausible — a shorter, more focused prompt should produce
  more faithful quotes — and the A and B ranges do not overlap across three
  repeats. But n=3 on 20–36 items, and strategies ran grouped rather than
  in counterbalanced order, so execution order and thermal state are not
  ruled out. Treat it as a lead worth testing, not a measured effect.
* **Cost is not the axis.** All three arms made 6 extraction calls within
  6% of the same input tokens. Narrowing does not save meaningful work; it
  extracts less per call.
