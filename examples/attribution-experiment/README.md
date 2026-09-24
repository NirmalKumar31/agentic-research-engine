# Cross-attribution experiment

What it costs to give every piece of evidence a query-level lineage.

Under the production default, 79.6% of evidence on this corpus is
`cross_attributed`: it answers a sub-question whose queries never
retrieved that source. The cause is
structural — `dispatch_extraction` shows every source every open
sub-question. The obvious fix is to show each source only the
sub-questions it was retrieved for, which drives cross-attribution to zero
**by construction**. So "does it work?" is the wrong question. It cannot
fail. The question is what it costs.

Run 2026-09-23 against commit `597924aa`, clean tree, `qwen3:4b`
(`359d7dd4bcdab3d8`, Q4_K_M) through Ollama. **$0.00** — no cloud model.
Reproduce with:

```bash
agentic-research attribution evaluations/corpus-cloud.json -r 3
```

| File | What it is |
|---|---|
| `attribution-qwen3-4b.json` | Raw artifact: 9 passes, per-repeat rows, aggregates, full provenance block |

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

9/9 passes completed. **No invariant breaches**, no extraction failures,
no rate-limit refusals, no failed provider requests, no source excluded.

## Decision: production keeps `all_open`

Evidence coverage — at least two verified items from at least two distinct
sources — was **identical in all three repeats of every arm**: 100%,
16.7%, 50.0%, zero variance. That is not sampling noise. It is structural,
and it is the whole finding.

The cause is in the corpus, not the model. Each source here was retrieved
for a mean of **1.17** sub-questions. Under `retrieved_only`, most
sub-questions can therefore only ever see a single source, and a criterion
requiring *two distinct sources* becomes close to unreachable. B's 16.7% is
near its structural ceiling on this corpus, not a tuning failure.

So the two strategies optimise genuinely competing objectives:

* **`all_open` optimises evidence discovery and corroboration.** More
  sub-questions reach the two-source bar.
* **`retrieved_only` optimises query-level lineage.** Every item can name
  the query that fetched its source.

Given how sources are discovered here, you cannot have both. Coverage is
worth more: a report that cannot corroborate a sub-question across two
sources is a worse report, whereas a citation whose lineage stops at the
source is still a fully checkable citation.

## Cross-attribution is not broken provenance

Worth stating plainly, because the name invites the wrong reading. A
cross-attributed item still has an exact, verified chain:

```
Claim → EvidenceItem → verbatim quote (+ page) → SourceDocument
```

What it lacks is a *search query that retrieved that source specifically
for that sub-question*. That is the conditional half of the chain, and it
is recorded as absent rather than filled in with an unrelated query.

The human analogy is exact: you open a paper to answer one question and
notice something that answers another. The finding is no less real, and
you can still cite the page. You simply did not go looking for it there.

## What this does not establish

* **One corpus.** Six sources on one question. The 1.17 sources-per-
  sub-question discovery structure drives the coverage result, and a
  corpus where sources are retrieved by many queries would narrow the gap.
* **Quote fidelity is the soft finding.** A 75.9% → 88.3% improvement under
  narrowing is plausible — a shorter, more focused prompt should produce
  more faithful quotes — and the A and B ranges do not overlap across three
  repeats. But n=3 on 20–36 items, and strategies ran grouped rather than
  in counterbalanced order, so execution order and thermal state are not
  ruled out. Treat it as a lead worth testing, not a measured effect.
* **Cost is not the axis.** All three arms made 6 extraction calls within
  6% of the same input tokens. Narrowing does not save meaningful work; it
  extracts less per call.
