# Validation history

The full release-by-release measurement record behind the one-paragraph
summary in `README.md`'s "Measured results" section. Moved here so the
README states the headline numbers without requiring eight releases of
forensic history to find them -- nothing below is summarized away, it's
relocated verbatim.

## Measured results

Three recorded local `qwen3:4b` runs, one round each, verified by the
pinned DeBERTa NLI classifier at threshold 0.98. Every substantive claim
was checked against each of its own quotes — `checked == checkable` in
all three — and only claims one quote carried on its own were published.

| | RAG comparison | NIST framework | Fraud detection |
|---|---|---|---|
| Unique sources | 5 | 5 | 5 |
| Evidence items | 10 | 26 | 30 |
| Generated substantive claims | 11 | 19 | 16 |
| Exact duplicates removed | 5 | 12 | 8 |
| Unique candidates checked | 6/6 | 7/7 | 8/8 |
| **Withheld** | 4 | 4 | 2 |
| **Published** | **2** | **3** | **6** |
| Quote fidelity (exact) | 70% | 92% | 90% |
| Page-cited evidence | 0 | 6 | 0 |
| Duration | 1478s | 1453s | 1067s |

21 unique candidates across the three runs, all checked, **11 published
and 10 withheld**.

### Release validation

Publishing something is easy; publishing only what the evidence supports
is the claim being made. So every candidate — not only the published
ones — was reviewed against the exact quote the gate selected, **with the
automated verdict, score, guard results and publication decision
hidden**, and the labels joined back by case id afterwards. The packet,
the labels and the join are in
[`examples/release-audit/`](../examples/release-audit/).

| | published | withheld |
|---|---|---|
| **reviewer: supported** | 11 | 8 |
| **reviewer: unsupported or uncertain** | **0** | 2 |

Precision 1.00, recall 0.58. **Zero unsupported published claims, zero
uncertain ones.**

Of the 10 withheld: 5 scored below the entailment threshold and 5 were
refused by a deterministic guard before the score mattered (3 atomicity,
1 hedge, 1 numeric). One of those — a claim dropping the source's
*"often"* — is the frequency-deletion rule catching a real overclaim on
live data.

This is a **blinded self-review, not an independent benchmark**: the
reviewer built the system. A packet for a genuinely independent second
reviewer, carrying only claims, quotes and sources — no verdict, score,
guard result or prior label — was built at
[`examples/release-audit/reviewer-packet.json`](../examples/release-audit/reviewer-packet.json)
and has since been used.

**That independent review found what the self-review could not catch
itself: one published claim (of 21) that an independent reviewer judged
unsupported.** A second, independent adjudicator was then given only
that one claim and its quote — no prior label, score, guard result or
publication outcome — and judged it supported. The two independent
judgments disagree. Per the pre-registered decision rule, a disagreement
like this is recorded, not resolved by picking a side: both labels
stand, nothing was tuned or re-scored to make the disagreement go away,
and the case remains open, tracked in issue #32, rather than quietly
counted as resolved either way. Full trail —
packet, labels, the join, the leak found and fixed in the second
adjudicator's packet, and the final result — in
[`examples/release-audit/`](../examples/release-audit/).

### Adversarial quality set

Eleven cases, seventeen claims, frozen in
[`examples/quality-eval/`](../examples/quality-eval/) and run
credential-free. Entailment is pinned per (claim, evidence) pair,
because the 0.98 threshold is not what this measures — the subject is
evidence selection, the guards, relevance, and coverage.

`python examples/quality-eval/run_eval.py`

| | v1.1.1 | v1.2.0 |
|---|---|---|
| **Irrelevant claims published** | **6** | **0** |
| Irrelevant publication rate | 0.429 | 0.0 |
| Wrong evidence selected | 1 | 0 |
| Mean selected source quality | 0.855 | 0.935 |
| Lowest selected source quality | 0.55 | 0.88 |
| Claims published | 14 | 6 |
| Correct claims wrongly withheld | 0 | **1** |
| Cases publishing nothing | 2 | **6** |

**Read the bottom three rows as carefully as the top one.** This is a
precision-for-recall trade, and it is a large one: published claims
fell from 14 to 6, one correct claim is now wrongly withheld that was
not before, and six of eleven cases publish nothing at all. What was
bought is that no claim in the set is published that does not answer
its question, where previously 43% were.

Whether that trade is right depends on what the report is for. For a
research tool whose entire premise is that a citation means something,
withholding a true claim costs a line; publishing a well-cited
irrelevance costs the premise.

### Hosted acceptance

Deployment acceptance, not a research-quality evaluation: one live run
through the deployed endpoint under the public budgets, captured byte
for byte. Everything in
[`examples/live-validation/v12-20260929-001737/`](../examples/live-validation/v12-20260929-001737/)
is derived from those bytes offline, and rebuilding it is a command —
a file that no longer matches `checksums.sha256` was edited, not
derived.

*"How does a large language model differ from a neural network?"* —
chosen because the previous release answered it badly.

| | |
|---|---|
| Generated / checked / **published** | 7 / 7 / **3** |
| Withheld: below threshold | 1 |
| Withheld: guard failure (atomicity) | 1 |
| Withheld: **supported but irrelevant** | **2** |
| Duration | 165.4s of a 240s ceiling |
| Cost | $0.009466 OpenAI, 6 Tavily credits |

The two withheld for irrelevance are the point. Both were entailed by
their own quotes — at **0.996** and **0.990** — and correctly cited:

> *"Neural networks consist of layers of nodes, with each node
> representing a mathematical function."*
> → withheld: *defines neural networks but does not distinguish them
> from LLMs or explain their relationship.*

True, sourced, and not an answer. Under the previous release it would
have published.

The engine then disagreed with its own output. It published three
claims and wrote in its limitations that *"this research did not answer
the question. Nothing published states how large language model (LLM)
and neural network differ; what survived describes them separately."*

**Two findings, both recorded in full, not smoothed over**, in
[`manual-review.md`](../examples/live-validation/v12-20260929-001737/manual-review.md):

1. A rewrite was refused with *"cannot fill the contrast slot"* and the
   identical sentence published — correct, because the two claims
   declared different slots, and unreadable, because the slot was not
   serialised. Fixed for future runs; **the artifact for this run
   cannot be repaired**, because the field was never sent and the
   deployment admits one run a day.
2. For *"how does X differ from Y"* where Y is a superset of X, the
   honest answer is a subset relationship — which the engine found,
   published, and then reported as not answering, because the
   contract's core slot for a comparison was a direct contrast.
   **Fixed:** a comparison is now answered by a contrast *or* a
   relationship. Verified by replaying this run's own contract and
   published slots through the new coverage — the change post-dates
   the capture, and the deployment admits one run a day, so it is
   not covered by a second live run.

**Proposition decomposition was not exercised by this run.** No claim
decomposed into more than one part, so that path is covered by tests
and unproven in production. One run is not a benchmark either: three
published claims here says nothing about the next question.

### The fixes, verified on the deployment

The two defects above were fixed *after* the acceptance capture, so
neither had been through the live pipeline. A second run, on the public
demo at `1d21b110`, closed that:
[`examples/live-validation/v121-20260929-024544/`](../examples/live-validation/v121-20260929-024544/)

Same question. 6 claims generated, **2 published**, 155.2s, $0.008523.
The decisive pair arrived on its own:

| Claim | Slot | Outcome |
| --- | --- | --- |
| *An LLM is a type of neural network that specifically uses transformer…* | `relationship` | **published** |
| *The evidence distinguishes LLMs as a specific class built on n…* | `direct_contrast` | withheld, entailment **0.007** |

The only claim that would have filled the contrast slot directly was
refused by its own evidence, the core requirement was discharged by the
alternative, and the report carries **no** "did not answer the question"
limitation. On this same question, v1.2.0 reported the opposite.

The gate did not become permissive: four of six claims were still
withheld, by four different mechanisms — two on entailment, one by the
relevance judgement, one by the structural check.

**And the run found a third defect.** `AnswerContract.to_dict()` dropped
`satisfied_by`, so nothing outside the engine could see that a core slot
had been discharged by an alternative. The interface recomputes coverage
from the published claims, so it concluded the question was unanswered
and would have printed that directly above a report saying otherwise.
Found by pointing the interface's own logic at the run's payload.

**Not verified by this run:** the missing-`answer_slot` path. This model
declared a slot on every claim, so the behaviour that had made local
mode publish nothing was never reached. Unit tests and two local runs
cover it; this run is silent on it.

## What live research does and does not do

Measured across eight live runs on a deployment, each committed with its
raw stream under [`examples/live-validation/`](../examples/live-validation/).

**What it does.** Completes in about 150s of a 240s ceiling for roughly
$0.009. Grounds the question in a contract before retrieving anything.
Refuses what the evidence does not carry, and says when it has not
answered. Every claim it publishes resolves to an exact-normalised quote
at a live URL. Across those eight runs, **every refusal inspected was the
correct refusal.**

**What it does not do — and where that depends on the question.** The
first eight runs all asked the *same* question, and an earlier version
of this section generalised from them that the reports are "too thin to
use as research". Four more runs across different question shapes
contradict that, so it is retracted:

| Shape | Question | Published |
| --- | --- | --- |
| definition | What is retrieval-augmented generation? | **2 of 3** |
| numeric | What is the context window size of GPT-4 Turbo? | **1 of 1** |
| procedural | How do you fine-tune a model using LoRA? | 1 of 4 |
| comparison (×8 runs) | How does an LLM differ from a neural network? | 0–3, mostly 0–1 |

The question those eight runs used is a **hypernym comparison** — the
engine's worst case by construction. A comparison's core slot is a
direct contrast, a contrast asserts two things, and the atomicity guard
refuses compound claims because every other guard reasons about "the
sentence that supports this claim". The one shape under test was the one
whose core slot is close to unfillable.

On definitional and factual-lookup questions it produces short,
correct, cited answers in 50–120s for under a cent. It stays weak on
comparisons and on procedural questions.

The bottleneck on the weak shapes is measured and it is not the
verification machinery: across 24 generated claims, **58% were refused
on synthesis quality** (38% irrelevant, 21% guard failures) and 17% on
evidence below threshold.

**This is v1.9.0 data.** Six later releases (v1.10–v1.15) targeted the
comparison weakness specifically — see
[`HOW-THIS-WAS-BUILT.md`](HOW-THIS-WAS-BUILT.md) for what each
one found. On the fixed before/after question used there, published
claims went 5 → 7 and complete comparison pairs 0 → 2 at peak. That
measurement is real but is a single question tracked across releases,
not a replacement for the comparison row above — the two are different
measurements and neither supersedes the other. Three of those six
releases were prompt changes with no causal proof behind them, and the
measurement loop was deliberately stopped once run-to-run variance
exceeded the effects being chased; see
[`LIMITATIONS.md`](LIMITATIONS.md#comparisons) for why a
comparison is structurally the hardest shape regardless.

**Live research is still labelled experimental and fail-closed**, and
the interface says so where a visitor is about to use it — one question
per shape is coverage, not a benchmark, and nothing here licenses a
claim of general correctness. What it does license is not overstating
the negative.

The study is at
[`examples/live-validation/question-shapes/`](../examples/live-validation/question-shapes/).
The [adversarial set](../examples/quality-eval/) is the measurement.

### Eight hosted runs, and what each one found

Every run is committed with its raw stream, derived artifacts and a
written review, under
[`examples/live-validation/`](../examples/live-validation/).

| Run | Published | What it found |
| --- | --- | --- |
| v1.2.0 | 3 of 7 | `answer_slot` not serialised; a relationship should answer a comparison |
| v1.2.1 | 2 of 6 | `satisfied_by` not serialised, so the page contradicted the report |
| v1.4.0 | — | the run died at the coverage critique |
| v1.4.1 | 0 of 3 | two of three claims were about the evidence, not the subject |
| v1.5.0 | 0 of 3 | meta-claims gone; no slot marked required |
| v1.6.0 | 1 of 5 | a contrast is not atomic |
| v1.6.1 | 0 of 13 | removing the claim bound made the report worse |
| v1.6.1+ | 1 of 5 | the bound and the judge fixed; output unchanged |

**Every refusal inspected across all six was the correct refusal.**
What was wrong each time was upstream — what the synthesiser was
told, or a value the engine computed and then did not pass on. Six
instances of that second pattern are recorded in the changelog; four
of them passed every test, because the test fakes constructed the
object correctly while production did not.

The v1.6.0 report publishes one claim:

> **LLMs are built upon deep neural networks.** [S5]
>
> *"At their core, LLMs are built upon deep neural networks, enabling
> them to process vast amounts of text and learn complex patterns."*

with an exact-normalised quote resolving to a live URL, and a
limitations section naming what the evidence did not establish. It
does not claim to have failed, because locating one subject inside
the other is what "how does X differ from Y" means when Y is a
category containing X.

One published claim is a thin report and not a measurement of
quality. The [adversarial set](../examples/quality-eval/) is the
measurement; these are deployment evidence.

### How the audits went

Six manually reviewed canonical audits. The first three each published
exactly one claim that survived every automated check and failed a human
read, and each produced a general rule rather than a patch:

| Audit | What escaped | Fix |
|---|---|---|
| 1 | `"might lack"` published as `"lack"` | hedge-deletion guard |
| 2 | `"We demonstrate that X"` published as `"X"` | research-voice guard |
| 3 | `"our dataset"` → `"datasets"`, inside a two-sentence claim | proposition-level atomicity |
| 4 | — | clean |
| 5 | — | clean, blinded |
| 6 | — | clean, blinded, and the first run with atomicity actually wired |

Audit 6 exists because an independent code review found that audits 4
and 5 were produced with the clause-level atomicity check **written and
tested but never called** — the guard still counted sentences. The
property had been reported as enforced and was not. Every guard now has
an integration test that drives the real publication path with a scorer
entailing everything, so a disconnected guard fails a test rather than a
review.

These are product artifacts, served by the demo. Live search is
nondeterministic, so re-running these questions does not recover these
sources — see [LIMITATIONS.md](LIMITATIONS.md).

### Attribution experiment

Three extraction strategies over one frozen six-source corpus, three
repeats each, `qwen3:4b`.

| | all-open | retrieved-only | adjacent |
|---|---|---|---|
| Evidence coverage | **100%** | 16.7% | 50.0% |
| Citable evidence | 27.3 | 17.7 | 18 |
| Cross-attributed | 79.6% | 0% | 59.1% |

No variation was observed across the three repeats of any arm. Three
draws cannot establish that sampling noise is absent; the defensible
point is structural. On this corpus each source was retrieved for a mean
of 1.17 sub-questions, so retrieved-only extraction mechanically limits
how many sub-questions can reach a two-source coverage bar. Production
uses all-open because this project prioritises multi-source coverage
over complete query-level lineage. Scope and caveats:
[`examples/attribution-experiment/`](../examples/attribution-experiment/).

An earlier local-versus-cloud comparison used a corpus whose source text had
been stripped and is excluded from current results.
