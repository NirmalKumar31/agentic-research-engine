# Manual review — hosted run 20260928-045059

**Nothing was published, so there is nothing to classify.**

The release gate is "zero category-C publications". Zero publications
satisfies it, and it would be dishonest to present that as a quality
result. It is the absence of one.

Reviewer: project author, assisted by Claude. Reviewed 2026-09-28.

## What the run did

Six substantive claims were generated, all six were checked against
their own cited evidence, and all six were withheld. No source was
cited. The report contains no sections and no key findings: what a
reader gets is twelve labelled excerpts from the sources, plus eight
statements about what the evidence did not establish.

## Why each claim was withheld

Recorded verbatim from the run, in `withheld-reasons.json`:

| # | Reason |
| --- | --- |
| 1 | best entailment **0.976** from S2-e1, below the 0.98 threshold |
| 2 | best entailment 0.948 from S3-e4, below the threshold |
| 3 | best entailment 0.007 from S2-e3, below the threshold |
| 4 | every cited quote failed the **attribution** guard |
| 5 | every cited quote failed the **atomicity** and **numeric** guards |
| 6 | every cited quote failed the **atomicity** guard |

Two further entries record a contradiction whose left side failed the
framing guard and whose right side scored 0.001.

## The part worth arguing about

**One claim missed by 0.004.** Entailment 0.976 against a threshold of
0.98. That single claim is the difference between a report with a
finding in it and a report with none.

That is not evidence the threshold is wrong. It was calibrated against
human labels with a zero-false-positive acceptance gate, and the whole
design says a claim that cannot be shown is not shown. But it is
evidence of what the design costs, and this run is the clearest
demonstration of it so far: a public visitor asking a reasonable
question received no findings at all.

Three of six were withheld by deterministic guards rather than by
entailment — attribution, atomicity, numeric. Those are the P1 items:
the claims were rejected for *how they were phrased*, not for lacking
support. Fixing generation so quantitative findings arrive atomic, and
so attribution names the right actor, is the work that would change
this outcome without touching the gate.

## What this run does and does not establish

**Does:** the deployed HTTP/SSE path works end to end under the public
limits; the verifier recovers from a genuine cold start; publication
fails closed; quote fidelity was 34/34 exact; the pre-dispatch
reservation held at 3.3x actual spend.

**Does not:** anything about answer quality. A zero-publication run is
not a benchmark, and it is not evidence the engine answers questions
well. On the current evidence, **the recorded replay runs remain the
better demonstration of what this project produces**, and the live path
is a constrained public demonstration of the verification machinery.
