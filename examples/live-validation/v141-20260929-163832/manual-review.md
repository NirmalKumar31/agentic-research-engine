# Manual review

## Verdict

**Live research works. The report was empty, and that is a synthesis
problem, not a pipeline problem.**

The run completed in 113s with zero recorded errors, well inside every
budget. Nothing in the engine failed. Three claims were generated and
the gates refused all three — correctly, in each case.

## Nothing was published

Zero claims reached the reader. Stated plainly because a report with
an empty findings section and a confident tone is the failure this
project is built to avoid, and because three generated claims is a
thin sample from which to conclude anything at all.

## Why nothing published

| Claim | Slot | Refused by |
| --- | --- | --- |
| *The evidence describes a neural network by its interconnected-node…* | `direct_contrast` | atomicity guard |
| *The LLM architectures discussed in the study are based on…* | `relationship` | entailment 0.031 |
| *An LLM is trained on massive amounts of text to predict the next token.* | `dimension` | relevance judgement |

Each refusal is right. The first is two assertions. The second is a
statement about what a study discusses, which its quote does not
support. The third describes training and does not address how the
two differ.

**Two of three claims were about the evidence rather than the
subject.** That is the finding worth acting on, and it is not a model
whim — the synthesiser prompt was teaching it. Two of its three worked
examples for splitting a compound claim began *"The source reports"*.
Measured on the pinned checkpoint, against a quote reading "Large
language models are built on artificial neural network
architectures":

| Claim form | Entailment |
| --- | --- |
| plain assertion | **0.998** — publishes |
| *"The source reports that…"* | 0.856 — withheld |
| *"The evidence describes…"* | 0.519 — withheld |

Adding a frame the quote does not have costs up to 0.48 and
guarantees refusal. The prompt now says so, with those numbers in it,
and keeps the one real exception: when the quote itself is framed
("we demonstrate that X"), the frame must be carried, because
deleting it publishes one paper's result as the field's agreement.
That was audit 2 of this project.

**The fix landed after this capture and is therefore unverified.**

## The selection fix

An arXiv source at 0.98, academic, was selected and fetched, and gave
six citable quotes. A pre-fix local run on this question selected six
blogs at best 0.57. The engine is now reading better material.

Whether it *cites* it is still unanswered: nothing published, so no
source was cited. That was the original defect and it remains open on
the hosted path.

## What this run says about v1.4.1's resilience fix

Nothing. No stage failed, so no fallback ran. The fix is covered by
tests; this run neither supports nor contradicts it.

## Budgets

| | Used | Ceiling |
| --- | --- | --- |
| Wall clock | 113.0s | 240s |
| Provider requests | 13 | 30 |
| OpenAI cost | $0.009305 | $0.05 reserved |
| Search credits | 6 | 8 |

## Honest summary

Three hosted runs on this question have published 3, 2 and 0 claims.
The engine reliably completes, reliably refuses what its evidence does
not carry, and its output quality is limited by what the synthesiser
writes and what search returns — not by the verification machinery,
which has behaved correctly in every case examined.
