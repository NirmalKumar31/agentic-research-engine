# Manual review

## Verdict

**One published claim. The gate's second branch, taken as written.**

Live research is correct and thin. It completes, refuses accurately,
publishes a claim a reader can follow to a source, and does not claim
to have done more. It is not a report.

## The published claim, read carefully

> **An LLM is a transformer-based neural network.**

Its quote says exactly that, verbatim from the source, at a resolvable
URL. Against the question — how does an LLM differ from a neural
network — this answers: an LLM *is* one, of a particular kind. When one
subject is a category containing the other there is no contrast to
draw, and the `relationship` slot exists to say so.

One sentence is a starting point, not an answer. Both things are true
and the artifact says both.

## The refusals

All four were correct, and one is structural rather than incidental.

The `direct_contrast` claim asserted two things and the atomicity guard
refused it. That is the tension recorded in the v1.6.0 review: the
contract asks for a contrast, a contrast is compound by nature, and the
guard refuses compound claims because each of the other guards reasons
about "the sentence that supports this claim". The `relationship`
alternative covers the case and covered it here.

Two `dimension` claims described neural networks without contrasting
them — the original v1.1.x failure, caught. One failed the modality
guard.

## What the two fixes under test achieved

| | unbounded | this run |
| --- | --- | --- |
| Generated | 13 | 5 |
| Contrast attempted | no | yes |
| Falsely reported unanswered | — | no |
| Published | 0 | 1 |

Both worked. Neither moved the published count, which is the finding.

## Why the work stops here

The bottleneck is measured. Across 24 generated claims in the earlier
runs, 58% were refused on synthesis quality and 17% on evidence. Six
fixes have now been made upstream of the gates — five of them defects
where a computed value never reached the thing that needed it, and two
where a prompt taught the wrong thing. Each was real, each is tested,
and the published count has moved between zero and three throughout.

Another prompt change would have been a fourth attempt at the same
layer, unverifiable without another paid run, and re-running until the
number looked better is the practice this project exists to refuse.

## Budgets

| | Used | Ceiling |
| --- | --- | --- |
| Wall clock | 142.0s | 240s |
| Provider requests | 14 | 30 |
| OpenAI cost | $0.006774 | $0.05 reserved |
| Search credits | 6 | 8 |

## Honest summary

Eight hosted runs have published 0, 0, 0, 1, 1, 2, 3 and 1. Every
refusal inspected across all of them was the correct refusal. The
verification design is sound and demonstrated end to end on a
deployment; the reports it produces are too thin to be useful as
research.

Live research is therefore experimental and fail-closed, and the
interface, the README and this artifact all say so in those words. The
recorded runs are the demonstration.
