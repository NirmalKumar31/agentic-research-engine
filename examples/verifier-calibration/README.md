# Verifier calibration

Twenty claim/evidence pairs from the three canonical recordings, for
measuring whether the entailment verifier agrees with a human reader.

## Why this exists

The verifier judged 18 of 20 generated claims partially supported or
unsupported, and the publication gate removed all 18. That is what the
system did. Whether it was *right* is a separate question, and the
project cannot currently answer it: nobody has compared this verifier
against human labels.

Without that, "the verifier caught 18 overreaches" and "the verifier is
too strict" are equally consistent with the evidence. This fixture makes
the difference measurable.

## What is here

`cases.json` — one case per claim, carrying:

- a stable `case_id`
- the claim text and every cited evidence id
- each evidence item in full: quote, source title, domain, site category, page
- `verifier_verdict` — what the current `qwen3:4b` verifier decided
- `human_label` — **null until reviewed**
- `human_rationale` — one line of reasoning

## Labelling

Read the claim against its evidence and nothing else. No outside
knowledge, and do not read `verifier_verdict` first: a label copied from
the verifier measures agreement with itself.

Use the same three labels the verifier uses. Where a case is genuinely
ambiguous, say so in the rationale rather than forcing a label to make
the numbers tidy.

## Scoring

```bash
python examples/verifier-calibration/score.py
```

Reports exact agreement, precision and recall for *supported*, false
positives and negatives, and the confusion matrix. A false positive here
is the expensive direction: it means an unsupported claim would have been
published.

## What this is not

Not a benchmark. Twenty cases from one model on three questions, labelled
by the author. It is enough to tell a systematically over-strict verifier
from a working one, and not enough for a general claim about entailment
checking.

Fix general rules if a pattern emerges. Do not tune against individual
cases until the score reaches 20/20 — that fits the fixture rather than
the problem.
