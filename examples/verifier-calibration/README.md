# Verifier calibration

Thirty claim/evidence pairs from the three canonical recordings, for
measuring whether the entailment verifier agrees with a human reader.

## Why this exists

35 substantive claims were generated across the three runs. One exact
duplicate was removed, and the verifier evaluated the remaining 34: 28
partially supported, 6 unsupported, none supported. The fail-closed gate
removed every one, so the demos publish no synthesised claims at all.

That is what the system did. Whether it was *right* is a separate
question, and the project cannot currently answer it — nobody has
compared this verifier against human labels.

Without that, "the verifier caught 34 overreaches" and "the verifier is
too strict" are equally consistent with the evidence. There is reason to
suspect the second: many rejections are compound claims whose halves
each have supporting evidence. This fixture settles it with labels
instead of argument.

## Blind by construction

`blind_cases.json` holds the claim, its kind, and every cited evidence
item in full — quote, source title, domain, site category, page. It
contains **no verifier verdict and no verifier reason**.

An earlier version put the verdict beside the claim and asked the
reviewer not to look. That is not blind: a label produced after seeing
the model's answer measures agreement, which is the thing being tested.
The verdicts stay in `cases.json` and are joined by `case_id` after
labelling.

Case order is shuffled with a fixed, recorded seed, so one run's
rejections do not appear as a block and hint at the pattern.

## Four excluded cases

Four rejected claims were reconstructed from `CitationIssue.claim_text`,
which truncates at 200 characters for logs and the UI. Their original
wording is unrecoverable — the run artifact stores the same truncation.

They are listed under `excluded_cases` with the truncated text and
`unusable_reason`, and are not offered for labelling. Completing the
missing tails would fabricate the input to a gold label.

`ClaimJudgment` now preserves the complete text of every candidate, so
future recordings cannot lose it this way.

## Labelling

```bash
python examples/verifier-calibration/label.py
```

Shows one case at a time and asks for a label. Progress is saved after
every answer, so you can stop with Ctrl-C and resume.

Read the claim against its evidence and nothing else. No outside
knowledge. Where a case is genuinely ambiguous, say so in the rationale
rather than forcing a label to make the numbers tidy.

### Definitions

**supported** — every material clause is directly stated by, or a
straightforward implication of, the **complete cited evidence set taken
together**. Different clauses may be supported by different evidence
items; a compound claim is supported when every clause is covered
somewhere in the set. Do not require one quote to establish the whole
claim.

**partially_supported** — the core proposition has evidence, but at
least one material clause, qualifier, scope, modality, causal statement,
ranking, quantity or attribution is stronger than what the evidence
states.

**unsupported** — the core proposition is absent, contradicted, or rests
only on a heading or topic label rather than substantive content.

## Scoring

```bash
python examples/verifier-calibration/score.py
```

Joins the labels to the verdicts and reports exact agreement, precision
and recall for *supported*, false positives and negatives, and the
confusion matrix.

Precision and recall are defined against *supported* because that is the
only verdict that publishes. A false positive means an unsupported claim
reached the report — the expensive direction. A false negative costs a
true claim its place, which fails safe.

The current verifier predicted zero supported claims, so supported
precision will be undefined at baseline. Recall and the false-negative
count are still informative, and that is the point.

## What this is not

Not a benchmark. Thirty cases from one model on three questions, labelled
by one reviewer. Enough to tell a systematically over-strict verifier
from a working one; not enough for a general claim about entailment
checking.

If a pattern emerges, fix the general rule causing it. Do not tune
against individual cases until the score reaches 30/30 — that fits the
fixture rather than the problem.
