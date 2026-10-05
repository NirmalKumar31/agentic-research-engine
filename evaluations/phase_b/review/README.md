# Phase B blinded reviewer packet -- two independent reviewers

## What this is

43 outputs from Benchmark Phase B's frozen-corpus comparison,
blinded of model/provider identity, in two **separately randomized**
packets: `reviewer_a/` and `reviewer_b/`. Each has its own random IDs
and its own candidate ordering -- a candidate's ID in one packet tells
you nothing about its ID in the other. 5 runs timed out and
produced no output; they are marked as such in each question's packet
file and are **not** part of either scoring set, but they remain in the
final operational-reliability numbers regardless of any quality score.

Two reviewers, not one: `docs/BENCHMARK-PROTOCOL.md` requires this --
"two independent reviewers, with disagreement recorded rather than
averaged away" -- because a single reviewer's judgment on subjective
dimensions like clarity or completeness is not itself reproducible
evidence, and because the person who built and ran this benchmark is
not a blind reviewer of their own study even when the packet is
correctly redacted.

## Workflow

1. Each reviewer works from **only their own** `reviewer_a/` or
   `reviewer_b/` directory. Do not compare notes or open the other
   reviewer's directory before both have finished scoring.
2. Open `packet/<question_id>.md` one question at a time.
3. For each `Candidate <random_id>`, score it on that reviewer's own
   `scores_template.csv` against the dimensions below, using the
   question's own rubric and forbidden-overclaims list (shown in the
   packet file).
4. **Do not open `unblinding_key.json` until every row you intend to
   score is filled in.** That file maps each random ID back to its real
   question/repetition/arm -- opening it early defeats the point of
   blinding.
5. Once both reviewers have finished, run
   `python scripts/join_phase_b_review.py --reviewer a` and
   `--reviewer b` to validate and unblind each independently, then
   `python scripts/reconcile_phase_b_review.py` to compute per-dimension
   agreement/disagreement between them and produce the final
   descriptive summary. Disagreements are never silently averaged --
   they are flagged for separate adjudication.

## Scoring dimensions

- **relevance**: 1-5. Does the answer address what the question actually asked, including its pre-registered expected_answerable/expected_refusal_condition?
- **completeness**: 1-5. Does the answer cover the axes/sub-parts a correct answer needs (see the question's own rubric)?
- **clarity**: 1-5. Is the answer well-organized and unambiguous to read?
- **claim_support**: 1-5. Are the claims made actually backed by the cited evidence, not just asserted?
- **citation_usefulness**: 1-5. Do the citations let a reader verify the claim, or are they decorative/misattributed?
- **harmful_or_unsupported_claims**: Yes/No -- does the output contain a claim from the question's forbidden_overclaims list, or another unsupported/harmful claim not on that list?

Record a one-line `rationale` per candidate -- not required to be long,
but a bare number with no reasoning is harder to trust or revisit later.

## What this cannot produce

Per the preregistered protocol: n=2 repetitions per question means any
number that comes out of this process is descriptive, not statistical
evidence of significance, and this benchmark's 12 questions do not
license a general claim about which model or provider is better for
research questions overall. The five local-arm timeouts are a measured
operational fact and are reported as such regardless of what any
quality score says about the runs that did complete. Quality comparison
itself is further limited to the 19 questions-x-repetitions where
**both** arms produced output (see `reconcile_phase_b_review.py`'s
paired-vs-unpaired split) -- the other 5 cloud-only outputs exist
because the local arm timed out on that specific repetition, which is a
selection effect, not a fair additional data point for "cloud quality."
