# Phase B blinded review -- reconciled descriptive summary

**Reviewers: two independent Codex (AI) sessions, not the human reviewers docs/BENCHMARK-PROTOCOL.md specifies -- a disclosed deviation, not an implied substitute. n=2 repetitions per question. All numbers below are descriptive means over scored outputs, never blended into a single pooled number without showing both raw scores. This is not statistical significance evidence.**

## Paired quality comparison (both arms produced output)

19 question-repetitions where both arms completed (38 scored outputs, 19 per arm). This is the only fair arm-vs-arm quality comparison in this document.

| Dimension | local: a | local: b | cloud: a | cloud: b |
|---|---|---|---|---|
| relevance_1to5 | 2.53 (n=19) | 2.63 (n=19) | 4.00 (n=19) | 4.16 (n=19) |
| completeness_1to5 | 2.47 (n=19) | 2.47 (n=19) | 3.58 (n=19) | 3.58 (n=19) |
| clarity_1to5 | 3.63 (n=19) | 3.58 (n=19) | 4.26 (n=19) | 4.32 (n=19) |
| claim_support_1to5 | 3.47 (n=19) | 3.26 (n=19) | 4.42 (n=19) | 4.05 (n=19) |
| citation_usefulness_1to5 | 2.74 (n=19) | 3.05 (n=19) | 3.37 (n=19) | 3.53 (n=19) |

**Inter-rater disagreements (|diff| >= 2): 2.** Not averaged away -- see `reconciliation.csv`'s `adjudicated_*` columns, left blank pending manual resolution.
  - Q3-procedural rep1 local: claim_support_1to5
  - Q3-procedural rep2 local: claim_support_1to5

**Harmful/unsupported-claims flag disagreement: 4.** One reviewer flagged, the other did not -- needs adjudication, not averaging.
  - Q10-long-tail rep1 cloud
  - Q10-long-tail rep2 cloud
  - Q10-long-tail rep2 local
  - Q11-adversarial-evidence-shape rep1 local

## Unpaired cloud-only outputs (selection-biased, reported separately)

5 question-repetition(s) where the cloud arm produced output but the local arm timed out on that specific repetition. These exist only because of that timeout -- including them in the paired comparison above would silently compare cloud's performance on the full question set against local's performance on an easier subset. Reported here, separately, not averaged into the paired table.

- relevance_1to5: mean 3.40 (n=10, both reviewers pooled)
- completeness_1to5: mean 3.00 (n=10, both reviewers pooled)
- clarity_1to5: mean 4.00 (n=10, both reviewers pooled)
- claim_support_1to5: mean 4.40 (n=10, both reviewers pooled)
- citation_usefulness_1to5: mean 3.70 (n=10, both reviewers pooled)

## Operational reliability (unconditional, independent of any quality score)

- local: 19/24 completed, 5/24 timed out
- cloud: 24/24 completed, 0/24 timed out

These counts hold regardless of this review's quality scores -- a timeout produced no output to score and is not reinterpreted as a quality judgment of any kind.
