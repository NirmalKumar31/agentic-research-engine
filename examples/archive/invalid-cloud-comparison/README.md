# Archived: invalid local-vs-cloud comparison

**These artifacts are excluded from current results.** They are kept for
methodology, not for their numbers.

The comparison ran on a frozen corpus whose source text had been stripped.
Every source failed the usability check, which drove `citation_integrity`
to 0% in both arms. The corpus loader now rejects such input
(`DegradedCorpusError`), so this cannot recur.

The claim-support figures from that run — 55.6% local, 81.2% cloud — are
not valid measurements and must not be quoted. Both arms did see identical
input, so the relative comparison was internally consistent, but a run
whose integrity metric reads 0% is not a result.

`comparison-local-vs-luna.json` is the second attempt, whose cloud arm is
empty: the account's daily request quota was exhausted partway through.
It is committed unedited, because a failed arm is part of what happened.

A valid replacement experiment has not been run.

| File | What it is |
|---|---|
| `comparison-local-vs-luna.json` | Second attempt; cloud arm empty, quota exhausted |
| `cloud-run-metrics.json` | Full-graph cloud run: 22 calls, $0.0078, 76s |
| `cloud-run-report.md` | The report that run produced |

The cloud run metrics and report are from a real execution and are
accurate for what they measure. They predate several fixes the run itself
exposed, so they describe an older implementation.
