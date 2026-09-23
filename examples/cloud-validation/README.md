# Cloud validation artifacts

First paid run against `gpt-6-luna`, 2026-09-23. Total across the whole
validation: **$0.010464 over 42 billed requests.**

| File | What it is |
|---|---|
| `cloud-run-metrics.json` | Full-graph cloud run: 22 calls, $0.0078, 76s, 19/19 quotes exact |
| `cloud-run-report.md` | The report that run produced |
| `comparison-local-vs-luna.json` | Controlled A/B over a frozen corpus |

## Reading the comparison file honestly

`comparison-local-vs-luna.json` is the **second** comparison attempt, and
its `luna` arm is empty: the account's 50-requests-per-day quota was
exhausted, and `corpus_problems` / the `ab_arm_failed` log record why. It
is committed as-is rather than edited, because a failed arm is part of what
happened.

The claim-support numbers quoted in the README (55.6% local, 81.2% luna)
come from the **first** attempt, whose corpus had stripped source text.
That broke `citation_integrity` for both arms equally and is why the
corpus loader now refuses such input. `claim_support` is unaffected —
both arms saw byte-identical evidence — but the caveat belongs with the
number.

## What the run cost, precisely

| Item | Requests | Tokens in/out | Cost |
|---|---|---|---|
| Connectivity check | 2 | 393 / 202 | $0.000140 |
| Comparison, luna arm | 18 | 8,582 / 3,336 | $0.002500 |
| Full cloud run | 22 | 22,263 / 11,196 | $0.007824 |
| Second comparison (429) | 0 | 0 / 0 | $0.000000 |
| **Total** | **42** | **31,238 / 14,734** | **$0.010464** |

The provider counted 50 requests against the daily limit; the 8 unbilled
extra were 400s and the 429 itself, which consume quota but no tokens.
