# Benchmark Phase B

Start here: **[`RESULTS.md`](RESULTS.md)** -- the full write-up: run
count, cost, latency, the blinded-review findings, every disclosed
limitation and protocol deviation. Everything else in this directory
is the raw evidence that write-up is accountable to.

## Layout

| Path | What it is | Status |
|---|---|---|
| `manifest.json` | Frozen pre-run configuration: engine commit, model IDs, corpus hashes, ceilings | Public |
| `environment.json` | Hardware/software snapshot at execution time | Public |
| `raw_results.json` | All 48 arm-runs' metrics, flat | Public |
| `spend_ledger.json` | Every cloud call's recorded cost, cumulative | Public |
| `corpora_public/` | The 12 frozen corpora, redacted of verbatim third-party text | Public (redacted) |
| `runs/` | Full per-run records (48), including rendered markdown | Public |
| `blinded/` | Rendered markdown with model/provider identifiers redacted | Public |
| `review/` | The two-reviewer blinded-scoring process: packets, raw submissions, joins, reconciliation | Public (completed) |
| `adjudication/` | Packet for the 6 unresolved reviewer disagreements | Local only, unused |

## What's local-only, and why

`adjudication/` holds a blinded packet for an independent human
adjudicator -- not committed, because its `adjudication_key.json` would
prematurely reveal arm identity before any adjudicator has scored it.
See `adjudication/README.md` if that directory exists on your checkout
-- it is not committed, so this is not a link
(`scripts/prepare_adjudication_packet.py` regenerates it).

The original full-text corpora and one unredacted-quotes backup were
deleted after being encrypted and verified in duplicate; see
`RESULTS.md`'s redistribution note. Nothing full-text is reachable in
this repository or its git history.

## What's used by the test suite

- `tests/unit/test_phase_b_latency_claims.py` reads `raw_results.json`
  directly -- the published latency ranges in `README.md`/`RESULTS.md`
  are asserted to match it exactly.
- `tests/unit/test_phase_b_review_tooling.py` and
  `test_adjudication_packet_tooling.py` read `review/reconciliation.csv`,
  `blinded/*.md`, and `examples/benchmark/questions.json` to prove the
  review and adjudication tooling stays in sync with the real data.
- `tests/unit/test_benchmark_phase_b_preflight.py` covers the
  preflight fixes (corpus hashing, timeout policy) this run depended on.

Moving or renaming anything under `runs/`, `blinded/`, or `review/`
will break one of the above.

## What may legitimately be cited from this benchmark

Read `RESULTS.md`'s own stated limitations before citing anything --
in particular: n=2 repetitions (no significance claim anywhere), Track
1 only (frozen-corpus comparison, not a live end-to-end claim), and the
blinded review was performed by two separately run Codex (AI) sessions,
not the human reviewers the protocol specifies. A number pulled from
this directory without that context is not a fair representation of
what was measured.

## Reproducing or extending this

The scripts that produced everything here are all under `scripts/`:
`run_phase_b.py` (execution), `prepare_phase_b_review.py` /
`join_phase_b_review.py` / `reconcile_phase_b_review.py` (the review
pipeline), `prepare_adjudication_packet.py` (the unused adjudication
packet), and `redact_phase_b_corpora.py` /
`redact_phase_b_quote_excerpts.py` (the redistribution redaction).
`docs/BENCHMARK-PROTOCOL.md` is the preregistered protocol this run
executed against, including its post-execution status banner.
