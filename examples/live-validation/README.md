# Live validation

Index over this directory's 290 tracked files. Each run below is a
point-in-time capture against a real deployment or real providers, kept
as a record of what was checked and against what -- not a living
description of current behavior. Most subdirectories have their own
`README.md` with full detail; this page just says what each one is, so
the collection reads as curated rather than accumulated.

## Deployment configuration

- [`DEPLOYMENT-EVIDENCE.md`](DEPLOYMENT-EVIDENCE.md) -- what the live
  deployment is configured as, and what was verified against it without
  spending anything. Deployment acceptance, not a quality evaluation.

## Dated hosted/live runs (chronological)

One authorized live run each, captured with full artifacts
(`report.md`, `metrics.json`, `sources.json`, `environment.json`,
`checksums.sha256`, a manual review against pre-committed criteria):

| Directory | What it checked |
|---|---|
| [`20260927-224441-b64016/`](20260927-224441-b64016/) | First bounded live research run against real providers |
| [`hosted-20260928-045059/`](hosted-20260928-045059/) | First run through the deployed HTTP/SSE endpoint |
| [`v12-20260929-001737/`](v12-20260929-001737/) | v1.2.0 release candidate acceptance |
| [`v121-20260929-024544/`](v121-20260929-024544/) | v1.2.1 verification on the public demo |
| [`v141-20260929-163832/`](v141-20260929-163832/) | v1.4.1 -- live research restored |
| [`v150-20260929-170940/`](v150-20260929-170940/) | v1.5.0 -- the meta-claim fix |
| [`v160-20260929-174352/`](v160-20260929-174352/) | v1.6.0 -- a report that answers |
| [`v161-20260929-191831/`](v161-20260929-191831/) | A prediction that was half wrong |
| [`final-20260930-033343/`](final-20260930-033343/) | The pre-committed decision gate run |

## Coverage and failure evidence

- [`question-shapes/`](question-shapes/) -- a coverage sweep across
  multiple question shapes (definition, numeric, procedural, causal,
  comparison, list), each with its own capture. See its own `README.md`
  for which shapes were run and what each measured.
- [`failures/`](failures/) -- captures of runs that failed outright
  (no report, no publishable result), kept for diagnosis rather than
  as benchmark evidence. See its own `README.md`.

## Tooling

- [`tools/acceptance.py`](tools/acceptance.py),
  [`tools/offline_policy_eval.py`](tools/offline_policy_eval.py) -- the
  scripts used to produce and check these captures.

## What this directory does not claim

None of the above is a controlled benchmark: every run here has the
usual live-research confounds (retrieval variance, no repetition, no
frozen corpus). For a controlled frozen-corpus comparison between
models, see [`evaluations/phase_b/`](../../evaluations/phase_b/)
instead -- these two evidence sets measure different things and are
never pooled.
