# Changelog

Notable changes per release. Dates are UTC.

## v1.1.1 — 2026-09-28

Progress reporting only. No change to research behaviour, verification
thresholds, publication gates or recorded results.

- The runner emits `verifier_waking` **before** the readiness probe
  starts, rather than after it finishes. A wake announced afterwards
  describes a wait that is already over.
- The event carries the configured wake budget, so the interface can
  state how long the wait may be instead of leaving a reader to guess
  whether the page has stalled.
- `verifier_ready` follows a successful readiness check.
- The interface explains the cold start — "Waking the verifier (up to
  ~90s)" — where it previously showed step 1 with no explanation.
- Local verification shows no remote wake estimate. Quoting a
  scale-to-zero budget for a checkpoint loaded from disk would be a
  number invented for the occasion.

Why: a verifier at minimum replicas 0 takes roughly a minute to start,
and that happens after `started` and before the first pipeline stage.
Measured on the committed acceptance capture, 70.7s of pipeline stages
inside a 144.4s run left 73.7s outside them, carrying heartbeats and
nothing else. The work was real and was never narrated, so the page
read as hung for more than a third of the run.

## v1.1.0 — 2026-09-28

Live research integration. See
[the release notes](https://github.com/NirmalKumar31/agentic-research-engine/releases/tag/v1.1.0)
for the measured hosted run, which published 0 of 6 claims.

## v0.2.0

Replay-only deployment, with the generative claim verifier replaced by
a pinned NLI classifier and deterministic guards.
