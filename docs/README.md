# Documentation

## Current

- **[`ARCHITECTURE.md`](ARCHITECTURE.md)** -- how the engine works and
  why each piece is shaped the way it is.
- **[`LIMITATIONS.md`](LIMITATIONS.md)** -- what's known-incomplete or
  known-risky right now, by area (correctness, security, product,
  reproducibility). Read this before trusting an unqualified claim
  elsewhere in the docs.
- **[`ANSWER-SHAPES.md`](ANSWER-SHAPES.md)** -- what a run's answer
  contract decides deterministically, and what it doesn't.
- **[`BENCHMARK-PROTOCOL.md`](BENCHMARK-PROTOCOL.md)** -- the
  preregistered protocol for the local-vs-cloud benchmark, including
  its post-execution status banner. The actual results are in
  [`evaluations/phase_b/RESULTS.md`](../evaluations/phase_b/RESULTS.md).
- **[`HOW-THIS-WAS-BUILT.md`](HOW-THIS-WAS-BUILT.md)** -- the
  measurement-driven history of output-quality fixes: which
  measurement justified each change, and which proposed fixes were
  withdrawn when the measurement said no.
- **[`VALIDATION-HISTORY.md`](VALIDATION-HISTORY.md)** -- the
  authoritative evidence location: full release-by-release measured
  results, every table, behind the one-paragraph summary in the root
  README's "Measured results".

## Historical

**[`history/`](history/)** -- preserved evidence from past releases and
deployment states. Everything in it describes the project *as it was
at the time it was written* -- commit SHAs, deployment names, and
"current" statements inside those files are not current now. See
[`history/README.md`](history/README.md).

## Root-level documents

[`../README.md`](../README.md) is the entry point (purpose, install,
quickstart, modes, one measured-results summary). [`../CHANGELOG.md`](../CHANGELOG.md)
is the full per-release changelog.
