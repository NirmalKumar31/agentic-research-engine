# Evaluations

`evaluations/` is gitignored by default: it's the output directory for
`agentic-research freeze`, `compare`, and `attribution` -- ad hoc local
scratch, not meant to accumulate in version control. The files you'd
see here on a local checkout (`corpus-*.json`, `comparison-*.json`,
`attribution-*.json`, `benchmark-local-*.json`) are exactly that kind
of scratch, untracked, and not part of this repository.

## The one deliberate exception

[`evaluations/phase_b/`](phase_b/) is force-added past the gitignore on
purpose: it's Benchmark Phase B's public, redistributable benchmark
record (redacted corpora, all 48 raw run results, the blinded review,
the spend ledger, environment snapshot) -- not the original full-text
corpora, which are not retained in the repository or working tree.
Preserved because the whole point of a benchmark is that someone else
can check it. See
[`evaluations/phase_b/README.md`](phase_b/README.md) for what's in it
and how the pieces fit together.

Nothing else under `evaluations/` is tracked, and nothing else should
be added here without the same deliberate `git add -f` + redistribution
review `phase_b/` went through.
