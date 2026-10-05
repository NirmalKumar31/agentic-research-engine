# Release test matrix — v1.9.0

Run on the release-prep tree, branched from `9e347cbe` (9/9 CI green).
Every line is a real exit code or a count from real output. Three gates
could not run on this machine and say so rather than being reported as
passes.

## Tests and coverage

| Gate | Result |
| --- | --- |
| full pytest (CI marker expression) | **2309 passed, 31 skipped, 20 deselected** |
| branch coverage | **89%** (7899 statements, 2104 branches) |
| real-NLI suite (`-m nli`, pinned DeBERTa checkpoint) | exit 0 |
| artifact / checksum suites | 41 passed |
| recording / replay suites | 80 passed |
| NLI pin + build contracts | 8 passed |
| security-focused suites (log redaction, trusted proxy, URL safety) | 90 passed |

## Static analysis

| Gate | Result |
| --- | --- |
| `ruff check src tests examples` | exit 0 |
| `ruff format --check src tests examples` | exit 0 |
| `mypy` (source) | exit 0 |
| `mypy examples` | exit 0 |
| `git diff --check` | exit 0 |

### The SSE whitespace exception, checked separately

A lossless SSE capture ends with the blank line the protocol requires: an
event block is terminated by a blank line, so a well-formed stream ends with
one. `git diff --check` reports that as *"new blank line at EOF"* when such a
file is first added.

Those files are **not modified to satisfy whitespace tooling** — altering a
capture would break both the lossless guarantee and its checksum. Verified
across the committed captures: each ends with exactly `\n`.

This release's changes contain **no SSE file**, so `git diff --check` is
clean on them without needing the exception. The exception is documented here
because it will arise the next time a capture is added, and because
re-running the check with `':(exclude)*/stream.raw.sse'` also exits 0 —
confirming no *other* whitespace defect is hiding behind it.

## Web

| Gate | Result |
| --- | --- |
| `npm run typecheck` | exit 0 |
| `npm test` | **66 passed** across 10 files |
| `npm run build` (production) | exit 0 — 255,399 B JS, 16,964 B CSS |
| `npm audit --omit=dev --audit-level=high` | exit 0 |

## Security

| Gate | Result |
| --- | --- |
| `pip-audit --skip-editable` | exit 0 |
| gitleaks — complete history | exit 0 |
| gitleaks — tracked tree (551 files) | exit 0 |

### A correction worth recording

Scanning the working directory with `--no-git --source .` reports 23
findings, and **none of them are tracked**. They are in `.env`,
`.pytest_cache/v/cache/nodeids` and two `__pycache__/*.pyc` files — all
untracked and all gitignored, confirmed with `git ls-files` and
`git check-ignore`. `.env` holds the owner's real local credentials and was
neither read nor echoed.

That invocation is the wrong gate for "tracked tree": it scans local
artifacts the repository does not contain. The correct scan extracts
`git archive HEAD` and scans that — 551 files, exit 0.

## Packaging

| Gate | Result |
| --- | --- |
| wheel build | exit 0 — `agentic_research_engine-1.9.0-py3-none-any.whl` |
| sdist build | exit 0 — `agentic_research_engine-1.9.0.tar.gz` |
| clean-venv wheel install | exit 0 |
| version reported by installed package | **1.9.0** |
| CLI in installed package (`--help`) | exit 0 |
| imports exercised after install | `retrieval.manifest`, `question_form` |

## Not runnable on this machine

No container runtime is present (`docker` and `podman` both
`command not found`). These three are **CI-only** and are not claimed as
local passes:

| Gate | Covered by |
| --- | --- |
| Docker build | CI job *Build the image and run the CLI inside it* |
| CLI-in-container smoke | same job |
| web-container health / readiness smoke | CI job *Build the web image and probe its health endpoint* |

Both jobs were green on `9e347cbe` and will run again on the release SHA.

## Query-coverage non-vacuity checks

Each restores the exact pre-fix behaviour and confirms a test rejects it. A
mutation no test catches is not counted.

| Check | Result | Caught by |
| --- | --- | --- |
| 1. restore arrival-order truncation | regression fails | `test_the_node_interleaves_rather_than_taking_arrival_order` |
| 2. remove injected fallback queries | regression fails | `test_the_node_injects_a_query_for_an_uncovered_sub_question` |
| 3. remove the prompt's breadth-first instruction | regression fails | `test_the_prompt_requires_coverage_before_depth` |
| 4. restore the false retrieval gap cause | regression fails | `test_a_sub_question_nobody_queried_says_so` |

**4/4 counted.**

## Version locations changed

Four authoritative locations, and nothing else:

- `pyproject.toml`
- `src/agentic_research/__init__.py`
- `web/package.json`
- `web/package-lock.json` — the two root entries only, edited structurally
  (`d["version"]` and `d["packages"][""]["version"]`) so no dependency's own
  version could be touched. The diff is exactly two lines.

Historical dependency versions and earlier evidence artifacts were not
rewritten.
