# Live validation run — 2026-09-27

One bounded live research run against real providers, kept so the
numbers quoted elsewhere in this repository can be checked rather than
taken on trust.

**What this is.** One local CLI run demonstrating fail-closed
publication behaviour on one question, with real OpenAI, Tavily and
hosted-NLI calls.

**What this is not.** It is not hosted-deployment acceptance, not a
benchmark, and not evidence that the engine answers questions well. It
ran under local budgets roughly double the public deployment's, through
the CLI rather than the deployed HTTP/SSE path, against an
already-warm verifier endpoint. See `environment.json` for the exact
comparison.

## Contents

| File | What it holds |
| --- | --- |
| `environment.json` | Commit, budgets, models, verifier pin, cost basis |
| `metrics.json` | Counts, with every category reconciled |
| `published-claims.json` | The 5 published claims and the quotes they cite |
| `manual-review.md` | Hand-check of all 5 against their sources |
| `report.md` | The generated report as produced |
| `sources.json` | The 10 retrieved sources |
| `checksums.sha256` | Integrity of the above |

## The numbers

| Metric | Value |
| --- | --- |
| Question | "What is the measured evidence on whether AI coding assistants improve developer productivity?" |
| Engine commit | `f4f2ac1` (not the branch head — see `environment.json`) |
| Rounds / queries / sources | 2 / 12 / 10 retrieved, 9 usable |
| Evidence items | 54 |
| Claims checked | 40 |
| **Published** | **5** |
| **Withheld** | **35** (4 partially supported + 31 unsupported) |
| Not checked | 0 |
| Quote fidelity | 53/54 exact (98.1%) |
| Evidence reference resolution | 5/5 (100%) |
| OpenAI tokens | 40,222 in / 25,019 out |
| OpenAI cost (engine-calculated) | $0.016532 |
| Wall clock | 237.1s |

### Reading those numbers correctly

**Withheld is 35, not 31.** Only `supported` publishes, so the four
`partially_supported` claims are withheld too. An earlier summary of
this run said 31; that was an arithmetic error in the summary, not in
the engine, which reported "35 generated claim(s) were excluded" in the
report itself.

**Quote fidelity 98.1%** means 53 of 54 extracted quotes matched their
source text as exact normalised substrings. One did not match and was
dropped. Fuzzy matches are tracked separately, are not citable, and
there were none here.

**Evidence reference resolution 100%** is a structural check: all 5
evidence references resolved to citable evidence. It says the citations
point at something real. It is not independent proof that any claim is
true — `manual-review.md` is the closest thing here to that, and it is
a hand-check by an interested party, not an audit.

**Source quality 0.95–0.96** on the two arXiv sources are heuristic
scores computed by this engine from domain and document signals. They
are not objective measures of research quality.

**The cost figure covers OpenAI only.** It is the engine's arithmetic
over provider-reported token counts, priced from `pricing.toml`. Tavily
usage and Hugging Face endpoint compute were not measured, and no
provider dashboard was reconciled, so no total run cost is stated.

## Reproducing

```bash
git checkout f4f2ac1
agentic-research research "What is the measured evidence on whether AI coding assistants improve developer productivity?" --mode cloud
```

This will not reproduce the numbers. The web moves, the models are
non-deterministic, and search results differ by the hour. What should
reproduce is the shape: most generated claims withheld, published
claims carrying resolvable evidence, and limitations naming what the
evidence did not establish.
