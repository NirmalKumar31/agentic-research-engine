# Adversarial quality evaluation

A frozen set of cases for the v1.2.0 research-quality work, written
**before** any change so the baseline is honest about what the pipeline
did rather than what it was later made to do.

## What it measures, and what it does not

Entailment is **pinned per (claim, evidence) pair** in `cases.json`.
The 0.98 NLI threshold is frozen and is not under test here. What is
under test is everything around it: which evidence gets selected,
whether the deterministic guards fire, whether a supported claim
actually answers the question, and whether the report covers what was
asked.

That makes the run deterministic, credential-free and fast. It also
means **no calibration number may be derived from it** — the scores are
fixtures, not measurements.

## Running it

```bash
python examples/quality-eval/run_eval.py
python examples/quality-eval/run_eval.py --json out.json
```

## The v1.1.1 baseline

Recorded at commit `a6009e8`, whose only behavioural change from
v1.1.0 was verifier-wake reporting. Research behaviour is v1.1.0's.

| Metric | v1.1.1 |
| --- | --- |
| Claims evaluated | 10 |
| Published | **7** (3 expected) |
| **Irrelevant published** | **3** — must be 0 |
| Correct claims wrongly withheld | 0 |
| Evidence selected wrongly | 1 |
| Primary-source publications | 4 of 7 |
| Mean selected source quality | 0.796 |
| Zero-finding cases | 2 of 6 |

### What the baseline publishes that it should not

- **`comparison-llm-vs-nn/c2`** — "A large language model is a machine
  learning model designed for natural language processing tasks."
  True, well-supported, and a *definition* answering a *comparison*
  question. This is the live failure a visitor hit on 2026-09-28.
- **`comparison-answered-with-definitions/c2`** — the same shape,
  isolated. Two definitions are not a contrast.
- **`wrong-entity-evidence/c1`** — a MySQL benchmark published in answer
  to a question about PostgreSQL, at entailment 0.999. Entailment
  cannot catch this; nothing currently checks the entity.

### What it selects wrongly

**`comparison-llm-vs-nn/c3`** cites both a tweet (0.994 entailment,
source quality 0.55) and an arXiv paper (0.985, quality 0.96). The
pipeline picks by entailment alone, so it cites the tweet. Source
quality is computed and does not influence selection.

### One the baseline found that was not predicted

**`zero-publication-productivity/c1`** bundles two assertions — a
measured 19% figure, and a claim about self-reports that the quote does
not contain — and the atomicity guard **did not** catch it. It
published. That is a guard gap, not a generation gap, and it is now on
the record.

## Rules for changing this file

Do not edit a case to make a result look better. A case may be added,
or corrected if it misstates what a correct pipeline should do, and
either way `baseline-v1.1.1.json` stays as it is: it is the frozen
comparison point.
