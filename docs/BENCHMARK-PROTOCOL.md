# Local-vs-cloud benchmark protocol (Phase A)

> **Post-execution status (added after the fact; the Phase A text below is
> preserved unchanged as the historical preregistration, not rewritten to
> match what happened):** Phase B was subsequently authorized and executed.
> Full results: [`evaluations/phase_b/RESULTS.md`](../evaluations/phase_b/RESULTS.md).
> Track 1 (frozen-corpus comparison) only was run; Track 2 (live
> end-to-end) remains out of scope and unexecuted. Two deviations from
> this document as originally written: the freeze-time corpus-abort rule
> was changed mid-study (a single degraded source is no longer treated as
> fatal), and the blinded review below was performed by **two separately
> run, blinded Codex (AI) sessions** -- blinded to candidate identity,
> not independent in a statistical or institutional sense -- not the two
> human reviewers this protocol specifies -- a disclosed substitution,
> not a silent one. That review
> surfaced 6 reviewer disagreements (2 score, 4 harmful-claim-flag) that
> remain **unadjudicated** as of this banner; see RESULTS.md's "Blinded
> review results" for which ones and why.

This is construction and validation, not a result. No question in this
document has been run, and this document does not authorize running one.
Phase B — any paid or cloud run — needs separate, explicit authorization
from the project owner, after reviewing the exact run count and
worst-case cost stated at the bottom of this document.

## Why this exists

An earlier local-vs-cloud comparison was invalid: the evidence corpus it
compared both arms against had its source text stripped, which drives
citation integrity to 0% in both arms regardless of which model is
running — a broken fixture that briefly read as a finding. That specific
failure mode is now a structural guard
(`evaluation/ab.py`'s `EvidenceCorpus.validate_for_replay` and
`DegradedCorpusError`), not a thing to remember to check by hand. This
protocol is the replacement: two explicit tracks, a frozen manifest, a
fixed question set with a rubric decided before any answer exists, and a
stated stopping rule.

## The manifest

`evaluation/benchmark_manifest.py`'s `BenchmarkManifest` freezes, before
the first question is asked: the engine commit, the cloud model id, the
Ollama model/version/digest, the pinned NLI model and its 40-character
revision, the prompt and schema versions, a config fingerprint, the
corpus hashes, and the ceilings every arm runs under.
`completeness_problems()` names anything missing rather than letting an
incomplete manifest run silently.

## Two tracks

### Track 1 — frozen-corpus model comparison (the primary comparison)

Reuses `evaluation/ab.py` as built, not a parallel implementation: one
corpus is retrieved once, frozen (`EvidenceCorpus`), and `compare()`
replays only synthesis, citation resolution and verification against it
per arm — so both arms see byte-identical evidence by construction, not
by a promise. `validate_for_replay()` already refuses a corpus with
stripped source text before a comparison runs on it.

This is the fair comparison: retrieval variance is removed, so whatever
differs between arms is attributable to the model.

### Track 2 — end-to-end live research (reported separately, never conflated with Track 1)

Same questions, same fixed ceilings, same stop conditions, run fully live
per arm — including retrieval. Source and retrieval variation is
captured and reported explicitly as a confound, not controlled away.
Nothing from Track 2 is averaged together with Track 1; they answer
different questions ("do the models differ on the same evidence" versus
"does the whole pipeline differ in practice, retrieval included").

## The question set

Twelve questions across the required shapes, committed at
[`examples/benchmark/questions.json`](../examples/benchmark/questions.json):
definition, factual/numeric lookup, procedural, multi-part comparison,
relationship, causal (must not overclaim causation the evidence does not
state), time-sensitive, research-methods, ambiguous/underspecified
(expected to refuse), long-tail technical, one adversarial
evidence-shape case (modelled on the documented comparison/authority
tension in `docs/LIMITATIONS.md`), and one question with no defensible
answer (a future fact).

Each entry states, before any run: `expected_answerable`,
`expected_refusal_condition` where refusal is correct,
`forbidden_overclaims`, and a `rubric` a reviewer scores against. These
are the pre-registration; changing them after seeing a result would be
exactly the failure pre-registration exists to prevent.

## Metrics

**Already built and reused as-is** (`evaluation/evaluators.py`):
citation integrity, evidence integrity, citation coverage, claim support
rate, partial support rate, quote fidelity, quote drift, source
diversity, evidence coverage, unused-source rate. `ArmResult` already
carries latency, LLM calls, tokens and cost per arm.

**Not automated, and not pretended to be:** whether a published claim
is an "overclaim" against its own question's `forbidden_overclaims`
list, and whether a refusal was the *correct* refusal for the right
reason, are judgement calls a rubric supports a human making — not
something this protocol claims to score mechanically. Blinded human
review is where these are actually decided.

**Not built in this phase:** a live-mode-specific "retrieval source
overlap" metric for Track 2. Named in scope, not implemented, because
building it against live data that does not exist yet is premature.

## Blinded review

`evaluation/blinding.py`'s `blind()` replaces every known model/provider
identifier in an arm's rendered markdown with an opaque per-arm label
before a human sees it, and reports how many replacements were made — a
benchmark run where every question redacted zero identifiers is a signal
to check the identifier list, not an assumption of a clean run.
**Stated limitation, not hidden:** this catches identifying *strings*,
not a model describing its own architecture in other words. Two
independent reviewers, with disagreement recorded rather than averaged
away, is a human-process requirement this document states but cannot
itself execute.

## Harness validation

What this phase proves, each with an automated test and no provider call:

| Requirement | How it is proven |
|---|---|
| Both arms receive byte-identical frozen evidence | `ab.py`'s `compare()` passes one `EvidenceCorpus` object to every arm by construction; a test confirms this |
| No source text is stripped | `validate_for_replay()` / `DegradedCorpusError`, already built; a test confirms a stripped corpus is refused |
| Blinded output cannot reveal provider identity | `blind()` + `contains_no_identifier()`, tested against real model/provider strings |
| A broken metric fails, not passes quietly | a test asserting a metric that cannot compute raises rather than returning a placeholder value |
| Cloud spend caps stop dispatch before an extra provider call | already enforced by `UsageTracker.reserve_provider_request`, tested elsewhere in this repository (`test_budgets.py`, `test_provider_accounting.py`); this phase adds nothing new here, it points at what already exists rather than re-proving it |
| No secret enters a benchmark artifact | reuses the pattern `web/recordings.py`'s `assert_no_secrets` already applies to recorded runs |

## Phase B — the budget request, not an action

**"48 total runs" means Track 1 only, unambiguously.** 12 questions × 2
repetitions × **2 Track-1 arms** (local, cloud) over a frozen evidence
corpus = **48 total arm-runs**. This does not include the 12 corpus
freezes needed to produce those corpora (retrieval + extraction, run
once per question, priced separately and accounted in the same spend
ceiling) and it does not include any Track 2 (live end-to-end) run at
all -- Track 2 is out of scope for this authorization and requires its
own, separately authorized budget. An earlier draft of this document
left "48 total runs" ambiguous between the two tracks; this paragraph is
the fix, not a restatement of something that was already clear.

Local runs are sequential, to avoid Ollama contention. Arm order is
counterbalanced per question. No retries except a documented
infrastructure failure *before* a provider request is made, and a retry
never creates an additional billed run beyond the 48 preregistered ones.
Cloud runs stop immediately on a spend ceiling, a provider outage, or a
benchmark-integrity failure (any row in the table above failing during
the run, including a corpus-hash mismatch or a run exceeding its
timeout).

Before any cloud run: the exact run count, worst-case cloud cost, search
credits, expected wall-clock time, and this stopping rule go to the
project owner for explicit approval. This document does not grant that
approval. No question above has been run, no provider has been called,
and no cost has been incurred in writing it.

## What a result from this would not prove

Stated now, before any result exists, so it cannot be relaxed after
seeing one favourable to either arm: 48 runs on 12 questions is not a
general claim that one provider or model is better for all research
questions. Track 1's comparison holds retrieval fixed, which is the
right question for "do the models differ" and the wrong one for "which
should I use" if retrieval quality differs between modes in practice —
that is what Track 2 is for, reported separately, never pooled with
Track 1.
