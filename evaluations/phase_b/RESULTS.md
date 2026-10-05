# Benchmark Phase B results (Track 1: frozen-corpus comparison)

**This is a carefully controlled pilot, not a pristine confirmatory preregistration.**
The abort policy used while freezing corpora changed mid-run (see "Protocol
deviations" below). That change does not affect the frozen-corpus
comparison's validity -- both arms still saw byte-identical evidence per
question -- but it means this run deviated from its own written protocol
once, and that is disclosed here rather than smoothed over.

- Manifest: [`manifest.json`](manifest.json) -- engine commit `a451aa3ffef337abccd086c1a89e53a8fcde65a7` (see "Execution provenance" for what that commit does and does not pin)
- Cloud model: `gpt-6-luna`; Ollama model: `qwen3:4b` (digest, re-verified in full post-hoc: `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7` -- the manifest itself recorded only the first 12 hex characters; see "Execution provenance")
- NLI: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` @ `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7` (local checkpoint, this run -- not the remote endpoint this machine otherwise defaults to)
- Timeout policy: 120s wall-clock per arm-run (`ARM_TIMEOUT_SECONDS` in `scripts/run_phase_b.py`), enforced via `asyncio.timeout` around the graph invocation in `ab.py`'s `run_arm`
- **48 of 48 preregistered arm-runs were attempted and recorded** (12 questions x 2 repetitions x 2 arms); **43 produced output, 5 timed out** (all on the local arm). "48 completed" would overstate this -- see "Failures" and "Operational reliability" below. Verified 1:1 against `examples/benchmark/questions.json` with no missing or extra records.
- **24 cloud arm-runs**, comprising **48 logical cloud LLM calls** (2 per arm-run: the synthesizer call plus a structured-output repair/verification call), **101,202 input tokens, 33,946 output tokens** (cloud only; exact, summed from `runs/*.json`). The 24 local arm-runs make no cloud calls and record $0 cost by construction.
- **Known/recorded cloud cost: $0.027093**, against the $10.00 ceiling (full ledger: [`spend_ledger.json`](spend_ledger.json)). This is *not* represented as an exact total: every one of the 24 cloud runs recorded `cost_is_complete: false` (the pricing table does not cover every cost component OpenAI may bill), so the true cost may be marginally higher than this figure, though it is bounded by the enforced per-call ceiling (`max_cloud_cost_usd`), which gates on *projected* cost before dispatch regardless of whether the eventual actual figure is complete. Tavily search-credit cost during corpus freezing is **unmeasured** -- not priced anywhere in this repository -- and is not included in any total here. The $10 ceiling was therefore enforced against *known* cloud cost, not against every metered external service this run touched.
- A separate, one-time pre-study smoke test (one throwaway question, not part of the 48, not in any table below) made one real cloud call and cost **$0.000426** (`cost_is_complete` was `true` for that single call). Not included in the $0.027093 figure above.
- Wall-clock elapsed: 116.8 minutes (arm-run phase only; corpus freezing beforehand took considerably longer and was interrupted twice by infrastructure -- see "Protocol deviations").
- **AI scoring complete; disagreements intentionally unresolved.** Two independent Codex (AI) sessions scored every candidate -- not the human reviewers `docs/BENCHMARK-PROTOCOL.md` specifies, a stated protocol deviation. Reconciliation found 6 disagreements (2 score, 4 harmful-claim-flag); all 6 remain unadjudicated by design -- see "Blinded review results" for which ones, why, and what the results do and do not support claiming.

**Statistical caveat, stated once and binding throughout:** each cell below aggregates n=2 repetitions. No significance test is computed or implied; a difference between arms here is a measured observation at n=2, not a generalizable claim of model superiority, factual correctness, or broad applicability beyond these 12 questions.

## Protocol deviations

Stated prominently, not as a footnote, per this document's own standard for
reporting what happened:

1. **The freeze-time abort rule changed mid-run.** `docs/BENCHMARK-PROTOCOL.md`
   and the original runner treated *any* source-fetch problem during corpus
   freezing as fatal. After Q1-Q5 froze cleanly, Q6 hit a single failed
   source fetch (of 5) and the run aborted. The rule was changed, mid-study,
   to only abort when a corpus has zero citable evidence or zero
   sub-questions -- a single degraded source among several is now a logged
   warning, not a hard stop. Q6 and Q9 froze under the revised rule; Q1-Q5,
   Q7-Q8, Q10-Q12 never hit the condition at all, so the rule change had no
   effect on them either way. **This does not affect Track 1's core validity**
   -- both arms for every question still see one byte-identical frozen
   corpus, which is the property the comparison depends on -- but changing a
   stopping rule after seeing a failure, mid-study, is a deviation from
   strict preregistration and is disclosed as one.
2. **Two infrastructure interruptions during corpus freezing**, unrelated to
   the comparison logic: a connection drop that also killed the local Ollama
   server (restarted; freezing resumed from the last completed question),
   and the abort-rule change itself (item 1). Neither re-ran, re-froze, or
   re-paid for anything already completed -- `scripts/run_phase_b.py` checks
   disk for an already-frozen corpus or an already-completed arm-run before
   redoing either.
3. **A real blinding defect was found and fixed after this data was first
   committed, before any human review occurred.** The first version of the
   blinding step called `blind(arm.upper(), ...)`, which replaced every
   redacted identifier with the literal string `[LOCAL]` or `[CLOUD]` --
   directly naming the arm it was supposed to hide. This was caught before
   any blinded review happened (none had), fixed to a single constant,
   arm-neutral replacement token, and all 43 blinded outputs were
   regenerated from the untouched raw markdown in `runs/*.json` -- no
   provider was called again to do this. See "Execution provenance" for the
   before/after detail.
4. **Blinded review was performed by two AI reviewers, not two human
   reviewers.** `docs/BENCHMARK-PROTOCOL.md` explicitly requires human
   judgment for this step. Both reviews here were independent Codex
   sessions instead, authorized by the project owner. The reviewing model
   shares a provider (OpenAI) with the cloud arm's synthesizer
   (`gpt-6-luna`), which is a direct, disclosed limitation on how the
   review's findings should be read -- see "Blinded review results."

## Prominent limitations (read before the table)

These are not footnotes: they bound what this data can honestly be used to claim.

1. **Two corpora have one unfetchable source each.** Q6-causal (1 of 5 sources) and Q9-ambiguous (1 of 10 sources) each had one source fail to fetch live (paywall/block/timeout) at freeze time. Citable evidence still existed from the remaining sources in each case (16 and 52 citable items respectively), so freezing continued rather than aborting -- but both corpora are verifiably *not* at full source strength. Detail in "Limitations" below.
2. **Blinding redacted zero identifiers on 44 of 48 outputs.** The 4 exceptions are *all four* of Q2-numeric-lookup's runs (8, 13, 9, and 8 redactions respectively) -- expected, since Q2 literally asks about a named OpenAI model ("GPT-4 Turbo"), so its answer content legitimately contains provider/model strings on the blinding list. The other 44 outputs never mentioned an identifier at all.
3. **Blinded review was performed by two independent AI reviewers (Codex), not human reviewers.** This is a deviation from `docs/BENCHMARK-PROTOCOL.md`'s explicit requirement for human judgment, and the reviewing AI shares a provider with the cloud arm -- see "Blinded review results" below for the full disclosure and what it does and does not license claiming.

## Per-question, per-arm raw results

| Question | Rep | Arm | ok | timed_out | duration_s | cost_usd | evidence_integrity | citation_integrity | citation_coverage | claim_support | partial_support | quote_fidelity | quote_drift | evidence_coverage | source_diversity | contradiction_auditability | duplicate_avoidance | unused_source_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1-definition | 1 | local | True | False | 114.6 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.50 | 0.07 | 1.00 | 0.00 | 1.00 | 0.80 | n/a | n/a | 0.40 |
| Q1-definition | 1 | cloud | True | False | 23.3 | 0.0010 | 1.00 | 1.00 | 1.00 | 0.50 | 0.00 | 1.00 | 0.00 | 1.00 | 0.80 | n/a | n/a | 0.80 |
| Q1-definition | 2 | local | True | False | 102.1 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.50 | 0.14 | 1.00 | 0.00 | 1.00 | 0.80 | n/a | n/a | 0.40 |
| Q1-definition | 2 | cloud | True | False | 25.0 | 0.0011 | 1.00 | 1.00 | 1.00 | 0.33 | 0.17 | 1.00 | 0.00 | 1.00 | 0.80 | n/a | n/a | 0.80 |
| Q2-numeric-lookup | 1 | cloud | True | False | 13.2 | 0.0010 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.40 | n/a | n/a | 0.80 |
| Q2-numeric-lookup | 1 | local | True | False | 117.7 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.00 | 0.17 | 1.00 | 0.00 | 1.00 | 0.40 | n/a | n/a | 1.00 |
| Q2-numeric-lookup | 2 | cloud | True | False | 9.6 | 0.0008 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.40 | n/a | n/a | 0.80 |
| Q2-numeric-lookup | 2 | local | True | False | 79.5 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.25 | 0.25 | 1.00 | 0.00 | 1.00 | 0.40 | n/a | n/a | 0.80 |
| Q3-procedural | 1 | local | True | False | 100.7 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.14 | 0.00 | 0.90 | 0.05 | 1.00 | 0.20 | n/a | n/a | 0.80 |
| Q3-procedural | 1 | cloud | True | False | 22.9 | 0.0011 | 1.00 | 1.00 | 1.00 | 0.50 | 0.25 | 0.90 | 0.05 | 1.00 | 0.20 | n/a | n/a | 0.80 |
| Q3-procedural | 2 | local | True | False | 84.7 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.14 | 0.00 | 0.90 | 0.05 | 1.00 | 0.20 | n/a | n/a | 0.80 |
| Q3-procedural | 2 | cloud | True | False | 17.0 | 0.0009 | 1.00 | 1.00 | 1.00 | 0.80 | 0.00 | 0.90 | 0.05 | 1.00 | 0.20 | n/a | n/a | 0.40 |
| Q4-comparison | 1 | cloud | True | False | 27.3 | 0.0013 | 1.00 | 1.00 | 1.00 | 0.67 | 0.17 | 0.91 | 0.04 | 1.00 | 0.80 | n/a | n/a | 0.80 |
| Q4-comparison | 1 | local | False | True | 120.0 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Q4-comparison | 2 | cloud | True | False | 24.3 | 0.0012 | 1.00 | 1.00 | 1.00 | 0.67 | 0.17 | 0.91 | 0.04 | 1.00 | 0.80 | n/a | n/a | 0.80 |
| Q4-comparison | 2 | local | True | False | 87.0 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.62 | 0.12 | 0.91 | 0.04 | 1.00 | 0.80 | n/a | n/a | 0.60 |
| Q5-relationship | 1 | local | False | True | 120.0 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Q5-relationship | 1 | cloud | True | False | 21.4 | 0.0013 | 1.00 | 1.00 | 1.00 | 0.80 | 0.00 | 0.88 | 0.12 | 1.00 | 0.80 | n/a | n/a | 0.60 |
| Q5-relationship | 2 | local | False | True | 120.0 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Q5-relationship | 2 | cloud | True | False | 28.5 | 0.0015 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.88 | 0.12 | 1.00 | 0.80 | n/a | n/a | 0.40 |
| Q6-causal | 1 | cloud | True | False | 22.1 | 0.0011 | 1.00 | 1.00 | 1.00 | 0.67 | 0.00 | 1.00 | 0.00 | 1.00 | 0.75 | n/a | n/a | 0.50 |
| Q6-causal | 1 | local | True | False | 117.1 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.75 | n/a | n/a | 1.00 |
| Q6-causal | 2 | cloud | True | False | 23.0 | 0.0013 | 1.00 | 1.00 | 1.00 | 0.75 | 0.00 | 1.00 | 0.00 | 1.00 | 0.75 | n/a | n/a | 0.25 |
| Q6-causal | 2 | local | False | True | 122.8 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Q7-time-sensitive | 1 | local | True | False | 55.1 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.67 | 0.00 | 0.89 | 0.00 | 1.00 | 0.80 | n/a | n/a | 0.60 |
| Q7-time-sensitive | 1 | cloud | True | False | 8.8 | 0.0007 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.89 | 0.00 | 1.00 | 0.80 | n/a | n/a | 0.60 |
| Q7-time-sensitive | 2 | local | True | False | 54.5 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.20 | 0.40 | 0.89 | 0.00 | 1.00 | 0.80 | n/a | n/a | 0.80 |
| Q7-time-sensitive | 2 | cloud | True | False | 17.2 | 0.0012 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.89 | 0.00 | 1.00 | 0.80 | n/a | n/a | 0.60 |
| Q8-research-methods | 1 | cloud | True | False | 20.4 | 0.0012 | 1.00 | 1.00 | 1.00 | 0.80 | 0.20 | 0.91 | 0.05 | 1.00 | 0.80 | n/a | n/a | 0.60 |
| Q8-research-methods | 1 | local | True | False | 112.4 | 0.0000 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.91 | 0.05 | 1.00 | 0.80 | n/a | n/a | 0.60 |
| Q8-research-methods | 2 | cloud | True | False | 22.1 | 0.0012 | 1.00 | 1.00 | 1.00 | 0.83 | 0.00 | 0.91 | 0.05 | 1.00 | 0.80 | n/a | n/a | 0.40 |
| Q8-research-methods | 2 | local | True | False | 92.2 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.80 | 0.00 | 0.91 | 0.05 | 1.00 | 0.80 | n/a | n/a | 0.40 |
| Q9-ambiguous | 1 | local | False | True | 120.0 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Q9-ambiguous | 1 | cloud | True | False | 21.2 | 0.0014 | 1.00 | 1.00 | 1.00 | 0.80 | 0.00 | 0.85 | 0.12 | 1.00 | 0.78 | n/a | n/a | 0.78 |
| Q9-ambiguous | 2 | local | True | False | 115.7 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.43 | 0.29 | 0.85 | 0.12 | 1.00 | 0.78 | n/a | n/a | 0.78 |
| Q9-ambiguous | 2 | cloud | True | False | 19.5 | 0.0012 | 1.00 | 1.00 | 1.00 | 0.83 | 0.00 | 0.85 | 0.12 | 1.00 | 0.78 | n/a | n/a | 0.78 |
| Q10-long-tail | 1 | cloud | True | False | 22.8 | 0.0010 | 1.00 | 1.00 | 1.00 | 0.25 | 0.00 | 0.60 | 0.00 | 1.00 | 0.80 | n/a | n/a | 0.80 |
| Q10-long-tail | 1 | local | True | False | 90.5 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.00 | 0.40 | 0.60 | 0.00 | 1.00 | 0.80 | n/a | n/a | 1.00 |
| Q10-long-tail | 2 | cloud | True | False | 21.8 | 0.0010 | 1.00 | 1.00 | 1.00 | 0.50 | 0.00 | 0.60 | 0.00 | 1.00 | 0.80 | n/a | n/a | 0.60 |
| Q10-long-tail | 2 | local | True | False | 76.5 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.25 | 0.00 | 0.60 | 0.00 | 1.00 | 0.80 | n/a | n/a | 0.80 |
| Q11-adversarial-evidence-shape | 1 | local | True | False | 116.7 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.29 | 0.14 | 0.77 | 0.04 | 0.75 | 0.80 | n/a | n/a | 0.80 |
| Q11-adversarial-evidence-shape | 1 | cloud | True | False | 27.5 | 0.0012 | 1.00 | 1.00 | 1.00 | 0.67 | 0.00 | 0.77 | 0.04 | 0.75 | 0.80 | n/a | n/a | 0.80 |
| Q11-adversarial-evidence-shape | 2 | local | True | False | 118.1 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.29 | 0.00 | 0.77 | 0.04 | 0.75 | 0.80 | n/a | n/a | 0.80 |
| Q11-adversarial-evidence-shape | 2 | cloud | True | False | 24.7 | 0.0013 | 1.00 | 1.00 | 1.00 | 0.83 | 0.00 | 0.77 | 0.04 | 0.75 | 0.80 | n/a | n/a | 0.60 |
| Q12-no-defensible-answer | 1 | cloud | True | False | 15.7 | 0.0011 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.95 | 0.05 | 1.00 | 0.80 | n/a | n/a | 0.60 |
| Q12-no-defensible-answer | 1 | local | True | False | 106.9 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.71 | 0.00 | 0.95 | 0.05 | 1.00 | 0.80 | n/a | n/a | 0.60 |
| Q12-no-defensible-answer | 2 | cloud | True | False | 15.0 | 0.0010 | 1.00 | 1.00 | 1.00 | 0.67 | 0.00 | 0.95 | 0.05 | 1.00 | 0.80 | n/a | n/a | 0.60 |
| Q12-no-defensible-answer | 2 | local | True | False | 118.2 | 0.0000 | 1.00 | 1.00 | 1.00 | 0.56 | 0.00 | 0.95 | 0.05 | 1.00 | 0.80 | n/a | n/a | 0.60 |

## Failures

- Q4-comparison rep1 local: arm did not complete within 120s
- Q5-relationship rep1 local: arm did not complete within 120s
- Q5-relationship rep2 local: arm did not complete within 120s
- Q6-causal rep2 local: arm did not complete within 120s
- Q9-ambiguous rep1 local: arm did not complete within 120s

## Four separate questions this data answers -- not one blended verdict

**1. Evaluation-gate integrity (both arms passed, uniformly).** `citation_integrity` and
`evidence_integrity` are 1.00 for both arms on essentially every question that completed.
This says the project's own verification gate enforces the same bar regardless of which
model is behind it -- it is not itself a quality comparison between the models.

**2. Claim-support variation (descriptive, noisy, not a ranking).** `claim_support` and
`partial_support` swing widely per question and arm (0.00 to 1.00) with no consistent
winner across the 12 questions at n=2. This is reported as raw per-cell data in the table
above, on purpose -- averaging it into one number would manufacture a precision neither
arm earned at this sample size. **The same pairing caveat as point 3 applies here too:**
any summary computed from this column must restrict itself to the 19 question-repetitions
where both arms produced output, for the same selection-bias reason given below -- this
document does not compute such a summary itself, but a reader extending this table should
not average cloud's 24 raw values against local's 19 without first checking which
repetitions are actually paired (`evaluations/phase_b/review/reconciliation.csv`
makes this split explicit; see "Blinded review results" below for the completed
human-rubric -- in this case AI-rubric -- version of this same comparison).

**3. Operational reliability and latency (a real, asymmetric difference).** Cloud: 24/24
completed, 0 timeouts, 9-28s per run. Local: 19/24 completed in 55-123s per run; the other
5/24 hit the fixed 120s timeout and produced no output at all. The local arm's failures
cluster on relationship, comparison, causal and ambiguous question shapes -- not on
definition, numeric or procedural ones -- which is itself a more specific finding than
"local is slower." **Selection-bias note:** the 5 cloud outputs whose local counterpart
timed out are not a fair additional sample of "cloud quality" to pool with the other 19 --
they exist only because local failed on that specific repetition. Any arm-vs-arm quality
comparison belongs on the 19 paired repetitions only.

**4. Cost.** 24 cloud arm-runs, 48 logical cloud LLM calls, 101,202 input / 33,946 output
tokens, **known/recorded** cost **$0.027093** (`cost_is_complete=false` on every cloud run --
not an exact total); local calls $0. Tavily search-credit cost is unmeasured. Separately, one
pre-study smoke call cost $0.000426 and is not part of this total (see header).

These four are kept apart deliberately: collapsing them into a single "cloud won" or
"local won" headline would assert something none of them individually support.

## Limitations

- n=2 repetitions per arm per question: no statistical significance claimed or computable.
- Track 1 holds retrieval fixed (frozen corpus); it answers "do the models differ on the same evidence", not "which pipeline is better in practice" (Track 2, not run here).
- Whether a claim is an overclaim against its question's `forbidden_overclaims`, and whether a refusal was the correct one, are not scored mechanically -- the blinded review (see "Blinded review results" below; performed by two AI reviewers, not human) decides those, not this script.
- Tavily search-credit cost during corpus freezing is not priced per-credit anywhere in this repository and is not included in the $ figure above; it is assumed to remain within the account's free tier, consistent with every prior run in this project.
- `blind()` catches identifying strings, not a model describing its own architecture in other words.
- Non-fatal corpus degradation at freeze time (a source failed to fetch live, e.g. paywall/block/timeout; citable evidence still existed from the remaining sources, so freezing continued rather than aborting):
  - Q6-causal: 1 source(s) have no text (S5). Verification treats them as unretrieved, so citation integrity will read 0%. Build the corpus with `agentic-research freeze`, which keeps source text, rather than from outputs/<run>/.
  - Q9-ambiguous: 1 source(s) have no text (S3). Verification treats them as unretrieved, so citation integrity will read 0%. Build the corpus with `agentic-research freeze`, which keeps source text, rather than from outputs/<run>/.
- 5 local-arm run(s) failed outright; see Failures above for which capability was unavailable locally.
- Blinding redaction counts, exact, from `evaluations/phase_b/blinded/*.md`: 44 of 48 files redacted 0 identifiers. The remaining 4 are all four runs of Q2-numeric-lookup: rep1-cloud 8, rep1-local 13, rep2-cloud 9, rep2-local 8. No other question had any redaction.

## Blinded review results

**Review method deviation, disclosed prominently, not buried:**
`docs/BENCHMARK-PROTOCOL.md` requires two independent **human** reviewers.
Both reviews below were instead performed by two separately-run **Codex
(OpenAI) sessions**, each given only its own randomized packet (a copy of
`evaluations/phase_b/review/reviewer_{a,b}/packet/` and
`scores_template.csv`, distributed without the corresponding
`unblinding_key.json`), no shared context with each other, and no access
to either unblinding key before scoring.
This substitutes AI judgment for the human judgment the protocol calls
for -- authorized by the project owner, stated here exactly as what it is.

**Judge-family concern, stated directly:** the cloud arm's synthesizer is
`gpt-6-luna` (OpenAI); the reviewer in both sessions was Codex (also
OpenAI). LLM-as-judge research documents a measurable same-provider-family
preference effect that survives blinding of explicit identity strings.
The finding below -- cloud scoring higher than local on every dimension,
agreed by both reviewers -- should be read with this directly in mind: it
is evidence from two blinded-to-identity AI raters, not proof of quality
superiority independent of who is judging.

### Paired quality comparison (19 question-repetitions, both arms completed)

The only fair arm-vs-arm comparison in this document. Both reviewers
scored independently; both numbers are shown, never pooled into one mean.

| Dimension | local: reviewer A | local: reviewer B | cloud: reviewer A | cloud: reviewer B |
|---|---|---|---|---|
| relevance | 2.53 | 2.63 | 4.00 | 4.16 |
| completeness | 2.47 | 2.47 | 3.58 | 3.58 |
| clarity | 3.63 | 3.58 | 4.26 | 4.32 |
| claim_support | 3.47 | 3.26 | 4.42 | 4.05 |
| citation_usefulness | 2.74 | 3.05 | 3.37 | 3.53 |

(n=19 per cell; 1-5 scale.)

### Inter-rater agreement

- **215 dimension-score comparisons** (43 scored outputs x 5 dimensions):
  **2 disagreements at \|diff\| >= 2 (0.93%)**, both the same cell type --
  Q3-procedural, `claim_support`, local arm, both repetitions. Both
  reviewers' written rationales agree the output failed to give actual
  migration steps; they differ on whether `claim_support` should score the
  few claims it did make in isolation (reviewer A: 5) or overall adequacy
  against the question (reviewer B: 3) -- a rubric-interpretation
  ambiguity on this one dimension, not a disagreement about the
  underlying output's quality.
- **`harmful_or_unsupported_claims`: 4 of 43 scored outputs disagreed
  (9.3%)**, clustered on Q10-long-tail (3 of 4) and
  Q11-adversarial-evidence-shape (1 of 4) -- this benchmark's two hardest,
  most judgment-dependent question shapes (an obscure-API sourcing
  standard; a comparison requiring refusal). Representative disagreement:
  on Q10, reviewer A rated citations sufficient while reviewer B held them
  to "exact LangGraph documentation or source" and flagged a GitHub-issue
  citation as insufficient; both rationales are defensible readings of the
  same rubric line.
- **None of these are adjudicated here.** `evaluations/phase_b/review/reconciliation.csv`'s
  `adjudicated_*` columns are left blank, per the no-averaging-away
  requirement, for the project owner to resolve.

### Unpaired cloud-only outputs (5 reps, selection-biased, not pooled above)

Both reviewers' scores pooled (n=10) since only cloud produced output for
these repetitions:

| Dimension | Mean |
|---|---|
| relevance | 3.40 |
| completeness | 3.00 |
| clarity | 4.00 |
| claim_support | 4.40 |
| citation_usefulness | 3.70 |

### Directional observation, explicitly bounded

Both reviewers independently rated cloud higher than local on every one of
the 5 dimensions across the 19 paired repetitions. This is a more
consistent cross-dimension pattern than the automated `claim_support`
metric showed earlier in this document (which swung with no consistent
winner at the per-question level) -- but it is **not** a significance
claim: 19 paired observations are not independent in the statistical
sense (clustered within 12 questions), no test is computed, and the
same-provider-judge-family concern above directly limits what can be read
into this independent of the fact that the rater and the cloud arm share
a provider. Read as: two blinded AI reviewers, working from Codex's own
judgment, agreed cloud's outputs scored higher on this rubric for this
question set -- not as: cloud is the better research model in general.

Raw data: `evaluations/phase_b/review/reconciliation.csv` (per-cell, both
reviewers, disagreement flags, blank adjudication columns),
`reviewer_{a,b}/joined_results.csv` (full per-reviewer join including
rationale text), `_submitted_originals/` (the reviewers' CSVs exactly as
submitted, unmodified).

## Execution provenance

What is cryptographically pinned, what is not, and what was reconciled
after the fact -- stated explicitly so nothing here is mistaken for a
pre-run commitment it was not.

- **`frozen_engine_commit` (`a451aa3ffef337abccd086c1a89e53a8fcde65a7`) pins the
  engine/library code** (`src/agentic_research/`) that produced every number in
  this document. **It does not pin the orchestration script**
  (`scripts/run_phase_b.py`): that script did not exist as a committed artifact
  until after this run finished, and it was edited once *during* corpus
  freezing (the abort-rule change in "Protocol deviations"). The version of the
  script preserved in this repository -- sha256
  `f8afbf5f931a229a1f5e7a637a8623881c423a306314b5cf51c8e6c2bc03db2a` -- is the
  final, post-fix version (it also includes the later blinding-label fix,
  which changed nothing about the arm-dispatch, timeout, or budget logic that
  actually produced the 48 runs). Anyone re-running this exact script today
  would get the post-fix behavior throughout, not the sequence of behaviors
  this run actually went through.
- **`environment.json` recorded `dirty: true`.** At execution time the only
  uncommitted change was `scripts/run_phase_b.py` itself (the engine code at
  `a451aa3` was already merged and clean) -- consistent with the point above,
  not a separate concern.
- **Version inconsistency, root cause identified:** `environment.json`'s
  `packages.agentic-research-engine` reports `1.2.0`, while
  `provenance.engine_version` reports `1.15.0`. These come from different
  sources: `engine_version` reads the source-level `__version__` (authoritative,
  matches `pyproject.toml` and the `a451aa3` tag history), while the `packages`
  block reads `importlib.metadata.version(...)`, which reflects this machine's
  installed `.dist-info` -- stale from whenever `pip install -e .` was last run,
  not refreshed after many subsequent version bumps. **`1.15.0` is the correct
  engine version for this run; `1.2.0` is a local environment-metadata staleness
  bug, not a different build.**
- **Ollama digest, independently re-verified:** the manifest recorded only
  `359d7dd4bcda` (the 12-character prefix `ollama list` prints). The full
  64-character digest, re-queried from the same running Ollama instance via
  `/api/tags` after the fact, is
  `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`. This is
  presented as corroborating evidence obtained after the run, not as something
  that was pinned before or during it -- the manifest's own pre-run pin remains
  the truncated prefix.
- **Artifact hashes, for independent verification of what this PR actually
  contains** (sha256):
  - `scripts/run_phase_b.py`: `f8afbf5f931a229a1f5e7a637a8623881c423a306314b5cf51c8e6c2bc03db2a`
  - `manifest.json`: `7fb6fb08771fc9f4a8bfc2162558f1727815c2d6095293ac0fdd0de332e6dcac`
  - `raw_results.json`: `1c492a6638550366f81bffaaa304bd1fc2b55a75a4d35b2a77158c9c46344f18`
  - `spend_ledger.json`: `8f3ef510d661c8a2401df39a2e4f688fb30f096d8e75a5b8e5d76340e5afa6f9`

## Redistribution: corpora are published redacted, not in full

The 12 frozen corpora contain verbatim scraped third-party web text
(`SourceDocument.text`, `EvidenceItem.quote`) whose redistribution rights
this project has not established. `evaluations/phase_b/corpora_public/` is
what is actually committed here: the same 12 corpora with that verbatim
text replaced by a length-only placeholder, everything else (URLs,
titles, domains, `content_hash`, timestamps, quality scores, the model's
own `claim` paraphrases, relevance, citability) unchanged. See
`corpora_public/README.md` for the reproduction procedure. **These
redacted files will not reproduce `corpus_hash()` against the values in
`manifest.json`** -- that hash was computed over the original full-text
corpus, which is kept locally for the record and is not redistributed
in this repository.

**The same redaction was applied to 3 of the 48 run outputs.** Q2-numeric-lookup,
Q6-causal and Q10-long-tail's `rep1 local` runs hit a "no claim passed
verification" fallback path (`report.py`'s "Source excerpts" section) that
renders verbatim, often multi-sentence, source quotations directly into the
report -- the same redistribution concern as the corpora, just 3 files
instead of 12. `scripts/redact_phase_b_quote_excerpts.py` strips the quote
text from both `runs/*.json`'s `markdown` field and the corresponding
`blinded/*.md` file, in place, leaving every measurement (`ok`, `timed_out`,
`duration_s`, cost, tokens, metrics) and every other section of the report
untouched. Unredacted originals are kept locally, not committed. This
repository's git history was also rewritten (see below) to remove the
original unredacted blobs for both the corpora and these 3 files from every
reachable commit on this branch -- not merely untracked going forward.

**Git history on this branch has been rewritten** to remove the original,
full-text blobs (12 corpora, 3 run outputs) from every commit that ever
contained them. `git filter-repo --strip-blobs-with-ids` was tried first and
rejected: it strips GPG signatures from every commit it reprocesses as a
side effect, which cascaded through roughly 125 shared, already-released,
already-tagged commits (e.g. `v1.7.1`) -- changing their hashes even though
their content never changed, and would have desynced this branch from
`main`'s real shared history. A plain text/substring replacement was also
rejected for a related reason: one of the quotes needing removal (GPT-4
Turbo's context window size) is independently quoted, for an unrelated
reason, in that same already-released `v1.7.1` history -- a blanket
text-replace would have silently rewritten it too. Used direct git plumbing
instead (`git commit-tree`): a single new commit built on the exact,
byte-verified tree of the already-sanitized final state, parented directly
on the unchanged engine commit `a451aa3`. It has no ancestor that ever
contained the unredacted blobs, so there is no intermediate history for
them to be reachable through, and it cannot touch anything outside itself.
`main`, every tag, and every commit this branch shares with them are
byte-for-byte unchanged -- verified by direct SHA comparison, both locally
and against the actual remote, before force-pushing. Old and new commit
SHAs for this branch are recorded in the PR description.
