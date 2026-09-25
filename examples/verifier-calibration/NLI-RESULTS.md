# NLI verifier calibration

Replacing the generative verifier with a dedicated NLI classifier plus
deterministic guards. Every number here comes from a real run of a real
checkpoint at the pinned revision; the artifacts beside this file hold
the per-pair probabilities the tables are computed from.

## What was being decided

The publication question is binary: may this claim be published? Human
`supported` is the positive class; `partially_supported` and
`unsupported` are both negative, because both withhold.

A claim publishes when **some cited quote passes every deterministic
guard and entails the claim at or above the support threshold**. Quotes
are scored individually and never concatenated.

Selection rule, fixed before the results were seen:

> among thresholds with **zero supported false positives**, take the
> highest supported recall.

## Development calibration: 30 cases, 50 claim/quote pairs

These thirty cases shaped three earlier verifier designs, so they are
**development calibration cases, not an independent benchmark**. A good
number here partly reflects the design having been fitted to them. The
adversarial suite below exists because of that.

Nine cases are reviewer-`supported` (the positives); twenty-one are not.

At threshold 0.98:

| model | TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|---|
| DeBERTa-v3-large-mnli-fever-anli-ling-wanli | 5 | **1** | 4 | 20 | 0.83 | 0.56 |
| cross-encoder/nli-deberta-v3-base | 3 | **1** | 6 | 20 | 0.75 | 0.33 |
| cross-encoder/nli-MiniLM2-L6-H768 | 0 | 0 | 9 | 21 | 1.00 | 0.00 |

**No model meets the acceptance rule.** The full sweep from 0.50 to 0.99
is in `nli-calibration.json`. The large model holds FP=1 at every
threshold up to 0.98; the base model holds FP=1 throughout; MiniLM
reaches FP=0 only at 0.96 and above, where recall is also 0, which is
not a verifier.

### The single blocking false positive

One case, `rag-vector-vs-search-removed-13`, is scored as entailed by
**all three** models — 0.998, 0.998 and 0.955 — and no guard fires on it.

> **Quote:** "Quantization (compressing vectors by using fewer bits per
> dimension, like reducing 32-bit floats to 8-bit integers) reduces
> memory costs with minimal recall impact. Reducing 32-bit floats to
> 8-bit integers cuts memory 75% while maintaining high accuracy."
>
> **Claim:** "Quantization reduces memory usage by 75% while maintaining
> high recall accuracy by compressing vectors from 32-bit floats to
> 8-bit integers."
>
> **Reviewer label:** partially supported

The overreach is "minimal recall impact" plus "high accuracy" becoming
"high **recall** accuracy". Every literal matches, the modality matches,
there is no ranking and no causal language, so the guards correctly
abstain — catching it needs to know that an attribute stated of
*accuracy* was reattached to *recall accuracy*, which is the sentence
parsing the guards deliberately do not attempt.

Two things worth stating about it. It is the same transformation class
as the stress case `intensity-minimal-to-high`, which every model here
does refuse in isolation — so the label is not in doubt. And it is a
**compound claim**: three propositions in one sentence, of which two are
fully supported. Scored as separate atomic claims, the memory and
bit-width propositions would publish and the recall proposition would be
judged against "minimal recall impact" on its own.

## Adversarial stress suite: 36 synthetic cases

Written from the transformation classes rather than from observed
failures, and sharing no claim, quote or subject with the development
cases — so unlike the table above, this is not fitted. 23 cases must be
withheld, 13 must publish. Categories: modality, scope, numeric,
ranking, causal, exclusivity, intensity, attribution, temporal,
negation, comparison, irrelevance.

| model | correct | unsafe publishes | over-withheld |
|---|---|---|---|
| DeBERTa-v3-large | 35/36 | **0** | 1 |
| deberta-v3-base | 32/36 | 3 | 1 |
| MiniLM2-L6-H768 | 31/36 | 3 | 2 |

Both smaller models publish the same three overclaims: generalising a
single study to a general finding, stripping "according to the vendor's
own documentation", and dropping "as of the 2021 release". Each scores
above 0.98 entailment, so no threshold separates them.

The large model's one miss is over-withholding: "requires 64 GB" →
"may require 64 GB" scores 0.778. That withholds a true claim, which the
design accepts.

## Runtime and memory

Measured on this machine, CPU only, 50 pairs, batch size 8.

| model | artifact | RSS after load | peak RSS | load | 50 pairs |
|---|---|---|---|---|---|
| DeBERTa-v3-large | 881 MB | 1149 MB | 1384 MB | 6.1 s | 53.7 s |
| deberta-v3-base | 749 MB | 883 MB | 1142 MB | 5.1 s | 2.2 s |
| MiniLM2-L6-H768 | 331 MB | 552 MB | 675 MB | 4.5 s | 0.7 s |

None fits the 512 MB free tier in-process, including MiniLM — its peak
RSS is 675 MB, which is why file size is not a substitute for measuring.
Since MiniLM also fails the semantic gate, the INT8 ONNX route was not
pursued; the verifier instead has a remote mode that calls a hosted
endpoint and fails closed on timeout, 429, 5xx, malformed body and
network error.

## Label ordering

The three checkpoints do not agree on class order:

- `DeBERTa-v3-large-…`: entailment 0, neutral 1, contradiction 2
- both cross-encoders: **contradiction 0**, entailment 1, neutral 2

The adapter reads `id2label` from each checkpoint. Hardcoding index 0 as
entailment would have inverted the gate for two of the three models —
publishing precisely the claims that should be withheld — while still
producing plausible-looking numbers.

## Reproducing

```bash
python examples/verifier-calibration/nli_calibrate.py   # scores, per model
python examples/verifier-calibration/nli_sweep.py       # sweep + selection
python examples/verifier-calibration/nli_stress.py 0.98 # adversarial suite
```

The sweep reads only the stored score artifacts, so the tables can be
recomputed without downloading any weights.
