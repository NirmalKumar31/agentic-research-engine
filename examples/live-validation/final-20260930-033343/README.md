# Hosted run — the decision gate

One live run, one attempt, no retries. The owner set the rule before it
was taken: two or more relevant supported findings would tag the release
and end feature work; zero or one would stop the patching and reposition
live research as an experimental, fail-closed integration.

**It published one. The second branch was taken.**

| | |
| --- | --- |
| Service | `agentic-research-engine-live` — the public demo |
| Commit | `aeb4b07f` |
| Duration | 142.0s engine, inside the 240s ceiling |
| Claims | 5 generated, **1 published** |
| Errors recorded | 0 |
| Cost | $0.006774 OpenAI, 6 Tavily credits |
| Provider requests | 14 of a 30 ceiling |

## What published

> **An LLM is a transformer-based neural network.**
>
> *"A large language model (LLM) is a transformer-based neural network
> trained on massive text corpora to predict…"* — exact-normalised
> match, resolving to a live URL.

Read against the question, it is an answer: it locates one subject
inside the other, which is what "how does X differ from Y" means when Y
is a category containing X. The report carries no "did not answer the
question" limitation.

## What was withheld, and why each was right

| Slot | Refused by |
| --- | --- |
| `direct_contrast` | atomicity — the claim asserted two things |
| `dimension` | irrelevant — described neural networks without contrasting them |
| `dimension` | irrelevant — described how they are trained, without contrast |
| `dimension` | modality guard — every cited quote failed it |

Each refusal was inspected and each was correct.

## What the two fixes under test did

Both behaved as designed, and neither moved the published count.

| | unbounded run | this run |
| --- | --- | --- |
| Claims generated | 13 | **5** |
| `direct_contrast` attempted | no | **yes** |
| "did not answer the question" | — | **absent** |
| Published | 0 | 1 |

The contract bound held the report to five claims where the unbounded
run wrote thirteen. The contrast was attempted and refused on
atomicity — the structural tension recorded in the v1.6.0 review, where
a contrast asserts two things by nature.

## Why this run ended the work rather than prompting another fix

The bottleneck is measured and it is not the verification machinery.
Across 24 generated claims in earlier runs, **58% were refused on
synthesis quality and 17% on evidence**. Another prompt change would
have been a fourth attempt at the same layer, unmeasurable without
another paid run, and the gate existed precisely to stop that.

## Verifying it

```
cd examples/live-validation/final-20260930-033343
shasum -a 256 -c checksums.sha256

python examples/live-validation/tools/acceptance.py build \
  --run-dir examples/live-validation/final-20260930-033343
```

## Reading the numbers

**Source quality is a heuristic** from domain type and structure, not a
measure of truth. **Cost is per provider, never totalled**: Tavily was a
free-tier key, and Hugging Face bills per hour of endpoint uptime rather
than per request.
