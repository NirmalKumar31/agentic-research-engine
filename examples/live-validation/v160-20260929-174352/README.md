# Hosted verification — v1.6.0, a report that answers

One live run through the deployed demo, captured byte for byte. This
is the run that closes a sequence of five.

**Deployment verification, not a research-quality evaluation.** One
published claim is not a measurement of how well the engine answers
questions.

| | |
| --- | --- |
| Service | `agentic-research-engine-live` — the public demo |
| Commit | `7fcce65d` |
| Version | 1.6.0 |
| Duration | 166.0s engine, inside the 240s ceiling |
| Claims | 5 generated, **1 published** |
| Errors recorded | 0 |
| Cost | $0.009190 OpenAI, 6 Tavily credits |
| Provider requests | 13 of a 30 ceiling |

## What it verified

**A contrast claim was written.** Slots claimed: `direct_contrast` 1,
`relationship` 1, `dimension` 3. The first `direct_contrast` across
five hosted runs. The v1.5.0 run wrote three `dimension` claims and
no contrast, because the prompt listed all three slots identically;
marking the required one is what changed.

**The report published, and did not claim failure.** One claim, with
an exact-normalised quote resolving to a live URL:

> **LLMs are built upon deep neural networks.** [S5]
>
> *"At their core, LLMs are built upon deep neural networks, enabling
> them to process vast amounts of text and learn complex patterns."*
> — johnsnowlabs.com

No "did not answer the question" limitation: the `relationship` claim
discharged the core slot, which is the alternative added in v1.2.0
doing its job.

## What it found

**A contrast is not atomic.** The `direct_contrast` claim — *"LLMs
learn to predict token sequences in large text corpora, whereas…"* —
failed the atomicity guard. Correctly: a contrast asserts two things.

So the contract asks for a contrast and the guard refuses compound
claims, which means `direct_contrast` may be systematically
unfillable while atomicity is enforced. The `relationship` alternative
covers it, and did here.

**Recorded, not changed.** Loosening atomicity to admit contrasts
would reopen the defect three audits were spent closing — a fused
claim defeats every other guard, because each reasons about "the
sentence that supports this claim" and a compound claim hands them
two.

## The sequence this closes

| Run | Version | Published | What it found |
| --- | --- | --- | --- |
| v12 | 1.2.0 | 3 of 7 | `answer_slot` not serialised; a relationship should answer a comparison |
| v121 | 1.2.1 | 2 of 6 | `satisfied_by` not serialised |
| v140 | 1.4.0 | — | run died at the coverage critique |
| v141 | 1.4.1 | 0 of 3 | two of three claims were about the evidence |
| v150 | 1.5.0 | 0 of 3 | meta-claims gone; no slot marked required |
| **v160** | **1.6.0** | **1 of 5** | a contrast is not atomic |

Every refusal inspected across all six was the correct refusal. What
was wrong each time was upstream — what the synthesiser was told.

## Verifying it

```
cd examples/live-validation/v160-20260929-174352
shasum -a 256 -c checksums.sha256

python examples/live-validation/tools/acceptance.py build \
  --run-dir examples/live-validation/v160-20260929-174352
```

## Reading the numbers

**Source quality is a heuristic** from domain type and structure, not
a measure of truth. **Cost is per provider, never totalled**: Tavily
was a free-tier key, and Hugging Face bills per hour of endpoint
uptime rather than per request.
