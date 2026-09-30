# Does it work as a research engine? Four question shapes, measured

Not run artifacts. A coverage study: several hosted runs compared
against each other rather than each accepted on its own.

## Why this exists

Eight hosted runs had been taken before this, and **every one used the
same question** — *"How does a large language model differ from a
neural network?"* Conclusions about the engine were drawn from them,
including a published statement that its reports are "too thin to use
as research".

That question is a **hypernym comparison**, and it is the engine's
worst case by construction. The contract's core slot for a comparison
is `direct_contrast`; a contrast asserts two things; the atomicity
guard refuses compound claims because every other guard reasons about
"the sentence that supports this claim". So the one shape under test
was the one shape whose core slot is close to unfillable.

Eight of the nine contract shapes had never been exercised on a
deployment.

## What four shapes actually do

| Shape | Question | Published | Cost | Time |
| --- | --- | --- | --- | --- |
| **definition** | What is retrieval-augmented generation? | **2 of 3** | $0.009275 | 119s |
| **numeric** (read as definition) | What is the context window size of GPT-4 Turbo? | **1 of 1** | $0.005784 | 52s |
| **procedural** | How do you fine-tune a language model using LoRA? | 1 of 4 | $0.010065 | 96s |
| **comparison** (hypernym, ×10 runs) | How does an LLM differ from a neural network? | 0–3, mostly 0–1 | ~$0.009 | ~150s |

### The definition run produced a usable answer

> **Retrieval-augmented generation (RAG) is an artificial intelligence
> framework that combines information retrieval systems with generative
> large language models.**
>
> **RAG retrieves relevant external documents from a knowledge source
> and incorporates them into the generation process.**

Both contract slots filled, both quotes exact-normalised, no false
limitation. The single refusal was correct — it asserted three things.

### The numeric run answered in 52 seconds

> **GPT-4 Turbo has a context window of 128,000 tokens.**

Entailed at 0.9975 by a quote that says exactly that. One claim, one
question, correct.

## The comparison fix: mechanism verified, outcome unchanged

`comparison-fixed-20260930-045048/` — the run that tested letting a
comparison be answered by one claim per subject.

**The mechanism works.** A `dimension` claim published with
`judge=False` recorded against it. Nine prior runs could not do that:
the judge vetoed every `dimension` claim for "describing neural
networks, not how LLMs differ", which is the report's question applied
to a single claim.

```
slot=dimension  pub=True  judge=False  stage=judged
```

**The outcome did not improve.** 1 published of 5, and coverage
correctly reports *"This research did not answer the question"* —
because the one published claim speaks about LLMs only, and a
comparison is answered when the claims between them cover both
subjects.

So the remaining blocker is not a gate. It is that the synthesiser
writes one-sided claims for this question, and the gates are right to
refuse to call that an answer.

**Ten runs on this question have published 0, 0, 0, 1, 1, 2, 3, 0, 1,
1.** Five separate fixes were made to the comparison path — the
`relationship` alternative, serialising `answer_slot`, marking the
required slot, a prompt asking the judge to judge against the listed
parts, and finally narrowing the judge in code. Each was a real defect
and each is tested. **None of them moved this number.**

That is the finding, and it is why the work stopped rather than
continuing to a sixth attempt.

## The correction this forces

The statement that the engine's reports are "too thin to use as
research" was **generalised from one question shape, and that shape was
the hardest one**. On definitional and factual-lookup questions it
produces short, correct, cited answers — which is what it was built to
do.

It remains weak on comparisons, for the structural reason above, and on
procedural questions, where three of four claims were refused.

**This is four questions, one each.** It is coverage, not a benchmark,
and it does not license a claim of general correctness. What it does
license is retracting an over-general negative claim that four runs
contradict.

## One finding worth fixing

The numeric run's published claim cites two quotes. One entailed it at
**0.9975** and carried it. The other scored **0.0011** and failed the
guards — it is shown to the reader alongside the one that did the work,
with nothing in the citation list distinguishing them.

The claim is true and properly supported. But a reader clicking the
second citation sees a quote the verifier itself rejected, and cannot
tell that from the interface. The per-evidence scores are in the audit
record; they are not in the presentation.

## Reproducing

```
python examples/live-validation/tools/acceptance.py build \
  --run-dir examples/live-validation/question-shapes/shape-definition-20260930-040817
```

Each directory holds the raw SSE stream, its timing index, and the
files derived from those bytes. Costs are per provider and never
totalled: Tavily was a free-tier key and Hugging Face bills per hour
of endpoint uptime rather than per request.
