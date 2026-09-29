# Manual review

## Verdict

**Verified.** Both fixes that landed after the v1.2.0 acceptance
capture now work on a deployment, and the run stayed inside every
budget. One new defect was found and fixed; one path remains unproven
in the cloud and is named below rather than glossed.

## The case that mattered

v1.2.0 was asked how a large language model differs from a neural
network, found that one is a subset of the other, published it, and
then reported that it had not answered the question. That was the
second finding of its review.

The same question, on the same deployment code today:

- *"An LLM is a type of neural network that specifically uses
  transformer…"* — declared `relationship`, judged relevant,
  **published**.
- *"The evidence distinguishes LLMs as a specific class built on
  n…"* — declared `direct_contrast`, withheld at entailment **0.007**.

The claim that would have filled the contrast slot directly was refused
by its own evidence. The core requirement was discharged by the
alternative instead, and the report carries no claim of failure. That is
the fix working, and it worked on the case it was written for without
being handed it.

Worth noting what did *not* happen: the fix did not make the gate
permissive. Four of six claims were still withheld, by four different
mechanisms — two on entailment, one by the relevance judgement, one by
the structural check refusing a claim that described a single subject
under a contrast slot.

## The defect this run found

The interface had to be pointed at the run's own payload to see it.
`AnswerContract.to_dict()` dropped `satisfied_by`, so nothing outside
the engine could tell that a core slot had been discharged by an
alternative. The page recomputes coverage from the published claims —
deliberately, so it cannot drift from a stale field — and with the
alternative invisible it concluded the question was unanswered. It
would have printed that directly above a report whose own limitations
said the opposite.

A page contradicting the report beneath it is worse than either verdict
alone, because a reader cannot tell which to believe. Fixed, with the
run's payload kept as sent.

This is the second time a v1.2.0 record has been unreadable for the same
underlying reason: a decision was made inside the engine and the thing
that explained it did not travel. The first was `answer_slot`.

## Not verified here

**The missing-slot path.** `gpt-6-luna` declared a slot on every claim,
so the behaviour that had made local mode publish nothing was never
reached. Two local runs on `qwen3:4b` and a set of unit tests cover it.
This run is silent on it, and a reader should not take "the release is
verified" to include it.

**Quality.** Two published claims on one question. The adversarial set
is the measurement; this is a deployment check.

## Budgets

| | Used | Ceiling |
| --- | --- | --- |
| Wall clock | 155.2s | 240s |
| Provider requests | 12 | 30 |
| OpenAI cost | $0.008523 | $0.05 reserved |
| Search credits | 6 | 8 |

Consistent with the v1.2.0 run at $0.009466 and 14 requests, on the
same question and the same ceilings.

## Quota

The demo reports `quota_namespace: ""`, distinct from the `"rc"` the
v1.2.0 candidate used in the same store. Both values were read from
running processes rather than from a dashboard, which is the
verification the v1.2.0 review had to leave open.
