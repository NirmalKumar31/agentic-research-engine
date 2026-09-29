# Hosted verification — v1.2.1 on the public demo

One live run through the deployed demo, captured byte for byte, with
everything else derived from those bytes offline.

This run exists for a specific reason. The v1.2.0 acceptance capture
found two defects, both fixed *after* it — so the fixes had never been
through the live pipeline. Unit tests covered them and one was replayed
against the earlier run's real data, but neither had run end to end on
a deployment. This closes that.

**Deployment verification, not a research-quality evaluation.** One run
on one question. It says nothing about how well the engine answers
questions in general.

| | |
| --- | --- |
| Service | `agentic-research-engine-live` — the public demo itself |
| Commit | `1d21b110` |
| Version | 1.2.1 |
| Question | How does a large language model differ from a neural network? |
| Duration | 155.2s engine, inside the 240s ceiling |
| Claims | 6 generated, 6 checked, **2 published**, 4 withheld |
| Cost | $0.008523 OpenAI, 6 Tavily credits |
| Provider requests | 12 of a 30 ceiling |

## What it verified

**`answer_slot` is serialised.** All six judgments carry one. The
v1.2.0 capture carried none, which is why its artifact could not
explain why the same sentence was refused as a repair and published as
a claim.

**A relationship answers a comparison.** The decisive case, and it
arrived on its own:

| Claim | Slot | Outcome |
| --- | --- | --- |
| *An LLM is a type of neural network that specifically uses transformer…* | `relationship` | **published** |
| *The evidence distinguishes LLMs as a specific class built on n…* | `direct_contrast` | withheld, entailment 0.007 |

So the only claim that would have filled `direct_contrast` directly was
refused on its evidence, and the core requirement was discharged by the
alternative. The report carries **no** "did not answer the question"
limitation. On this same question, v1.2.0 reported the opposite.

**All four withholding paths fired**: two below the 0.98 threshold
(0.007 and 0.851), one rejected by the relevance judgement, one by the
structural relevance check.

## What it did not verify

**The missing-slot path.** Every claim this cloud model produced
declared a slot, so the behaviour that had made local mode publish
nothing was never reached here. It is covered by unit tests and by two
local runs, and this run says nothing about it.

**Anything about quality.** Two published claims on one question is not
a measurement.

## What it found

`AnswerContract.to_dict()` dropped `satisfied_by`. A client therefore
could not tell that a core slot had been discharged by an alternative:
the interface recomputed coverage, found `direct_contrast` unfilled,
and would have rendered *"this report does not answer the question"*
directly above a report whose own limitations said otherwise.

Found by pointing the interface's own logic at this run's payload.
Fixed after the capture — `contract.json` here records the payload as
it was actually sent, without `satisfied_by`, because that is what the
run produced.

## Verifying it

```
cd examples/live-validation/v121-20260929-024544
shasum -a 256 -c checksums.sha256
```

```
python examples/live-validation/tools/acceptance.py build \
  --run-dir examples/live-validation/v121-20260929-024544
```

Deterministic, so a file that no longer matches its checksum was edited
rather than derived.

## Reading the numbers

**Source quality is a heuristic** computed from domain type and
structure. It is not a measure of truth.

**Cost is per provider, never totalled.** Tavily was a free-tier key;
Hugging Face bills per hour of endpoint uptime, not per request, so no
per-run figure exists. `known_cost_usd` is marked incomplete because a
response reported a token category this project records no rate for;
the charge uses the full input rate, which cannot understate it.
