# Manual review

## Verdict

**The fix this run tested works. It exposed the next one.**

## Nothing was published

Zero claims reached the reader, from three generated. Stated plainly,
because three claims is a thin sample and a report with an empty
findings section is the outcome this project would rather show than
dress up.

## What the fix did

v1.5.0 changed the synthesiser prompt because two of its three worked
examples for splitting a compound claim began *"The source reports"*
— teaching the model to write claims about the evidence, which the
verifier then refused at 0.007, 0.031 and 0.115.

| Run | Claims about the evidence |
| --- | --- |
| v1.2.0 | 1 |
| v1.4.1 | 2 of 3 |
| **v1.5.0** | **0** |

The model stopped. That is the fix working, measured on the path it
was written for.

## What it exposed

All three claims declared `dimension` — an **optional** slot. None
filled `direct_contrast`, which is what a comparison requires.

| Claim | Slot | Refused by |
| --- | --- | --- |
| *A neural network is a layered machine-learning model…* | `dimension` | relevance: defines, does not contrast |
| *Large language models are described as generative AI models…* | `dimension` | relevance: describes, does not compare |
| *Neural networks can support tasks including computer vision…* | `dimension` | modality guard |

Two refusals are the relevance gate doing exactly its job: a
definition of one subject is not an answer to how two differ. That is
the original v1.1.x failure, caught.

But the model was never told which slot mattered. The prompt listed:

```
- direct_contrast: An explicit statement of how the subjects differ
- dimension: A named dimension along which they differ
- relationship: How the subjects relate…
```

Three options, rendered identically. The contract marks
`direct_contrast` as core and the call site flattened the slots to
`(name, description)` pairs, dropping `core` before the prompt ever
saw it.

This is the sixth time in this project a value has been computed and
then not passed to the thing that needed it — after the durable
quota, `answer_slot`, `satisfied_by`, `SourceIdentity`'s authority,
and the slot descriptions in the relevance prompt. Four of those
passed every test, because the fakes constructed the object correctly
while production did not.

**The fix landed after this capture and is unverified.**

## Budgets

| | Used | Ceiling |
| --- | --- | --- |
| Wall clock | 171.3s | 240s |
| Provider requests | 13 | 30 |
| OpenAI cost | $0.008139 | $0.05 reserved |
| Search credits | 6 | 8 |

## Honest summary

Four hosted runs on this question have published 3, 2, 0 and 0
claims. The verification machinery has been correct in every case
examined — every refusal inspected here was the right refusal. What
has been wrong each time is upstream: what the synthesiser was told.
