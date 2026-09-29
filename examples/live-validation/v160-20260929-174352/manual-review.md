# Manual review

## Verdict

**The engine answers the question, and the answer is checkable.** One
claim, one exact quote, one live URL, and a limitations section that
names what the evidence did not establish.

That is a small report. It is not a broken one, and the difference
matters: the five preceding runs either published nothing or
published claims that did not address what was asked.

## What published

> **LLMs are built upon deep neural networks.** [S5]

Quote: *"At their core, LLMs are built upon deep neural networks,
enabling them to process vast amounts of text and learn complex
patterns."* — johnsnowlabs.com, exact-normalised match.

Read against the question, this is an answer: it locates one subject
inside the other, which is what "how does X differ from Y" means when
Y is a category containing X. The report says so rather than claiming
to have failed, because `relationship` discharges the comparison's
core slot — the v1.2.0 alternative, working on the case it was
written for.

## What the fix did

v1.6.0 marked the required slot. The v1.5.0 run had written three
`dimension` claims and no contrast, because all three slots were
listed identically.

| | v1.5.0 | v1.6.0 |
| --- | --- | --- |
| `direct_contrast` claims | 0 | **1** |
| Claims generated | 3 | 5 |
| Published | 0 | **1** |

The contrast claim itself was refused, which is the finding below —
but it was *written*, and that is what the change was for.

## What it found

**A contrast is inherently two assertions.** The claim was:

> *LLMs learn to predict token sequences in large text corpora,
> whereas…*

The atomicity guard refused it, and the guard is right: that is two
propositions, and a claim resting on two cannot be verified against
one quote.

So there is a structural tension. The comparison contract asks for a
`direct_contrast`; the atomicity guard refuses compound claims; a
contrast is compound by nature. `direct_contrast` may be
systematically unfillable while atomicity holds.

**I am not changing either.** Loosening atomicity to admit contrasts
reopens the defect audits 1–3 were spent closing: a fused claim
defeats every other guard, because each reasons about "the sentence
that supports this claim" and a compound claim hands it two. The
`relationship` alternative already covers the case, and covered it
here.

What could be done, and is not being done today because it is
unmeasured: let a contrast be expressed as two atomic claims filling
`direct_contrast` jointly. That is a contract-design change and it
deserves its own evidence.

## Budgets

| | Used | Ceiling |
| --- | --- | --- |
| Wall clock | 166.0s | 240s |
| Provider requests | 13 | 30 |
| OpenAI cost | $0.009190 | $0.05 reserved |
| Search credits | 6 | 8 |

## Honest summary

Six hosted runs, six findings, every one upstream of the verification
machinery. The gates have refused correctly in every case examined —
including here, where they refused a contrast claim that genuinely
was two assertions.

One published claim out of five is a thin report, and the limitations
say why: the evidence found was general commentary rather than
material that contrasts the two subjects directly. That is a
retrieval outcome, not a verification failure.
