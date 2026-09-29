# Manual review

## Verdict

**The change under test made the output worse, and the run says so.**

This artifact exists because a prediction was written down before the
run and the run disagreed with it. Keeping it is the point.

## Nothing was published

Zero claims reached the reader, from thirteen generated. The previous
run on the same question generated five and published one.

## The prediction, and what happened

The claim cap divided the remaining call budget by one call per claim.
That was correct when entailment was a generative call. The NLI
classifier is not a model call, and the relevance judgement and
wording repair are each batched over the whole report, so an extra
claim costs nothing. Removing the cap should therefore have cost
nothing and gained claims.

| | previous | this run |
| --- | --- | --- |
| Generated | 5 | 13 |
| Published | 1 | 0 |

The cap was binding — that part was right. But it was doing a second
job: keeping the report focused. Unbounded, the synthesiser wrote
thin claims until the evidence ran out.

I removed a spend justification and took a quality bound with it. That
is a reasoning error, not an unlucky draw, and the fix is to reinstate
the bound with an honest justification rather than to restore the old
arithmetic.

## The deeper finding

The critic contradicted itself between runs on the same answer:

> previous: *"LLMs are built upon deep neural networks."* — relevant,
> published.
>
> this run: *"An LLM is a neural network."* — irrelevant, "states the
> relationship but does not explain how an LLM differs from a neural
> network."

Both declared the `relationship` slot. That slot exists because when a
question asks how two things differ and one is a kind of the other,
there is no contrast to draw and saying so *is* the answer. The judge
refused a claim for failing a test the contract had already excused it
from.

The cause is that the judge was asked its own question — "does this
help answer?" — while being shown a list of parts it was not asked
about. It applied its own notion of answering and contradicted the
contract the report is scored against.

Both were changed after this capture:

- the claim bound is now the contract's, two claims per required part;
- the judge is asked whether a claim fills one of the listed parts.

The second is a tightening, not a loosening: the judge may no longer
freelance, and a claim filling no listed part still fails.

## Budgets

| | Used | Ceiling |
| --- | --- | --- |
| Wall clock | 187.5s | 240s |
| Provider requests | 14 | 30 |
| OpenAI cost | $0.010072 | $0.05 reserved |
| Search credits | 6 | 8 |

## Honest summary

Seven hosted runs now. The verification machinery has refused
correctly in every case examined, including here — thirteen thin
claims were refused for thirteen stated reasons. What produced them
was a change I made, and the run is committed with the prediction it
falsified.
