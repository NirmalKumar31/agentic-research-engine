# Hosted run — a prediction that was half wrong

One live run through the deployed demo, captured byte for byte. It
exists because a change needed testing and the test disagreed with the
change.

**Deployment verification, not a research-quality evaluation.**

| | |
| --- | --- |
| Service | `agentic-research-engine-live` — the public demo |
| Commit | `1c1ace31` |
| Version | 1.6.1 |
| Duration | 187.5s engine, inside the 240s ceiling |
| Claims | **13 generated, 0 published** |
| Errors recorded | 0 |
| Cost | $0.010072 OpenAI, 6 Tavily credits |
| Provider requests | 14 of a 30 ceiling |

## The prediction

The claim cap priced a per-claim verification cost that stopped
existing when the NLI classifier replaced the generative verifier.
Removing it should produce more claims, and since roughly a quarter of
claims survive the gates, more published ones.

| | previous run | this run |
| --- | --- | --- |
| Claims generated | 5 | **13** |
| Published | **1** | **0** |

The first half held: the cap was binding. The second did not.

## Nothing was published, so look at a replay instead

Zero claims reached the reader. The three recorded runs served by the
demo's **replay** path show complete reports with resolvable
citations, and cost nothing to explore.

## What it showed

**The cap had a second job nobody had written down.** Unbounded, the
synthesiser wrote thin claims until the evidence ran out, and a larger
batch of thin claims fared worse at the relevance gate than a smaller
batch of considered ones. Removing the spend justification was right.
Removing the cap with it was not.

**The critic contradicted itself across runs:**

| Run | Claim | Verdict |
| --- | --- | --- |
| previous | *LLMs are built upon deep neural networks.* | relevant → published |
| this | *An LLM is a neural network.* | **irrelevant** — "states the relationship but does not explain how an LLM differs" |

Those are the same answer. The `relationship` slot exists precisely to
say that a contrast is unnecessary when one subject is a kind of the
other, and the judge refused a claim for not being one.

## What changed because of it

The claim bound was reinstated as a **quality** bound tied to the
contract — two claims per required part — rather than to the call
budget. And the relevance judge is now asked the contract's question,
*does this fill one of the listed parts*, instead of its own.

**Both landed after this capture and are unverified on the hosted
path.**

## Verifying it

```
cd examples/live-validation/v161-20260929-191831
shasum -a 256 -c checksums.sha256

python examples/live-validation/tools/acceptance.py build \
  --run-dir examples/live-validation/v161-20260929-191831
```

## Reading the numbers

**Source quality is a heuristic** from domain type and structure, not
a measure of truth. **Cost is per provider, never totalled**: Tavily
was a free-tier key, and Hugging Face bills per hour of endpoint
uptime rather than per request.
