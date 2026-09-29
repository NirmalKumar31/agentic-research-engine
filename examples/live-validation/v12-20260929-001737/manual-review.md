# Manual review

Read against the capture, the citations and the contract. What the
engine cannot tell you about itself.

## Verdict

**Accept, with two findings.** Every gate v1.2.0 adds behaved as
designed on a question v1.1.x answered badly, and the run stayed
inside every budget. Neither finding is a correctness defect; one is
a provenance gap in the artifact and one is a question about contract
design that is worth deciding before the next release rather than
this one.

## What the run demonstrates

**The relevance gate did the thing it was built for.** Two claims were
entailed by their evidence at 0.996 and 0.990, cited correctly, and
withheld anyway:

- *"Neural networks consist of layers of nodes, with each node
  representing a mathematical function."* — true, sourced, and not an
  answer to how the two differ.
- *"LLMs can typically generate, summarize, translate, and analyze
  text in many contexts."* — same.

This is the exact failure v1.1.x shipped: a live run published five
claims that passed entailment and answered nothing. Support and
relevance are now separate questions, and both are asked.

**The judgement was used, not just the free check.** All five
decisions in `relevance-decisions.json` carry `stage: judged`, so the
batched critic call ran and its verdicts are recorded per claim
rather than collapsed into a reason string.

**The repair refused itself.** One claim failed the atomicity guard
(*"LLMs are a specific, language-focused subset of neural networks"* —
two assertions). The rewrite dropped "language-focused", which the
validator accepted as a weakening rather than an addition; the claim
was then refused again at the relevance gate for the slot it had
declared. Both wordings are in `repairs.json`. Previously this would
have been one log line.

**The engine contradicted its own output, honestly.** It published
three claims and then wrote in its limitations: *"This research did
not answer the question. Nothing published states how large language
model (LLM) and neural network differ; what survived describes them
separately."* A report that publishes claims and still says it did
not answer is the coverage assessment working. Most systems would
have presented three sourced sentences as an answer.

**Provenance holds.** All three published claims resolve to an
`exact_normalized` quote and a live URL. No dangling evidence id.

## Finding 1 — the artifact cannot explain its own repair

`repairs.json` records the rewrite *"LLMs are a specific subset of
neural networks."* as refused with *"still irrelevant: the claim
describes one subject rather than contrasting them, so it cannot fill
the contrast slot"*. That identical sentence appears as a **published
claim** in `report.md`.

There is no contradiction — the two claims declared different
`answer_slot` values, and a sentence that cannot fill
`direct_contrast` can legitimately fill `relationship`. But
`answer_slot` is not in the serialised payload, so a reader has no way
to reach that conclusion from the artifact. They see a sentence
rejected and published in the same run.

The slot drives the relevance decision and is the one input to it that
is not recorded. It should be added to the payload. It cannot be
recovered for this run: the daily allowance was one, and re-deriving
from the captured bytes cannot produce a field that was never sent.

## Finding 2 — `direct_contrast` may be the wrong core slot here

The contract classified this as a comparison and made
`direct_contrast` its only core slot. For *"how does X differ from
Y"* where Y is a **superset** of X, the honest answer is a subset
relationship, not a contrast — and that is precisely what the engine
found and published. Coverage then reported the question unanswered
because the core slot was unfilled.

The report is literally correct and a reader would disagree with it.
"An LLM is a neural network, specifically a transformer trained on
text" does answer the question as asked.

Whether `relationship` should also count as core for comparisons is a
design decision with a real cost on the other side: for a genuine
comparison, stating a relationship is not an answer, and making two
slots core weakens the gate there. Recording it rather than changing
frozen semantics mid-acceptance.

## What this run does not show

**Proposition decomposition was never exercised.** `propositions.json`
is empty because no generated claim decomposed into more than one
part. The bundled claim that *did* assert two things was caught by the
atomicity guard, which is a separate mechanism. The per-assertion
entailment path is covered by tests and is **unproven in production**.
Saying otherwise would be inventing evidence.

**Retrieval was mediocre and is the limiting factor.** Six sources,
cited three: a Wikipedia article and a small blog. The highest-quality
source retrieved — an arXiv survey at 0.95 — was never cited. The
gates cannot publish a direct contrast that the evidence does not
contain, so this run measures the gates against weak evidence rather
than measuring the engine at its best.

**One run is not a benchmark.** Three published claims here says
nothing about the next question.

## Budgets

| | Used | Ceiling |
| --- | --- | --- |
| Wall clock | 165.4s | 240s |
| Provider requests | 14 | 30 |
| OpenAI cost | $0.009466 | $0.05 reserved |
| Search credits | 6 | 8 |

The reservation over-estimated by 3.3x, as expected: it is denominated
in a byte bound rather than real tokens.

## Isolation

The candidate and the public demo shared one Key Value store,
separated only by `DEMO_QUOTA_NAMESPACE`. Verified before the run
against `/api/readiness`, which reports `quota_namespace: "rc"` —
the running process, not the dashboard. After the run a second request
was refused `429` at the quota gate before any provider call, so the
one-run daily admission is enforced rather than merely configured.

The public demo was healthy before and after, still on v1.1.1 with its
five-run allowance. Its own used count is not observable from outside,
so isolation rests on the key construction and its tests rather than
on a runtime reading of the demo's counter. That gap closes when
v1.2.0 merges and the demo reports its own namespace.
