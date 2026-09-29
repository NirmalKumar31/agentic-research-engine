# Changelog

Notable changes per release. Dates are UTC.

## v1.2.0 — 2026-09-28

Research quality. The pipeline now knows what question it was asked
and checks its answers against that, rather than only checking that
each sentence follows from a quote.

- **Answer contract.** A question is decomposed into canonical slots
  before retrieval, and refuses rather than guesses: a comparison
  naming fewer than two entities produces an unusable contract with
  no slots at all.
- **Proposition decomposition.** Support is checked per assertion,
  not per sentence. A claim bundling a measured figure with an
  unsupported assertion used to publish at 0.983 because the sentence
  as a whole was close enough to the quote as a whole.
- **Relevance gate.** Support and relevance are separate questions and
  only one was being asked. A live run published five claims that were
  entailed by their evidence and answered nothing. Structure is checked
  free; the judgement is one batched critic call, asked of the critic
  rather than the synthesiser, and withheld rather than guessed when
  it cannot be obtained.
- **Bounded wording repair.** One rewrite attempt for claims refused
  on phrasing alone, validated before re-verification. A rewrite may
  not add a number, introduce a subject, invent causation or
  strengthen a modality, and causal, exclusivity, framing and hedge
  failures are never eligible — rephrasing those is laundering.
- **Answer coverage.** A report that publishes claims and answers none
  of the contract's core slots now says so in its limitations.
- A comparison is answered by a contrast **or** by a relationship.
  Hosted acceptance asked how a large language model differs from a
  neural network, found and published that one is a subset of the
  other, and then reported that it had not answered -- because a
  subset is not a contrast. It was the answer. When one subject is a
  category containing the other there is no contrast to find, and
  demanding one makes the engine wrong about itself.

  Verified by replaying the captured run's own contract and published
  slots through the new coverage, not by a second live run: the
  change post-dates the acceptance capture and the deployment admits
  one run a day. The cost is stated in the code -- for two unrelated
  subjects a vague relationship claim now discharges the core slot
  too, with the relevance judgement as the backstop.
- A claim that declares no `answer_slot` is no longer refused for
  that alone. The slot is the synthesiser's statement of intent, not
  a property of the claim, and a smaller local model omits it on
  every claim: a real run on `qwen3:4b` withheld three otherwise
  publishable claims for a missing field and published nothing at
  all. The fake synthesiser in the tests always declares one, so the
  whole suite passed while local mode was unusable.

  What fails closed is unchanged. A slotless claim still needs an
  affirmative relevance judgement, still passes every support gate,
  and still counts toward no slot -- so a report built only from such
  claims reports that it did not answer the question.
- `AnswerCoverage.answered` requires **every** core slot, which is
  what the code always did; the docstring said "at least one". Only
  multi-part questions have more than one core slot, and that is
  exactly where the strict reading matters.

Provenance, because a decision that leaves no record cannot be
audited:

- The contract, the propositions with their per-part entailment, the
  relevance decisions and every repair attempt — accepted and refused
  — are carried in the result payload. They were previously prose in
  a `reason` string, or discarded entirely. `claim.text` is
  reassigned in place on repair, so a published claim's earlier
  wording existed nowhere.
- Each claim carries its `answer_slot`. The slot decides the
  relevance verdict and was the one input to it that went
  unrecorded: hosted acceptance produced a rewrite refused with
  "cannot fill the contrast slot" and published the identical
  sentence under a different slot, which was correct and unreadable.
- `/api/readiness` reports `quota_namespace`.
- Recording schema 4. The three committed recordings carry
  `contract: null` and empty answer slots, which is truthful: they
  predate the contract and declared no slots against it.

Deployment:

- `DEMO_QUOTA_NAMESPACE` separates one deployment's daily counter from
  another's sharing a store. Empty by default, so an existing
  deployment's key is unchanged.
- `deploy/render-rc.yaml` deploys a release candidate beside the
  public demo. One line changes per acceptance: `branch`.
- `examples/live-validation/tools/acceptance.py` performs the hosted
  capture: credential-free checks, one run streamed to disk byte for
  byte, then the artifact derived offline from those bytes.

## v1.1.1 — 2026-09-28

Progress reporting only. No change to research behaviour, verification
thresholds, publication gates or recorded results.

- The runner emits `verifier_waking` **before** the readiness probe
  starts, rather than after it finishes. A wake announced afterwards
  describes a wait that is already over.
- The event carries the configured wake budget, so the interface can
  state how long the wait may be instead of leaving a reader to guess
  whether the page has stalled.
- `verifier_ready` follows a successful readiness check.
- The interface explains the cold start — "Waking the verifier (up to
  ~90s)" — where it previously showed step 1 with no explanation.
- Local verification shows no remote wake estimate. Quoting a
  scale-to-zero budget for a checkpoint loaded from disk would be a
  number invented for the occasion.

Why: a verifier at minimum replicas 0 takes roughly a minute to start,
and that happens after `started` and before the first pipeline stage.
Measured on the committed acceptance capture, 70.7s of pipeline stages
inside a 144.4s run left 73.7s outside them, carrying heartbeats and
nothing else. The work was real and was never narrated, so the page
read as hung for more than a third of the run.

## v1.1.0 — 2026-09-28

Live research integration. See
[the release notes](https://github.com/NirmalKumar31/agentic-research-engine/releases/tag/v1.1.0)
for the measured hosted run, which published 0 of 6 claims.

## v0.2.0

Replay-only deployment, with the generative claim verifier replaced by
a pinned NLI classifier and deterministic guards.
