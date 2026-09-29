# Changelog

Notable changes per release. Dates are UTC.

## Unreleased

Source quality now reaches the two decisions that should have been
using it. Both defects were found by reading the committed run
evidence rather than the code.

In both hosted runs the best eligible source — usable, with citable
evidence — was never cited. One was an arXiv survey scoring 0.95 with
six citable quotes, passed over for a blog. The engine classifies
sources and neither decision could see the classification.

- **Synthesis is shown the better source first.** `build_package`
  ordered evidence by extraction confidence alone, so among quotes
  that answer a sub-question equally well the choice was arbitrary,
  and the per-question cap dropped good sources at random. Ordering
  is now contradictions, then a relevance band, then the source, then
  relevance again. Banded deliberately: sorting by quality outright
  would put a barely-relevant quote from a good source above the one
  that actually answers the question.
- **The entailment gate receives the authority it ranks by.**
  `verify_claim` orders equally-entailed quotes by source authority
  and quality. `SourceIdentity` was built in production with only
  `domain` and `title`, so every source ranked UNKNOWN at quality 0.0
  and that ordering collapsed to entailment alone. Implemented,
  tested against hand-built identities, and inert where it mattered.

  Reverting the wiring broke no test, which is how it survived. The
  adversarial eval builds its own identities correctly, so it
  exercised the ranking the whole time and could never have caught
  this. There are now tests on the function that builds the identity.

- `authority_of` / `authority_rank_of` map a source kind to how close
  it is to what it reports, beside the enum that makes the same
  distinction rather than in a second table that would drift.

Neither field ever enters the NLI premise; the existing test that the
scorer never sees them still holds, and a new one checks it through
the real wiring.

No change on the frozen adversarial set: 0 irrelevant published, 6
published, unchanged throughout.

## v1.3.0 — 2026-09-29

The interface shows what the question required. Verified on the
deployment, which found a defect that only a deployment could show.

- **The answer contract is rendered beside the report**: each required
  slot, which the published claims filled, and a plain statement when
  the report does not answer the question. It names the slot that
  discharged a core requirement when an alternative did. Derived from
  the published claims rather than read from a field, so it cannot
  drift from what was published. An unfilled slot is grey, not red —
  a gap in the answer is not an error in the run.

  The three committed recordings carry no contract, so the panel is
  hidden for them rather than rendered empty. Showing one would imply
  they were held to a contract and failed it.
- `AnswerContract.to_dict()` now serialises `satisfied_by`. It did
  not, so nothing outside the engine could tell that a core slot had
  been discharged by an alternative: the interface recomputed
  coverage, found the contrast slot unfilled, and would have rendered
  "this report does not answer the question" directly above a report
  whose own limitations said otherwise. A page contradicting the
  report beneath it is worse than either verdict alone.

  This is the second v1.2.0 record unreadable for the same underlying
  reason — a decision made inside the engine whose explanation did not
  travel. The first was `answer_slot`.

### Verified on the deployment

One authorised run on the public demo at `1d21b110`, on the question
v1.2.0 answered wrongly:
[`examples/live-validation/v121-20260929-024544/`](examples/live-validation/v121-20260929-024544/)

| Claim | Slot | Outcome |
| --- | --- | --- |
| An LLM is a type of neural network that uses transformer… | `relationship` | **published** |
| The evidence distinguishes LLMs as a specific class… | `direct_contrast` | withheld, entailment 0.007 |

The only claim that would have filled the contrast slot directly was
refused by its own evidence; the core requirement was discharged by the
alternative, and the report carries no "did not answer the question"
limitation. v1.2.0 reported the opposite on the same question. Four of
six claims were still withheld, by four different mechanisms — the fix
did not make the gate permissive.

**Not verified by that run:** the missing-`answer_slot` path. The cloud
model declared a slot on every claim, so the behaviour that had made
local mode publish nothing was never reached. Unit tests and two local
runs cover it.

Also recorded in LIMITATIONS: the deployed demo is no longer
blueprint-managed, so a change to `render-live.yaml` will not reach it.

## v1.2.1 — 2026-09-29

Deployment naming only. No change to research behaviour, verification
thresholds, publication gates or recorded results.

- The public demo's Render service is `agentic-research-engine-live`
  again. v1.2.0 renamed it to `agentic-research-engine`, which cannot
  be deployed: a blueprint matches an existing service by name, so
  changing the name reads as "delete that service and create a
  different one" rather than as a rename, and Render declines it on a
  sync. Two syncs produced nothing.

  Reverted rather than pursued. Making it stick means a full teardown
  with every credential re-entered, for a cosmetically shorter
  hostname. The reason is now a comment in the blueprint so the next
  person does not try it again.
- The replay blueprint keeps `agentic-research-engine-replay`. That
  part of the rename was a real improvement and nothing was deployed
  under the old name.
- The README's demo link points at the service that exists.

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
