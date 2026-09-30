# Changelog

Notable changes per release. Dates are UTC.

## Unreleased

Two corrections, both from one hosted run that falsified a prediction
written down before it.

**The claim bound is the contract's, not the call budget's.** v1.6.1
removed the cap because it priced a per-claim verification cost that
the NLI classifier had made free. Removing the justification was
right; removing the cap with it was not:

| | cap of 7 | no cap |
| --- | --- | --- |
| Claims generated | 5 | **13** |
| Published | **1** | **0** |

The cap had a second job nobody had written down. Unbounded, the
synthesiser wrote thin claims until the evidence ran out, and a larger
batch of thin claims fared worse at the relevance gate than a smaller
batch of considered ones. The bound is back as a **quality** bound,
labelled as one, and tied to the thing that says how much answer was
asked for: two claims per required part of the contract.

**The relevance judge is asked the contract's question.** It had been
asked "does this help answer the question?" while being shown a list
of parts it was not asked about, so it applied its own notion and
contradicted the contract the report is scored against:

| Run | Claim | Verdict |
| --- | --- | --- |
| v1.6.0 | *LLMs are built upon deep neural networks.* | relevant → published |
| v1.6.1 | *An LLM is a neural network.* | **irrelevant** |

Both declared `relationship`. That slot exists because when one
subject is a kind of the other there is no contrast to draw and
saying so *is* the answer. The judge refused a claim for failing a
test the contract had already excused it from.

It is now asked whether a claim fills one of the listed parts, and
told to judge each claim on its own rather than as candidates for one
place. **This is a tightening**: the judge may no longer freelance,
and a claim filling no listed part still fails. Both the refusal
clause and the structural checks are asserted by tests.

**The bound is a request, not a ceiling**, and the docstring says so.
A local run asked for six claims and produced eight; nothing trims
the surplus. Not enforced deliberately — every claim is gated
individually and an extra one costs no model call, so exceeding the
request is untidy rather than unsafe, while truncating a report to a
count would discard claims before anything had looked at them. What
the request buys is measured: unbounded, thirteen claims; asked for
six, eight.

**A local model cannot evaluate the relevance-judge change.** Both
prompts were run head to head on `qwen2.5:7b-instruct` against the
two claims a hosted run judged oppositely plus two controls. Every
claim came back *no* under both, including the one the hosted critic
published. The local judge is saturated at refusal, so an A/B on it
would report "no change" whatever the change was. Recorded in
LIMITATIONS: a prompt change to this gate can only be evaluated on
the hosted path.

So of the two corrections, the contract bound is verified end to end
locally and the judge change is not verifiable without one hosted
run. The run that motivated both is committed at
`examples/live-validation/v161-20260929-191831/`, with the prediction
it falsified.

## v1.7.1 — 2026-09-30

Retracts an over-general claim v1.7.0 published about its own engine.

Every one of the first eight hosted runs asked the **same question**,
and v1.7.0 generalised from them that the reports are "too thin to use
as research". That question is a hypernym comparison — the engine's
worst shape by construction, since a comparison's core slot is a
direct contrast, a contrast asserts two things, and the atomicity
guard refuses compound claims. Eight of nine contract shapes had never
been exercised on a deployment.

Four runs across other shapes:

| Shape | Published | Cost | Time |
| --- | --- | --- | --- |
| definition | **2 of 3** | $0.009275 | 119s |
| numeric | **1 of 1** | $0.005784 | 52s |
| procedural | 1 of 4 | $0.010065 | 96s |
| comparison (×8) | 0–3, mostly 0–1 | ~$0.009 | ~150s |

The definition run filled both contract slots with exact-normalised
quotes and no false limitation. The numeric run answered correctly in
52 seconds — *"GPT-4 Turbo has a context window of 128,000 tokens"*,
entailed at 0.9975.

So the blanket negative is withdrawn and replaced with what four runs
support: **output quality varies by question shape.** Definitional and
lookup questions do well; comparisons and procedures do not. One
question per shape is coverage, not a benchmark, and live research
stays labelled experimental and fail-closed.

**One finding.** The numeric run's published claim cites two quotes.
One entailed it at 0.9975 and carried it; the other scored 0.0011 and
failed the guards, and the citation list shows both with nothing
distinguishing them. The claim is properly supported, but a reader
clicking the second citation sees a quote the verifier rejected and
cannot tell. The per-evidence scores are in the audit record and not in
the presentation. Recorded, not fixed.

Study at `examples/live-validation/question-shapes/`.

## v1.7.0 — 2026-09-30

Live research is described as what it is.

One run at `aeb4b07f` under a decision gate set before it was taken:
two or more relevant supported findings would end feature work with a
tag; zero or one would stop the patching and reposition the live path.
**It published one.**

Both fixes under test behaved as designed and neither moved the
published count. The contract bound held the report to five claims
where the unbounded run wrote thirteen; a `direct_contrast` claim was
attempted for the first time and refused on atomicity; the report
carried no false "did not answer the question".

So the product now says what eight measured runs show:

- The interface tells a visitor, where they are about to use it, that
  live research is **experimental and fail-closed** — a run may
  publish nothing and that is the design working.
- The recorded runs are labelled **the better demonstration**, and the
  README says to start with them.
- The README gains "What live research does and does not do", with the
  eight runs' published counts and the measured bottleneck: 58% of
  claims refused on synthesis quality, 17% on evidence.

No behaviour changed. What changed is that the claim matches the
measurement.

The run is committed at
`examples/live-validation/final-20260930-033343/` with the gate it was
taken under, and there were no retries.

## v1.6.1 — 2026-09-29

Deployment and evidence. No change to research behaviour.


- Daily admissions raised from **5 to 8** (`DEMO_PROVIDER_REQUESTS_PER_DAY`
  150 -> 240, still `// MAX_PROVIDER_REQUESTS`).

  Raised once four hosted runs had been measured rather than guessed
  at. They used 14, 12, 13 and 13 provider requests and cost $0.0085,
  $0.0093, $0.0081 and $0.0085 — roughly a fifth of the $0.05 per-run
  ceiling the budget is sized against. Eight runs is ~$0.07/day
  measured, $0.40/day at the reserved worst case, against a $5 project
  cap worth about 590 runs.

  `MAX_PROVIDER_REQUESTS` stays at 30. Deriving the cap from a tighter
  per-run ceiling would buy the same runs from a smaller budget, but
  that number is also where a single run is cut off, and a run
  truncated mid-flight is worse than one fewer run a day.

  Quota is reserved before dispatch and not refunded on failure,
  deliberately — a refund path is how a broken loop spends a whole
  budget. Two of five admissions on 2026-09-29 produced nothing, so
  debugging spends the allowance about twice as fast as it reads.

### Hosted run: v1.6.0's required-slot fix confirmed

`examples/live-validation/v160-20260929-174352/` — 166.0s, $0.009190,
13 of 30 provider requests, zero errors.

| | v1.5.0 | v1.6.0 |
| --- | --- | --- |
| `direct_contrast` claims written | 0 | **1** |
| Claims generated | 3 | 5 |
| Published | 0 | **1** |

The first contrast claim across five hosted runs, and the first
report in three runs that publishes and does not claim to have
failed.

**Found: a contrast is not atomic.** That claim — "LLMs learn to
predict token sequences in large text corpora, whereas…" — was
refused by the atomicity guard, correctly, because it asserts two
things. The contract asks for a contrast, the guard refuses compound
claims, and a contrast is compound by nature, so `direct_contrast`
may be systematically unfillable while atomicity holds.

Recorded rather than changed. Loosening atomicity reopens the defect
audits 1–3 closed, and the `relationship` alternative already covers
the case — it is what published here. Expressing a contrast as two
atomic claims filling the slot jointly is a contract-design change
that deserves its own evidence.

## v1.6.0 — 2026-09-29

The synthesiser is told which slot the answer turns on.

A comparison's slots were listed to it identically:

```
- direct_contrast: An explicit statement of how the subjects differ
- dimension: A named dimension along which they differ
- relationship: How the subjects relate…
```

Three options, no signal. The hosted run on v1.5.0 wrote **three
`dimension` claims and no contrast**; the relevance gate refused two
of them for describing one subject instead of contrasting them —
correctly — and the report published nothing.

The contract marks `direct_contrast` as core. The call site flattened
the slots to `(name, description)` pairs and dropped `core` before
the prompt saw it. **Sixth instance in this project of a value
computed and then not passed to the thing that needed it**, after the
durable quota, `answer_slot`, `satisfied_by`, `SourceIdentity`'s
authority and the slot descriptions in the relevance prompt.

Slots now render as `(REQUIRED)` or `(optional)`, and the prompt
states the consequence: a report that fills optional parts while the
required one is missing has answered nothing. Naming the slot was
never enough — the v1.5.0 run named all three and the model picked
the easiest.

**Unverified on the hosted path.** The fix landed after the capture
that motivated it, and today's run allowance is spent.

### Hosted run: the meta-claim fix confirmed

`examples/live-validation/v150-20260929-170940/` — v1.5.0 at
`636e9e86`, 171.3s, $0.008139, 13 of 30 provider requests, zero
errors.

| Run | Claims about the evidence |
| --- | --- |
| v1.2.0 | 1, at entailment 0.007 |
| v1.4.1 | 2 of 3, at 0.031 and 0.115 |
| **v1.5.0** | **0** |

The prompt had been teaching it; it no longer does, and the model
stopped. That fix is verified. It published nothing, for the slot
reason above.

## v1.5.0 — 2026-09-29

A claim asserts something about the subject, not about the evidence.
The synthesiser prompt was teaching the opposite.

Three hosted runs produced claims of the form "The evidence describes
X" and "The architectures discussed in the study are based on Y".
Every one was refused, at entailment **0.007, 0.031 and 0.115** — the
premise is the quote, and the quote does not say what the evidence
describes, it just says the thing. In the v1.4.1 run that was **two of
three claims**, and the report published nothing.

The prompt caused it. Two of its three worked examples for splitting a
compound claim began *"The source reports"*. The model was following
the instruction it was given, and the verifier was correctly refusing
the result.

Measured on the pinned checkpoint, against a quote reading "Large
language models are built on artificial neural network architectures":

| Claim form | Entailment | |
| --- | --- | --- |
| plain assertion | **0.998** | publishes |
| "The source reports that…" | 0.856 | withheld |
| "The evidence describes…" | 0.519 | withheld |

Adding a frame the quote does not have costs up to 0.48 and
guarantees refusal. Those numbers are now in the prompt, because a
rule with a measurement behind it is one a model can weigh.

**The exception is kept and is load-bearing.** Audit 2 of this project
found "We demonstrate that X" published as bare "X", which presents
one paper's result as the field's agreement. When the *quote* is
framed, the claim must carry the frame. The rule is not "never
attribute" — it is carry the frame the quote has, never add one it
does not. A test asserts the framing guard still catches a deleted
frame, so the two rules cannot drift apart.

**Unverified on the hosted path.** The fix landed after the capture
that motivated it, and confirming it costs a paid run.

### Hosted run: live research restored

`examples/live-validation/v141-20260929-163832/` — v1.4.1 at
`c8f9d14f`, 113.0s, $0.009305, 13 of 30 provider requests, **zero
recorded errors**. Live research completes again after the v1.4.0
failure.

Selection now reaches better material: an arXiv source at quality
**0.98**, academic, six citable quotes, where a pre-fix local run on
the same question selected six blogs at best 0.57.

It published nothing, for the reason above, so whether the better
source gets *cited* is still open on the hosted path. The capture
tool now says when a stream is incomplete rather than printing a byte
count that reads like success — two captures were reported that way
and neither was a run.

## v1.4.1 — 2026-09-29

A run no longer dies because an advisory model call did.

The first hosted run on v1.4.0 reached `assessing_coverage` with six
sources and twenty-eight extracted quotes, then emitted an error
instead of a report. The capture is kept in
`examples/live-validation/failures/`.

The coverage critique is advice. Every number that routes the run --
the per-question verdicts, the ratio, the domain concentration -- is
computed from evidence already in hand *before* the model is called,
and the node's own comment said "the counted half still stands, so
routing remains sound". It caught `LLMError`. Anything else ended a
run that had already paid for four searches and twenty-eight
extractions, in order to lose an opinion.

Six other calls had the same shape: question analysis, planning,
query generation, follow-up generation, synthesis, wording repair and
the relevance judgement. Each has a real fallback -- the raw question,
a single dimension, the sub-question text, an evidence-only report,
the original refusals, withholding -- and each fired only for the
failure type someone happened to anticipate. A fallback like that is
a promise the code does not keep.

All seven now degrade on any exception, and record the exception type
into the run's error list so a bug surfaces as a degraded run rather
than as silence.

**This cannot swallow the wall-clock deadline.** `asyncio.timeout`
cancels with `CancelledError`, which derives from `BaseException` and
passes straight through `except Exception`. That property is what
makes the widening safe, and it is asserted by a test rather than
assumed.

**The root cause of that run's failure is not known.** The public
error message is deliberately generic and the server log was not
retrieved before it rotated. A local reproduction at the same commit
completed normally, so the failure did not reproduce. What changed is
that this class of failure now degrades instead of ending the run;
what has not changed is that nobody knows which exception it was.

## v1.4.0 — 2026-09-29

Source quality now reaches the three decisions that should have been
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

- **Selection prefers the better page before fetching it.** The layer
  that decides what the engine ever reads sorted on the search
  provider's relevance score alone. That is the most consequential of
  the three, because the engine reads six pages: a local run on this
  question selected five blogs and a sixth blog, so no later
  preference for better sources had anything to prefer. The kind of a
  page is knowable from its URL before it is fetched, which is what
  makes the decision possible there. Banded the same way — an
  authoritative page about the wrong subject is worse than a blog
  about the right one.
- **The relevance judge is told what each slot means.** It was shown
  bare names — `direct_contrast`, `dimension`, `relationship` — while
  the contract carried a sentence describing each, and which are
  required. Fourth instance in this release of the engine computing
  something useful that stopped at a boundary.

  Honest about the result: this was tried against local mode's
  zero-publication problem and **did not fix it**. A 4B critic still
  rejects claims that plainly fill a listed slot. The change is kept
  because asking a model to judge against a bare token is asking it
  to guess, not because it produced an improvement. A line telling
  the judge that filling one part suffices was also tried and
  **reverted** — it loosens the gate, showed no effect, and measuring
  it on the hosted critic costs a paid run.
- `authority_of` / `authority_rank_of` map a source kind to how close
  it is to what it reports, beside the enum that makes the same
  distinction rather than in a second table that would drift.

Neither field ever enters the NLI premise; the existing test that the
scorer never sees them still holds, and a new one checks it through
the real wiring.

Measured, on the same question and the same limits, before and after
the selection change:

| | before | after |
| --- | --- | --- |
| Best source selected | 0.57 blog | **0.96 academic (arXiv)** |
| Kinds selected | blog only | academic, other, blog |

Two live runs are **not a controlled comparison** — web search is
nondeterministic, and the candidate pool differed. The direction is
what the unit tests pin; this is evidence the mechanism reaches a real
run, not a measurement of how much it helps.

Both runs still published nothing. Locally that is the critic, not
retrieval: a 4B model rejects most of what it is given, which is
already recorded in LIMITATIONS. Better sources do not fix a weak
critic, and this release does not claim they do.

No change on the frozen adversarial set: 0 irrelevant published, 6
published, unchanged throughout. It supplies its own sources and never
exercises selection.

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
