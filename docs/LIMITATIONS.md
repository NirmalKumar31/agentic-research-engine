# Limitations

What this system does not do, and what its numbers do not mean. Current
as of the tree containing this file.

## Evaluation

**It does not verify truth.** Every metric measures faithfulness to
retrieved sources. A report citing five wrong pages, accurately, scores
well. Closing that needs labelled answers, which is a different project.

**Exact quote matching proves textual alignment, not entailment.** A quote
marked exact was found verbatim in the retrieved text. Whether it supports
the claim is a separate check, done by a model.

**Claim support is judged by a learned classifier, and it can be wrong.**
A pinned DeBERTa-v3-large MNLI checkpoint scores entailment; Python
applies the threshold and the guards. It is no longer the same model that
wrote the report, which was the previous and worse arrangement. But an NLI
model is a statistical classifier trained on a particular distribution of
sentence pairs, and research claims about vector databases or AI risk
frameworks are not that distribution. Measured behaviour on the cases
tested is recorded; behaviour outside them is not characterised.

**The guards are narrow on purpose and miss things.** Five checks —
numeric literals, modality strength, invented rankings, causation from
association, invented exclusivity — cover transformations that were
observed being approved. They do not parse sentences. An overclaim
expressed as a noun-phrase substitution, such as an attribute stated of
*accuracy* reattached to *recall accuracy*, passes every one of them.

**Published figures come from single runs**, except the attribution
experiment, which has three repeats. There are no confidence intervals.

**The twelve-question benchmark has never been run.** Construction and
validation for it exist — the question set, a frozen-configuration
manifest, and a blinded-review mechanism — at
[`docs/BENCHMARK-PROTOCOL.md`](BENCHMARK-PROTOCOL.md). No run has been
authorized or executed; that document states the exact cost and stop
rule a run would need approved first.

**There is no independent benchmark result for the verifier.** The thirty
labelled cases are development calibration data: they shaped three
generative verifier designs and the NLI replacement, so agreement measured
on them is partly fitted and is not a generalisation estimate. The only
unfitted measurement is the synthetic adversarial suite, which tests
refusal of specific transformation classes and nothing wider.

**No clean local-versus-cloud comparison exists.** The earlier one used a
corpus whose source text had been stripped, which drove citation integrity
to 0% in both arms. It is archived under `examples/archive/` and excluded
from current results; its 55.6%/81.2% figures are not valid measurements.

## The attribution experiment

Scope, precisely: one frozen corpus of six sources on one question, three
repeats, `qwen3:4b` (digest `359d7dd4bcdab3d8`, Q4_K_M), one round, five
sources per run.

On that corpus, retrieved-only extraction reduced evidence coverage from
100% to 16.7% while all-open stayed at 100%, so production kept all-open
extraction. The cause is the corpus's discovery structure: each source was
retrieved for a mean of 1.17 sub-questions, so under retrieved-only most
sub-questions can only ever see one source, and a criterion requiring two
distinct sources becomes near-unreachable. A corpus whose sources are found
by many queries would narrow the gap.

This is not a general claim that all-open extraction is optimal.

The quote-fidelity difference in that experiment (75.9% against 88.3%) is
**exploratory**. n=3, strategies ran grouped rather than counterbalanced,
and thermal state and execution order are not ruled out.

The frozen corpus itself is not committed, so the experiment is an
**archived measured run** rather than a reproducible one. The artifact
records the commit, model digest, prompt and schema hashes and
configuration fingerprint needed to reconstruct the conditions.

## Retrieval and evidence

**Cross-attributed evidence has no query provenance.** Between 20.8% and
83.3% of evidence across the three recorded runs answers a sub-question
whose queries never retrieved that source. The chain to the source is exact; the
chain back to a query exists only for the rest, and is recorded as absent
rather than guessed.

**Quote fidelity is measured against the copy held**, which for a
provider-supplied source is the search provider's text rather than an
independently fetched page. `content_origin` records which.

**PDFs cost an extra request.** Provider text for a PDF arrives with page
boundaries flattened, so likely PDFs are fetched directly. If that fetch
fails the source degrades to provider text — with a warning, but without
pages.

**A recorded run is a snapshot of one afternoon's web, not a
reproducible experiment.** Live search returns different results on
different days, so re-running the same question does not recover the
same sources, the same evidence or the same claims. Across five
recording passes of the identical three questions the candidate count
ranged from 6 to 24 per run and page-level citations appeared in
different runs each time. Each recording stores the commit, the model
and its digest, the NLI model, revision and threshold, the retrieval
configuration fingerprint and the timestamp — everything except the
corpus, which is why the runs are snapshots rather than experiments.
Only the frozen-corpus attribution experiment is reproducible in that
stronger sense.

**Which run carries PDF page evidence is not stable, and nothing
promises it.** Whether a PDF is reached depends on what search returns
that day: across four recording passes the page-numbered citations
moved between runs and twice vanished entirely. The current recordings
have none. Capability badges are therefore computed from each
recording's own contents rather than asserted by recording id, and a
schema check fails if a stored count disagrees with the payload. A
recording description may not promise a retrieval outcome.

**Page provenance is not page selection.** Extraction preserves which page
a quote came from; it does not steer the extractor toward the most
useful pages of a long document. In the NIST recording the first-party
quote supporting the four-functions claim is a table-of-contents line,
while the explanatory prose sits many pages later. The citation is
accurate about where the text came from and that is all it asserts.

**No OCR.** A scanned PDF is detected and reported, not read.

**JavaScript-only pages yield nothing** and are marked `EMPTY`.

**Source quality scoring is an untuned heuristic** over document type,
rank, length and recency. It orders sources for extraction; it is not a
claim about correctness.

**Coverage thresholds are heuristic.** `_SUFFICIENT_RATIO = 0.7`, and "two
exact-match items from two distinct sources", are defensible and untuned.

**Brave is implemented but never tested against the live API.** The
provider abstraction is proven at the type level only.

**English-centric.** Extraction prompts and quote matching are untested on
other languages.

## Comparisons

**A comparison cannot be sourced from the subjects' own documentation,
because neither documents the other.** `docs.langchain.com` describes
LangGraph; it never says "unlike LangChain". A vendor documents its own
product, so first-party pages are structurally single-subject.

The engine therefore has to *assemble* a contrast from one verified claim
per subject on a shared axis, which is what `build_comparison_pairs`
does. But a low-authority page comparing two products in a single
sentence offers a ready-made contrast, and that sentence is more
*relevant* to a contrast slot than any single-subject quote from
authoritative documentation. **Relevance and authority pull in opposite
directions here, and relevance wins — correctly, because a quote that
does not support the claim is useless however good its source.**

Seven hosted runs of "langchain vs langgraph differences?", one per
release from v1.9.0 to v1.15, show the shape.
The most-cited source was a 0.59 blog in one run and a 0.64 listicle in
another; in the same two runs, first-party documentation scoring 0.89 to
0.96 supplied **five and five** citable quotes respectively and was never
cited. The blogs supplied sentences like "LangChain handles state
implicitly ... LangGraph provides explicit control over state". The
documentation supplied accurate single-subject descriptions that fill a
contrast slot only once paired, and pairing requires both halves to be
present on the same axis in the same run.

Telling the synthesiser to prefer the authoritative source does not
resolve this, and v1.15 shipped that rule without being able to
demonstrate an effect: the rule applies *among quotes supporting the same
point*, and for a contrastive point there is frequently no competing
documentation quote at all.

**What this means in practice.** Comparison questions about two products
will often cite secondary sources even when primary ones were retrieved
and were usable. The citation is still exact, the claim is still
entailed by its quote, and the source is still named with its quality
score — but "verified" here means faithful to a blog, and the report says
which.

## Publication

**All three recorded demos publish nothing.** Under a `qwen3:4b`
synthesiser, 35 substantive claims were generated. One exact duplicate
was removed, and the verifier evaluated the remaining 34 unique claims:
28 partially supported, 6 unsupported. The fail-closed publication gate
removed every one, so no synthesised claim was published.

**The publication gate is binary, and the three-way verdict is not.**
Only `supported` publishes. `partially_supported` and `unsupported` both
withhold, and the boundary between them is derived from the scores after
the fact for readability. Nothing branches on it, and it is not a release
metric.

**The verifier was replaced because the previous one did not work.**
Three designs using `qwen3:4b` as an entailment classifier were measured
against human labels and all three failed: the first returned `supported`
for none of thirty claims, the second for twenty-five of thirty including
sixteen the reviewer had marked otherwise, and the third produced
malformed audits on eleven of thirty. Those experiments are preserved on
the `verifier-v1`, `verifier-v2` and `verifier-v3` branches. The
conclusion was that a 4B instruction model is not a stable semantic
classifier, not that the prompt needed more work.

**The thirty labelled cases are development calibration data, not a
benchmark.** They influenced four verifier designs. On them, at threshold
0.98, the selected model publishes 6 of the 9 reviewer-supported cases
and one case the reviewer marked partially supported. That one false
positive is a compound claim: the quote states a 75% memory reduction and
"high accuracy" and "minimal recall impact" separately, and the claim
fuses them into "maintaining high recall accuracy". No guard reaches it
and all three candidate classifiers score it above 0.95. The fix was in
synthesis — atomic claims — not in another guard.

**The release audit passes, and it took four attempts.** Every
published claim in the three canonical recordings was read against the
exact quote the gate chose for it: 11 of 11 supported, 0 unsupported.
The first three audits each published exactly one claim that survived
every automated check and failed a human read — a deleted hedge
("might lack" as "lack"), a deleted research voice ("We demonstrate
that X" as "X"), and a first-person scope deletion hidden inside a
two-sentence claim. Each produced a general rule. None of that is
evidence the next audit would be clean: the runs use live search and
produce different claims every time, and the only thing that caught
these was reading every published claim by hand.

**Atomicity is enforced by a heuristic, not a parser.** A claim must
assert one independently verifiable proposition, and this is checked
before publication rather than merely documented: the check splits on
coordinators and counts segments carrying their own predicate, treating
contrastive coordinators as compound on sight. Synthesis is asked for
atomic claims first; this is the backstop for when it does not comply.

It is a clause and predicate heuristic and will be wrong in both
directions. It calls some single-proposition sentences compound and
withholds them. A genuinely fused single-predicate proposition can
still evade it, because nothing here parses grammar. Where it cannot
tell, it treats the claim as compound — false positives withhold, which
is the chosen direction of error.

One release shipped with this check written, tested against a
seventeen-case matrix, and *not wired into the guard*, which still
counted sentences. The report said proposition atomicity gated
publication and it did not. Every guard now has an integration test
that drives the real publication path with a scorer entailing
everything at 1.0, so a disconnected guard fails a test rather than a
review.

**Zero supported false positives was not achieved on that development
set.** It was achieved on the synthetic adversarial suite, which is the
unfitted measurement. Both numbers are published because reporting only
the favourable one would misrepresent what is known.

Unsupported and partially supported claims are removed before publication,
not rewritten. A published report therefore contains no claim that failed
the support check — which means "every published claim passed this
verifier", not "every published claim is true". The counts of what was
generated and removed are kept in the verification record.

**A report can publish nothing, and that is a supported outcome.** The
current recordings all publish something, so they do not demonstrate
it; whether a run publishes zero depends on what search returns that
day, and re-recording until one does would be exactly the selection
this project refuses. The path is instead driven end to end in tests —
a real graph run whose verifier entails nothing, rendered through the
real renderer — asserting that every excerpt is a stored quote, that no
generated prose appears, and that the published count stays zero.
 When
every claim is withheld and citable evidence exists, the report renders
verbatim source excerpts under a notice saying no synthesized claim
passed verification. Those excerpts are quotations, not findings:
`final_published_claims` stays 0 and they are counted separately in
`evidence_only_excerpts`. Nothing generalises, joins or interprets them.

### Live research is experimental and fail-closed

Measured across eight live runs on a deployment, committed with their
raw streams under `examples/live-validation/`. Published claims: **0,
0, 0, 1, 1, 2, 3 and 1.**

The verification design works end to end and every refusal inspected
across those runs was the correct refusal.

**Those eight runs all asked the same question**, and it is the
engine's worst shape: a hypernym comparison, whose core slot is a
direct contrast that the atomicity guard refuses because a contrast
asserts two things. Four later runs across other shapes published 2 of
3 (definition), 1 of 1 (numeric) and 1 of 4 (procedural). A blanket
statement that the reports are too thin was generalised from one shape
and is withdrawn; see `examples/live-validation/question-shapes/`.

What holds is narrower: **output quality varies by question shape**,
definitional and lookup questions do well, comparisons and procedures
do not, and one question per shape is coverage rather than a
benchmark.

### Comparisons: five fixes, no movement

Ten runs on one comparison question published 0, 0, 0, 1, 1, 2, 3, 0,
1, 1. Five separate defects were found and fixed on that path — the
`relationship` alternative for a hypernym pair, serialising
`answer_slot`, marking the required slot to the synthesiser, a prompt
asking the judge to judge against the listed parts, and narrowing the
judge's authority in code so it cannot veto an optional-slot claim.

Each was a real defect. Each is tested. The last was verified firing
on the hosted path — a `dimension` claim published with a negative
judgement recorded against it, which no earlier run could do.

**None of them moved the published count.** After all five, the
synthesiser still writes claims about one of the two subjects, and the
report-level gate is right to refuse to call that an answer to how
they differ.

So the constraint on this shape is not a gate and is not a wiring
defect. It is what the synthesiser writes, and changing that is a
different kind of work than the five fixes above — one that needs an
evaluation harness rather than another patch and another paid run.

The bottleneck is measured and it is not the gates. Across 24
generated claims, **58% were refused on synthesis quality** (38%
irrelevant, 21% guard failures) and **17% on evidence** being below
threshold. The synthesiser writes few claims that both answer the
question and survive verification.

Six fixes were made upstream of the gates over those runs, each real
and each tested, and the published count moved between zero and three
throughout. That is the honest shape of the result: the defects were
genuine, and fixing them did not turn the engine into a research
assistant.

**So the recorded runs are the demonstration and the live path is an
integration test that visitors can run.** The interface says so where
a visitor is about to use it, rather than only here.

### The relevance judgement is only as good as the critic

Structural relevance is deterministic. The judgement that follows it
is a model call, and its quality tracks the model.

Measured, on the same question in the same week:

- Hosted, `gpt-6-luna`: asked how a large language model differs from
  a neural network, the critic judged *"LLMs are a specific subset of
  neural networks"* **relevant**, and it published.
- Local, `qwen3:4b`: given the near-identical claim *"Large language
  models are a specific type of neural network architecture"*, the
  critic judged it **irrelevant**, and it was withheld. Of six claims,
  it rejected four and published one.

The small model over-rejects. That direction is the safe one — the
gate withholds true claims rather than publishing irrelevant ones —
but it means local mode publishes sparse reports, and a zero-claim
local run is as likely to be the critic as the evidence.

**Measured across four local runs on the same question: zero
published claims, every time.** It is not one bad draw. Two things
compound:

- The local synthesiser writes definitions when asked for a
  comparison. One run generated 21 claims, nearly all of the form
  "large language models are/do X" — correctly refused by a contract
  whose core slot is a contrast.
- The local critic then rejects what survives, including claims that
  plainly fill a listed slot. It rejected *"Large language models
  often use transformer architectures"* against a contract with a
  `dimension` slot described as "a named dimension along which they
  differ".

Giving the judge each slot's description was tried against this and
**did not change the outcome** — the information was not what the 4B
model lacked. The descriptions were kept anyway, because a judge
shown a bare token like `direct_contrast` is being asked to guess;
the change is justified on its own terms, not by an improvement it
did not produce.

Neither symptom appears on the hosted path with a larger model, which
published 3 of 7 and 2 of 6 on the same question. **Local mode
exercises the pipeline; it does not demonstrate it.**

### A local model cannot be used to test the relevance judge

Measured, because it was tried. Two prompts for the relevance
judgement were run head to head on `qwen2.5:7b-instruct` against four
claims: the two that a hosted run had judged oppositely, and two
controls that must be refused.

Every claim came back *no*, under both prompts — including *"LLMs are
built upon deep neural networks"*, which the hosted critic judged
relevant and published. The local judge is saturated at refusal, so
it cannot distinguish a better prompt from a worse one, and an A/B on
it would report "no change" whatever the change was.

The consequence for anyone working on this gate: **a prompt change to
the relevance judgement can only be evaluated on the hosted path.**
Local runs will confirm that it executes, and nothing about whether
it decides better.

Two consequences worth stating plainly. A local run is not a fair
demonstration of what the pipeline can do. And the relevance numbers
in the adversarial set were produced with entailment pinned and no
critic call at all, so they measure the structural half only; the
judged half has no frozen benchmark, and one hosted run is not one.

## Operations

**Budget enforcement is per process, and not all of it is exact.**
Everything is reserved before dispatch rather than counted after, but the
dimensions differ in what they can promise:

- *Provider-request ceiling* and *output-token ceiling* — exact. Requests
  are counted, and each reserves its role's output cap.
- *Input-token accounting* — estimated at `len // 4` before the request
  is built, so the reservation is an approximation of the real count.
- *Pre-dispatch cost ceiling* — derived from the two above against a
  local price table, so it bounds expected spend rather than the invoice.

**The throttles and the admission counter have different scopes.**
Concurrency and the per-client hourly rate are process-local: they live
in the web process and reset when it does. They bound accidents and
casual abuse, not money.

The *global daily admission counter* is shared, through an external
atomic INCR against Render Key Value. It is the same count for every
web replica and it survives a web-service cold start, which the
in-memory version did not — a sleeping free instance used to wake with
the whole day's allowance back.

**A free Key Value instance is not persistent storage.** Render states
that data persistence is unavailable on the free plan, so a restart of
the Key Value service itself returns the count to zero used. Call this
counter shared, or distributed. Do not call it durable across datastore
restarts unless it is running on a paid plan with persistence enabled.

**The provider account's hard spend limit is the financial backstop.**
Not this counter, and not the per-run ceiling. Set one before enabling
anonymous live research.

**Admission fails closed.** If the counter is required and the store is
unreachable, the run is refused rather than admitted unbounded, and
readiness reports the instance as not ready rather than quietly turning
every visitor away.

**Client identification uses the trusted-hop rule.** `X-Forwarded-For`
is appended to by each proxy, so the address is read relative to the
configured number of trusted proxies rather than from the first entry,
which the caller supplies. Non-addresses and headers that did not
arrive through the expected path fall back to the socket peer. This
raises the cost of minting fresh per-client allowances; it is abuse
mitigation and not the financial boundary.

**Input token estimation is `len // 4`** — adequate for prose and *not*
a conservative bound. Code, dense punctuation, non-Latin scripts and
JSON schemas all exceed a quarter token per character, so the
pre-dispatch reservation can land under the true cost.

**No separate retry budget.** Retries, structured repairs and
compatibility retries are each a distinct provider request and reserve
against the same `max_provider_requests` and `max_cloud_calls` ceilings as
any other, so they are bounded — but nothing limits what share of the
budget they may consume. A run that retries heavily can exhaust its
request ceiling and stop early rather than overspend.

**Model calls are bounded by wall-clock deadlines, within the limits
of cooperative cancellation.** A run once sat at 0% CPU for seven hours
inside a single extraction: the host slept, the connection to Ollama
went quiet without closing, and `llm_timeout_seconds` never fired
because an HTTP client timeout bounds time between socket events and
there were no further events.

Every provider call now runs under an `asyncio.timeout` at the single
router boundary they share, which bounds elapsed time rather than
socket activity, and each run has an independent total wall-clock
deadline because many bounded calls still compose into an unbounded
run. Timeouts fail closed: the call is cancelled, a typed error is
raised, budget and concurrency are released, and nothing unverified is
published.

What this does not claim: that a model call can never hang. Asyncio
cancellation is cooperative. A coroutine that permanently suppresses
`CancelledError` and never yields cannot be forcibly terminated inside
one Python event-loop process, and no timeout here changes that. What
is bounded is ordinary cooperative async calls and the silent network
stall actually observed. Subprocess supervision would be needed for the
stronger guarantee and has not been added, because no provider has
demonstrated the need.

**Checkpoint recovery is untested under load.** No test resumes an
interrupted run.

**Base images are not pinned by digest.** `constraints.txt` pins Python
dependencies; `python:3.12-slim` and `node:24-slim` can move.

**No observability beyond logs.** No metrics endpoint, no trace export.

**The semantic verifier does not fit the free deployment tier.** Measured
peak RSS is 1384MB for the selected model, 1142MB for the DeBERTa base
alternative and 675MB for the MiniLM cross-encoder — against 512MB on
Render's free tier. File size is not a proxy for this: the MiniLM
checkpoint is 331MB on disk and still doubles that resident. MiniLM was
also the weakest on the adversarial suite, publishing three overclaims
the selected model refuses, so trading accuracy for size was not
available either.

The verifier therefore has a remote mode that calls a hosted inference
endpoint instead of loading the checkpoint in-process. It fails closed on
timeout, 429, 5xx, malformed body, network error, a response that does not
state which checkpoint served it, a checkpoint differing from the configured
one, a missing truncation flag, and scores that are not a probability
distribution — every one withholds the claim.

**Remote verification has been exercised against a real endpoint, from
the CLI only.** A private Hugging Face Inference Endpoint was created
on 2026-09-27, pinned to the calibrated revision and running the stock
text-classification handler. Against it, measured rather than assumed:

- All 50 calibration pairs scored, with **no publish decision differing**
  from the local checkpoint at the 0.98 threshold.
- Scores **bit-identical** across repeat runs and across batch sizes 8
  and 1, so batch padding does not move a verdict.
- Raw scores differ from the committed local ones by up to 9.4e-3. The
  cause is precision, not the endpoint: the committed scores are
  float16 and the endpoint runs the same commit in float32. See
  [the calibration results](../examples/verifier-calibration/NLI-RESULTS.md).
- Cold start from `scaledToZero` to a verified-ready verifier: **49.2s**.

What that does *not* establish: the endpoint's pinned revision is
checked out of band against the Hugging Face control plane rather than
echoed per response, because the stock handler reports neither the
model nor a truncation flag. Truncation is measured client-side
instead. This is a weaker arrangement than the project's own scoring
service provides, and it is chosen because the managed handler offers
nothing stronger.

**Hosted acceptance has happened and proves deployment, not research
quality.** Two runs through the deployed HTTP/SSE path under the public
limits (1 round, 6 sources), both on 2026-09-28.

The v1.1.0 acceptance run, at commit `6e34f908`, captured byte-for-byte
and committed under
[examples/live-validation/hosted-20260928-045059/](../examples/live-validation/hosted-20260928-045059/):

- verifier cold start from `scaledToZero`, complete run **144.4s**
  against the 240s ceiling;
- 6 sources retrieved, 34 evidence items, **34/34 quotes exact**;
- 6 claims generated, 6 checked, **0 published**, 6 withheld;
- measured OpenAI cost **$0.009472**, against a reserved upper bound of
  $0.03107 which held;
- the run reported its own cost as *incomplete*, because a response
  carried a token category with no recorded rate.

An earlier run at `7e565492` additionally showed the shared daily cap
refusing the next request, and refusing it again for three forged
`X-Forwarded-For` values, with replay still working once the allowance
was spent. Its result payload was lost to a truncating capture, which
is why the later one was captured raw.

**Neither run says anything about answer quality, and the later one
published nothing at all.** Three of its six claims fell below the 0.98
entailment threshold — one at 0.976 — and three were refused by
deterministic guards for attribution, atomicity and numeric phrasing:
rejected for wording, not for lacking support. The recorded replay runs
remain the better demonstration of what this engine produces. Improving
published usefulness is deliberately not part of this release.

Client disconnect and application-timeout cleanup have unit coverage
but no hosted measurement.

**Live research costs money at three independent providers.** OpenAI
tokens, Tavily search credits and Hugging Face endpoint compute are
billed separately and bound separately. An OpenAI project spend limit
does not stop Tavily or Hugging Face, and a warm inference endpoint
accrues cost whether or not anybody runs research. OpenAI's own limit
is a durable backstop rather than an exact per-cent guarantee, because
enforcement is not instantaneous.

**Live research refuses to start unless the whole path is proven
first.** The verifier is probed before any provider is called: if it
cannot answer, no OpenAI request and no Tavily search is made, and the
visitor is told the service is unavailable rather than shown a report
with nothing in it. The public daily cap is backed by an external
atomic counter, because a process-local count resets whenever a free
instance wakes; if that counter is required and unreachable, live
research is refused rather than admitted unbounded.

That counter was dead code until 2026-09-27: it existed, was tested,
and nothing on the request path called it, so the effective cap was
still the in-memory one. It is now reserved before the model router,
the search provider or the verifier is touched. On Render's free Key
Value plan the store itself has no persistence, so the cap is shared
across web instances and is *not* durable across a restart of the
store; the provider-side account limit is the backstop there.

**Replay still needs no credentials at all.**

**Provider prices are estimates** from a local table. They do not account
for cached input, long-context tiers, region or service tier.

### Source preference is ordering, not sourcing

Three decisions now take a source's kind into account: which pages to
fetch, which evidence the synthesiser is shown, and which of several
equally-entailed quotes carries a claim. All three are *orderings*.
None of them can produce a good source that the search provider did
not return.

A run whose candidate pool is entirely blogs will still read blogs.
Preference decides among what was found, and what is found is the
search provider's doing — one query set, one provider, six pages.

### The deployed demo is not currently blueprint-managed

`deploy/render-live.yaml` describes the public demo, and for now it
describes rather than governs it.

A rename was attempted and reverted (see v1.2.1). In between, the
blueprint's web service was replaced and then deleted, which left
`agentic-research-engine-live` running as a standalone service and the
blueprint instance managing only the Key Value store. Render cannot
adopt an existing unmanaged service into a blueprint by name, so that
instance reports a failed sync and will keep doing so.

The consequence to be aware of: **a change to the env vars in
`render-live.yaml` will not reach the running demo.** The committed
file is still the reviewable record of what the deployment should be,
and the deployment currently matches it, but the two are no longer
wired together. Re-establishing that means deleting the service and the
store and recreating both from the blueprint, with every credential
re-entered.

## Security

**DNS rebinding protection is proven against a test CA.** The wrong-SNI
refusal is demonstrated against a real TLS server with a real certificate
from an in-process CA on loopback. That proves the mechanism, not its
behaviour against the public certificate ecosystem.

**Prompt-injection defence is structural.** The extractor has no tools, and
a claim invented from a page references no evidence so it fails
resolution. No adversarial test has run against a live model.

**Secret scanning covers git, not runtime.** A key reaching a log via a
newly added field name would pass the redaction processor, which matches
on known key names.

**Fetched content is never scanned.** React escapes it on render; the
safety is React's.

## Product

**The public demo replays recorded runs by default.** Live research is a
separate deployment configuration.
