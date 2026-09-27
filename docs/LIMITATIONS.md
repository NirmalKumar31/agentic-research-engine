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

**The twelve-question benchmark has never been run.**

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

The *daily* run cap lives in memory, so a host that sleeps resets it on
every cold start. The provider account's own spend limit is the only
monetary control that survives a restart, and is the documented backstop
for any public live deployment.

**There is no durable or distributed quota**, which is why anonymous live
research is off by default on the public deployment. Enabling it safely
would need a persistent atomic quota store.

**Rate limiting keys on `X-Forwarded-For`**, which is spoofable.

**Input token estimation is `len // 4`** — conservative for prose, wrong
for code-heavy or non-Latin content.

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

**Remote verification is implemented and not yet proven against a live
endpoint.** The client, the endpoint container and the wire contract all
exist and are tested against mocked transports and a locally-run
instance of the same container. What has not happened is a hosted
acceptance run: no Hugging Face endpoint has been created, so no claim
about latency, cold-start behaviour or hosted parity is available. The
committed parity test refuses to run without an endpoint rather than
reporting a substitute.

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

**Replay still needs no credentials at all.**

**Provider prices are estimates** from a local table. They do not account
for cached input, long-context tiers, region or service tier.

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
