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

**Claim support is judged by a model, and in local mode by the same model
that wrote the report.** Nobody has compared it against human labels.

**Published figures come from single runs**, except the attribution
experiment, which has three repeats. There are no confidence intervals.

**The twelve-question benchmark has never been run.**

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

**Cross-attributed evidence has no query provenance.** Between 56% and 80%
of evidence across the three recorded runs answers a sub-question whose
queries never retrieved that source. The chain to the source is exact; the
chain back to a query exists only for the rest, and is recorded as absent
rather than guessed.

**Quote fidelity is measured against the copy held**, which for a
provider-supplied source is the search provider's text rather than an
independently fetched page. `content_origin` records which.

**PDFs cost an extra request.** Provider text for a PDF arrives with page
boundaries flattened, so likely PDFs are fetched directly. If that fetch
fails the source degrades to provider text — with a warning, but without
pages.

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

Unsupported and partially supported claims are removed before publication,
not rewritten. A published report therefore contains no claim that failed
the support check — which means "every published claim passed this
verifier", not "every published claim is true". The counts of what was
generated and removed are kept in the verification record.

## Operations

**Budget enforcement is per process.** Per-run request, token and cost
ceilings are checked before dispatch and hold. The *daily* cap lives in
memory, so a host that sleeps resets it on every cold start.

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

**Checkpoint recovery is untested under load.** No test resumes an
interrupted run.

**Base images are not pinned by digest.** `constraints.txt` pins Python
dependencies; `python:3.12-slim` and `node:24-slim` can move.

**No observability beyond logs.** No metrics endpoint, no trace export.

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
