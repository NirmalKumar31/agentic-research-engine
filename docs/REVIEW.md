# Standing weaknesses

A deliberately unflattering read of this repository. The point is to name
what is still wrong, not to justify it. Anything listed here is a real
gap; where a fix is known it is stated, and where the honest answer is
"not measured" it says so.

Reviewed against commit `9fa2e21`, with CI green on all eight jobs and
535 tests passing at 86% branch-aware coverage.

Two things are **still pending** at the time of writing and are marked as
such throughout rather than quietly omitted: the cloud validation
experiments (clean A/B, a current-code Luna run, a hybrid smoke) and the
public deployment.

---

## As an AI/LLM engineer

**No ground truth of any kind.** Every metric measures faithfulness to
retrieved sources. A report that cites five wrong pages perfectly scores
perfectly. Nothing in the evaluation framework can distinguish it from a
correct one. This is the single largest limitation and it is structural,
not a gap to be closed with another metric.

**Cross-attribution is the common case, and now measured rather than
excused.** Around three quarters of evidence answers a sub-question whose
queries never retrieved that source. The chain to the *source* is exact;
the chain back to a query exists for the rest. Narrowing it was tested:
on the six-source corpus, retrieved-only extraction cut evidence coverage
from 100% to 16.7% across three repeats, so production kept all-open. That
is a finding about *that corpus*, where each source was retrieved for a
mean of 1.17 sub-questions. It is not a general claim that all-open is
optimal, and a corpus whose sources are found by many queries would likely
narrow the gap.

**The quote-fidelity difference in that experiment is not causal.**
Narrowing appeared to improve exact-match fidelity (75.9% to 88.3%), and
the ranges did not overlap across three repeats. But n=3, the strategies
ran grouped rather than counterbalanced, and thermal state and execution
order are not ruled out. It is a lead, not a result.

**Entailment in local mode is a model grading its own work.** The verifier
is the same `qwen3:4b` that wrote the report. Nobody has checked it
against human labels. The earlier self-assessed support figures were
withdrawn rather than restated, because they came from a previous
evaluator generation and are not comparable to current metrics. The
measurement that would replace them is the frozen-corpus local-versus-cloud
comparison, which **has not been run**.

**Coverage sufficiency is arbitrary where it matters.** `_SUFFICIENT_RATIO
= 0.7`, and "two verified items from two distinct sources", are defensible
and untuned. Nothing establishes 0.7 over 0.6 — and the attribution result
shows how much that second threshold decides.

**Source quality scoring is an untuned heuristic.** Hand-weighted over
document type, rank, length and recency. It orders sources for extraction;
it is not a claim about correctness.

**Single search provider in practice.** Brave is implemented and unit
tested against recorded payloads, never against the live API. The
abstraction is proven at the type level, not the behaviour level.

**The planner fails often enough to matter.** One of the three committed
recordings is a run where planning failed and the engine researched the
question as a single dimension. Degrading rather than aborting is correct
behaviour, but the frequency is unmeasured and one-in-three in the
recordings is not reassuring.

---

## As an AI infrastructure engineer

**Budget enforcement is per process.** `UsageTracker` holds counts in
memory. Two workers, or one restart mid-run, and the spend ceiling is not
what it says. The per-run request, token and cost ceilings are real and
checked before dispatch; the *daily* cap is not, and cannot be on a host
that sleeps.

**There is no durable or distributed quota.** This is why anonymous live
research is disabled publicly rather than rate-limited. Enabling it safely
would need a persistent atomic quota store, which has deliberately not
been built: infrastructure whose only purpose is letting strangers spend
an API budget is not worth its complexity for a portfolio project.

**Input token estimation is `len // 4`.** Conservative in the common case,
wrong for code-heavy or non-Latin content, where it under-counts.

**No retry budget across a run.** Individual calls retry with backoff;
nothing caps total retries, so a flapping provider can multiply latency
inside the time limit without tripping anything.

**Checkpointing is written but never exercised under load.** SQLite
persistence exists; no test resumes an interrupted run from a checkpoint,
so "recoverable" is a property of the design rather than a demonstrated
one.

**The container is not pinned by digest.** `python:3.12-slim` and
`node:24-slim` are floating tags. `constraints.txt` pins Python
dependencies; the base images can move under a rebuild.

**No observability beyond logs.** Structured events exist; nothing
aggregates them. No metrics endpoint, no trace export.

**PDF fetching costs an extra request.** Provider-supplied text is reused
for HTML but deliberately not for PDFs, because it arrives with page
boundaries flattened and page-aware citation is the point. That is one
additional fetch per PDF source, and if it fails the source silently
degrades to provider text — with a warning, but still without pages.

---

## As a security engineer

**DNS rebinding is closed, and the proof is narrower than the claim.** The
connection goes to the validated address, the hostname rides in the `Host`
header and TLS SNI, each redirect hop is revalidated and re-pinned, a
target with no validated address fails closed, and failover never
re-resolves. The decisive property — that a *wrong* SNI is refused by
certificate verification — is tested against a real TLS server with a real
certificate. But that certificate comes from an in-process CA on loopback.
It proves the mechanism, not its behaviour against the public certificate
ecosystem.

**Prompt-injection defence is mostly structural, which is lucky.** The
extractor has no tools, so there is little to hijack, and a claim invented
from a page references no evidence and fails resolution. But no
adversarial test has run against a live model. Whether `qwen3:4b` can be
talked out of its task is unmeasured.

**Rate limiting keys on `X-Forwarded-For`.** Spoofable. Irrelevant while
live research is off; it would matter immediately if it were enabled.

**No authentication anywhere.** Correct for an anonymous replay demo.

**Secret scanning covers git, not runtime.** gitleaks protects the
repository, with a positive control that generates random planted
credentials each run so it cannot be silently disabled by an allowlist —
which is not hypothetical: an allowlist entry for fixed canary values
*did* disable it once, and the control caught it. Nothing stops a key
reaching a log if new code logs a settings object; the redaction processor
matches on key names and a novel field name would pass through.

**Fetched content is never scanned.** A malicious page is treated purely
as text. React escapes it on render — verified by rendering a
`<script>` payload through `react-dom/server` and confirming it emits
escaped text with no executable tag, with `dangerouslySetInnerHTML` absent
and a test enforcing its absence. The safety is React's; ours is not
adding an escape hatch.

---

## As a research/evaluation engineer

**The benchmark has twelve questions and has never been run.** They were
chosen to stress specific properties, and that intent is untested.

**One sample run is not a measurement.** The published local figures come
from single executions with a sampling model. The attribution experiment
is the only result here with repeats, and n=3 is still small.

**The clean frozen-corpus A/B has not been run.** The historical
55.6%/81.2% comparison came from a corpus whose source text was stripped,
which drove citation integrity to 0% in both arms. The loader now refuses
such input. Those numbers are retained only as a historical diagnostic and
must not be quoted as current.

**No current cloud measurement.** The last full cloud run predates the
bugs it exposed. A current-code Luna run and a hybrid smoke are both
pending.

**Category aggregation is implemented and unused.** `by_category` exists
in the report structure with no data behind it.

**The recordings are product artifacts, not benchmarks.** They demonstrate
the provenance chain. Their metrics should not be quoted as evaluation
results.

---

## As a skeptical hiring manager

**The public demo does not run anything.** It replays recorded executions.
That is defensible — an in-memory daily cap cannot bound an API quota on a
host that cold-starts, and the alternative was infrastructure whose only
job is letting strangers spend money — but a visitor cannot type their own
question, and no amount of framing changes that.

**"It works" still rests on a handful of runs.** The test suite is real
and well-targeted (535 passing, 86% branch coverage). Beyond it: one local
run, one stale cloud run, one attribution experiment with three repeats,
and three recordings.

**The commit history shows churn, including self-inflicted breakage.**
Several commits fix problems introduced one or two commits earlier — a
stream_mode regression, a secret scan disabled by its own test fixtures,
two CI failures caused by verifying in an environment that did not match
CI. Defensible as visible iteration; also evidence that changes landed
before they were fully checked.

**Eighteen minutes per run locally is not a usable tool.** It is a
demonstration.

**Scope is wide for one project.** Research engine, evaluation framework,
security layer, PDF pipeline, web app, deployment. Provenance and
evaluation are deep; the web app and deployment are thin.

**No user has ever used it.** No feedback, no failure reports from anyone
but its author, and no evidence the output is useful to someone who did
not build it.
