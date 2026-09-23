# Standing weaknesses

A deliberately unflattering read of this repository from five angles. The
point is to name what is still wrong, not to justify it. Anything listed
here is a real gap; where a fix is known it is stated, and where the honest
answer is "not measured" it says so.

Last reviewed 2026-09-23.

---

## As an AI/LLM engineer

**Cross-attribution is the common case, not the edge case.** 78% of
evidence in the measured run answers a sub-question whose queries never
retrieved that source. The provenance chain to the *source* holds, but the
evidence→query link — advertised as part of the design — exists for barely
a fifth of items. The cause is structural: every source is shown every open
sub-question. Narrowing that would trade recall for provenance and has not
been measured either way.

**Quote fidelity is 74% and nothing improves it.** A quarter of extracted
quotes are reworded or unlocatable, and the only handling is exclusion.
There is no re-prompt on a failed quote, no "find the nearest real span"
repair. The result is honest but lossy: roughly a quarter of extraction
spend is discarded.

**Entailment in local mode is a model grading its own work.** The verifier
role is the same `qwen3:4b` that wrote the report. A support figure
produced that way is close to meaningless, and the hybrid split exists
mostly to avoid it. Nobody has checked the verifier against human labels.

**Coverage sufficiency is arbitrary where it matters.** `_SUFFICIENT_RATIO
= 0.7` and "two verified items from two distinct sources" are defensible
and untuned. Nothing establishes that 0.7 is better than 0.6.

**Prompts are unversioned.** They materially change behaviour and live as
module constants with no identifier in run metadata, so two runs weeks
apart are not comparable and nothing records why.

**Single search provider in practice.** Brave is implemented and unit
tested against recorded payloads, never against the live API. The
abstraction is proven at the type level, not the behaviour level.

---

## As an AI infrastructure engineer

**Budget enforcement is per process.** `UsageTracker` holds counts in
memory. Two workers, or one restart mid-run, and the spend ceiling is not
what it says. For the free single-instance demo that is fine; the moment it
scales it is wrong, and the code does not stop you scaling it.

**Rate limiting has the same shape.** In-memory windows, so per-IP and
daily caps are per-replica. The real bound would be the per-run spend
ceiling, which is also per process.

**Input token estimation is `len // 4`.** The pre-dispatch cost check is
therefore approximate. It is conservative in the common case and will be
wrong for code-heavy or non-Latin content, where it under-counts and the
ceiling is discovered late.

**No retry budget across a run.** Individual calls retry with backoff, but
nothing caps total retries, so a flapping provider can multiply latency
within the time limit without tripping anything.

**Checkpointing is written but never exercised in anger.** SQLite
persistence exists; no test resumes an interrupted run from a checkpoint,
so "recoverable" is a property of the design rather than a demonstrated
one.

**The container is not pinned by digest.** `python:3.12-slim` and
`node:24-slim` are floating tags. `constraints.txt` pins Python
dependencies but the base images can move under a rebuild.

**No observability beyond logs.** Structured events exist; nothing
aggregates them. There is no metrics endpoint, no trace export, and the
LangSmith hook mentioned in the original design was never wired.

---

## As a security engineer

**DNS rebinding is open.** Addresses are validated then a *new* connection
is made by hostname, so a name that resolved public at check time can
resolve private at connect time. Closing it means pinning the validated IP
into the transport. This is the most substantive remaining hole and it is
documented rather than fixed.

**Prompt-injection defence is mostly structural, which is lucky.** The
extractor has no tools, so there is little to hijack. But no adversarial
test has been run against a *live* model — the injection suite checks
boundaries and plumbing, not whether a model obeys a page. Whether
`qwen3:4b` can be talked out of its task is unmeasured.

**The SPA catch-all serves index.html for every unmatched path**, which
also means `/docs` returns 200 in demo mode. Harmless, but it means "docs
disabled" is asserted by content rather than by routing, and a future route
added carelessly inherits the same fallthrough.

**Rate limiting keys on `X-Forwarded-For`.** Spoofable. The global daily
cap is the real defence and it is a blunt one — one abusive client can
exhaust the day for everyone.

**No authentication anywhere.** Correct for an anonymous demo, and it means
the only thing between the internet and the API key is the limit set.

**Secret scanning covers git, not runtime.** gitleaks protects the
repository. Nothing stops a key reaching a log if new code logs a settings
object; the redaction processor matches on key *names* and a novel field
name would pass through.

**Fetched content is never scanned.** A malicious page is treated purely as
text. That is the right scope, but it means a stored-XSS payload inside a
quote reaches the frontend, where React escapes it — the safety is React's,
not ours, and nothing tests it.

---

## As a research/evaluation engineer

**No ground truth of any kind.** Every metric measures internal
consistency. A report faithfully citing five wrong pages is
indistinguishable from a correct one, and nothing in the evaluation
framework can tell them apart. This is stated in the docs and is still the
single largest limitation.

**The benchmark has twelve questions and has never been run.** They were
chosen to stress specific properties, and that intent is untested — it is
plausible the sparse-evidence question is not actually sparse and the
contradiction questions do not actually contradict.

**One sample run is not a measurement.** Every published figure comes from
a single execution with a sampling model. There are no repeats, no
variance, no confidence interval. "74% quote fidelity" is one draw.

**Sampled and exhaustive support are labelled but not comparable.** The
committed figure is sampled (10 of 17). Nothing establishes how far a
sample of 10 diverges from the full set.

**Category aggregation is implemented and unused.** `by_category` exists in
the report structure with no data behind it.

**A/B harness is unexercised on a real comparison.** It is tested against
fakes and has never compared two genuine models, so the thing it was built
for has not happened.

---

## As a skeptical hiring manager

**"It works" rests on one local run and a test suite.** 359 tests are real
and well-targeted, but the end-to-end evidence is a single 18-minute run on
the author's laptop. No cloud run, no repeated runs, no deployed URL.

**The most impressive claim is the least demonstrated.** Evidence-level
provenance is genuinely well built — and the only place a visitor can *see*
it is a screenshot-free README, because the demo is not deployed.

**Two of the stated goals are incomplete.** The OpenAI validation never
happened (no key was available) and the site is not live. Both are
correctly reported as blocked rather than fudged, but incomplete is
incomplete.

**Eighteen minutes per run is not a usable tool.** Locally it is a
demonstration, not something anyone would reach for. The cloud path that
would make it usable is the untested one.

**The commit history shows churn.** Several commits fix problems introduced
two commits earlier — a stream_mode regression, a size-cap bug, a secret
scan tripping on its own fixture. Defensible as visible iteration; also
evidence that changes landed before they were fully thought through.

**Scope is wide for one project.** Research engine, evaluation framework,
security layer, PDF pipeline, web app and deployment config. Breadth like
that invites the question of whether any one part is deep enough, and the
honest answer is that provenance and evaluation are, while the web app and
deployment are thin.

**No user has ever used it.** No feedback, no failure reports from anyone
but its author, and no evidence the output is useful to someone who did not
build it.
