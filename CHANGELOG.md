# Changelog

Notable changes per release. Dates are UTC.

## v1.15.0 — 2026-10-03

**What this release does.** It makes source authority visible at
evidence-selection time, and it removes exact semantic restatements within
one answer slot.

**What is established about it.** Both changes are mutation-tested. Its
prompt-level improvements are **not separately causal-proven**. It does
**not** prove general real-model performance, and it does **not** turn the
engine into a universal research-answering system.

Selection, and a restatement published twice.

The v1.14 hosted run was the best measured — the question answered, two
complete comparison pairs, **7 claims at 35% evidence yield** against a
9.4% baseline. It also named the next bottleneck outright, because the
source accounting added in that release finally distinguished the two
kinds of uncited source:

    S3 reference.langchain.com 0.96 — not cited, 2 citable quotes extracted
    S4 docs.langchain.com      0.96 — not cited, 1 citable quote extracted
    S6 docs.langchain.com      0.91 — not cited, 2 citable quotes extracted

Five usable quotes from first-party documentation, none cited, while **six
of seven citations went to the two lowest-quality sources in the set**, at
0.62 and 0.64. Retrieval had done its job. Extraction had done its job.
Selection had not.

**The source kind now appears on the line where the evidence id is
chosen.** `build_package` has ranked by source authority for several
releases, and a sort is invisible to a model reading a list — it has no
way to know the order means anything. The classification existed the whole
time and reached nothing that made a decision. Each item now reads
`(supports, official docs 0.96)` or `(supports, blog 0.58)`, and the
synthesiser is told to prefer the more authoritative source **among quotes
that actually carry the claim** — authority does not override relevance,
and saying only the first half would trade one failure for another.

The source *id* is still withheld from that line. Showing it would invite
citing sources directly and reintroduce the ambiguity the design removes;
the kind of source is a different fact.

**A claim is a duplicate of one it merely rearranges.** The same run
published both of these, from one source, in one slot:

> LangChain components are the components on which LangGraph's
> orchestration layer is built.
> LangGraph is an orchestration layer built on LangChain components.

The engine noticed — it printed *"More than one published claim fills the
relationship slot; they may repeat each other"* — and published both.

Dedup had been exact-text only, and deliberately so: collapsing claims
that merely resemble each other is an editorial judgement made by a
threshold. This does not relax that. The test is **set equality on content
words within one answer slot** — not "these two are similar", but "these
two assert with exactly the same words, about the same part of the
answer". The two sides of a comparison pair are safe by construction,
since each names a different subject, rather than by a tuned cutoff.

Both changes are mutation-tested. Two mutants survived the first pass: one
exposed dead code — an explicit possessive strip that the word pattern and
length filter already handled, now deleted — and one was a badly
constructed mutant that left the asserted phrase intact, rebuilt and
caught.

### Release evidence

| | |
| --- | --- |
| Branched from | `dca1384` — the docs PR #19 merge, CI 9/9 success |
| Test totals | 2451 passed, 31 skipped, 20 deselected (Python); 92 passed (frontend) |
| Provider calls made for this release | **none** |
| Spend incurred for this release | **none** |

The release commit's own SHA and CI result are recorded in the pull
request and in the merge commit, not duplicated here: a SHA written into
the file it is a hash of cannot be correct.

**What was run:** the full gate set — `ruff check`, `ruff format --check`,
`mypy`, `mypy examples`, the complete non-integration test suite, the
frontend typecheck, unit tests and production build, a clean `npm ci`
against the regenerated lock, and the packaging checks: wheel and sdist
built, the wheel installed into an empty virtual environment, the CLI run
from it, the three bundled recordings confirmed present in the installed
package, and `pricing.toml` confirmed resolvable from outside the
repository.

The three container jobs — CLI-in-container, web image health probe, and
the real-NLI checkpoint job — were **not** run locally, because Docker is
not available in this environment. They run in CI on this commit, and the
result is the authority for them.

**What was not run:** no live research run, no paid provider request, no
new measurement of output quality. This release packages work already on
`main` and verified there; it changes no behaviour.

**Outstanding limitations, unchanged by this release:**

* Only the **query-writer** prompt mechanism has direct observable
  support — four searches instead of three, first-party documentation
  appearing where the previous run on the same question had none.
* The **three synthesiser prompt rules** remain non-causal observations.
  Reports improved while they shipped; that is not evidence that they
  caused it.
* Real-model evidence remains a **small, variable sample**. Run-to-run
  variance exceeded the effects being measured, which is why the
  measurement loop was stopped.
* **No claim is made that increasing the source count improves output.**
  The measurement contradicted it: across three runs on three different
  questions the engine extracted 85 evidence items and cited 8.
* **No broad claim is drawn from the LangChain/LangGraph example.** It is
  one question, used as a fixed before/after probe, not a benchmark.

---

## v1.14.0 — 2026-10-02

Where v1.12.0 released three streams that were chasing a bottleneck, this
one releases the two that found it. The engine now answers the question it
was failing to answer five runs earlier.

Measured on the same question throughout — *"langchain vs langgraph
differences?"*:

| | before v1.13 | after v1.13 |
| --- | --- | --- |
| Published claims | 3 | **6** |
| Complete comparison pairs | 0 | **2** |
| Answered the question? | no | **yes** |
| Evidence cited / extracted | 14% | **21%** |

[`docs/HOW-THIS-WAS-BUILT.md`](docs/HOW-THIS-WAS-BUILT.md), added in this
release, records how each cause was found, three proposed fixes that were
withdrawn when the measurement contradicted them, and two hypotheses that
turned out to be wrong.

### Slot labels and source accounting

Two self-inflicted losses, found in the first run that answered the
question.

The v1.13 hosted run on *"langchain vs langgraph differences?"* published
**two complete pairs and 6 claims at 20.7% yield** — the first time in five
runs on that question that `direct_contrast` was filled and the report did
not open by saying it had not answered. It also made two defects legible,
neither of which is an evidence failure.

**A claim may no longer aim at the contrast slot.** Two verified claims —
one defining each subject — were deleted for declaring `direct_contrast`
while naming a single subject. That slot is filled by the per-subject
claims the engine pairs; it is never a target. The label cost them twice:
they could not fill the slot they asked for, and carrying it stopped them
counting on the named axis they were actually about. The same report then
said *"The evidence did not establish how the subjects differ on purpose
and abstraction level"* — the axis those two claims were a complete pair
on. The synthesiser is now told this explicitly, and the loss is counted
and reported rather than absorbed into the generic exclusion line.

**An uncited source says which kind of uncited it was.** `retrieved, not
cited` covered two failures needing opposite fixes: a source whose text
yielded no citable quote is an extraction problem, one that yielded several
and was passed over is a selection problem. It now reads
*"retrieved, not cited — 2 citable quotes extracted"* or *"retrieved, no
citable quote could be extracted"*.

The case that prompted it: **AWS Prescriptive Guidance on LangChain and
LangGraph, 0.97, official docs** — a third-party authority discussing
*both* subjects, which first-party documentation structurally cannot, since
a vendor does not document its competitor. Retrieved, never cited, and
which of the two failures that was could not be determined from the report.

**A correction to v1.13.** Its off-subject diagnostic returned **zero** on
this run while six claims still failed at entailment 0.0004–0.0044. The
quotes were about the right subject and simply did not assert what the
claims asserted — a semantic mismatch inside one topic, which a lexical
check cannot see by design. The v1.13 hypothesis, that refused claims cite
off-subject quotes, was too coarse. The diagnostic is kept because it costs
nothing and reads zero honestly; it is not evidence of anything on this
run.

### Claim-to-evidence binding

Claim-to-evidence binding. The v1.12 retrieval fix worked and moved the
bottleneck one stage downstream.

That run was the first to retrieve good sources for this question —
`docs.langchain.com` at 0.98, `reference.langchain.com` at 0.96, two more
at 0.90, against a previous ceiling of 0.64 — and it published **three
claims and refused five**. Four of the five cited S1, GeeksforGeeks at
quality 0.64, the worst source in the set, at entailment **0.0013–0.0064**.
Those are not near misses: the quote does not carry the claim at all. The
three official documentation pages were never quoted.

The failed claims were all contrast-shaped, and S1 was the only page whose
*title* was a comparison. So the analyst reached for the source that looked
like the answer and attached evidence ids to sentences it had already
decided to write.

**The refusal now names which failure it was.** "The cited evidence did not
support them" describes an honest near miss and a retrofitted quote
identically, and the two need different fixes. A refused claim whose cited
quote is not even about its own subject now says so, and the count reaches
the reader: *"Of those, 4 cited a quote that was not about the claim's own
subject, which points at how the claim was assembled rather than at the
evidence."*

Diagnosis only, and deliberately incapable of more. It is computed after
the verdict and cannot withhold anything. A lexical check that could
withhold a claim would be a sixth deterministic gate, added in the release
that fixed two of them for refusing true claims — and a claim whose subject
is a pronoun would fail it while being genuinely entailed. There is a test
asserting it removes nothing.

**The evidence block says which axis can carry a contrast.** A comparison is
assembled from one claim per subject on a shared axis, which
`build_comparison_pairs` already does — but the synthesiser saw a flat list
and had to infer which axes had both sides. Official documentation is
single-subject, because a vendor does not document its competitor, so that
inference failed exactly when retrieval improved.

Each sub-question heading now states the subjects its evidence covers, and
each item carries the subject it names, like `[LangGraph]` or
`[LangChain + LangGraph]`. Where only one side is present the block says the
axis cannot carry a contrast and names the missing subject; where both are,
it asks for one claim per subject.

**And the synthesiser is told to write from the quote, not to the sentence.**

The binding diagnosis and the subject tagging are deterministic and
mutation-tested. The two prompt changes are not, and no offline test can be
— their effect needs a live run.

---

## v1.12.0 — 2026-10-02

Three merged streams released together, because none of them was tagged on
its own: output quality, claim yield with run telemetry, and retrieval.

The arc is worth stating plainly, since each stage only became visible
once the one before it was fixed. v1.9.0 answered *which sources get read*
and left the answer thin. v1.10 found the thinness was **arithmetic** — a
claim budget that ignored how broad the question was. v1.11 measured the
result and found **9.4% claim yield**, fixing two guards that were refusing
true claims. v1.12 found that with those fixed, what now caps the answer is
**retrieval**: the engine cannot cite documentation it never fetched.

### Retrieval, two honesty fixes, and a visible telemetry panel

Two honesty fixes, a visible telemetry panel, and the retrieval finding
that now caps the answer.

**Retrieval is the binding constraint, and the evidence is a pair of runs
on the same question six hours apart.** The first retrieved
`docs.langchain.com` three times and `reference.langchain.com` once, at
0.92–0.96. The second retrieved six commentary articles with a quality
ceiling of 0.64, and the report quoted a blog on LangGraph's state model
while the first-party documentation was never a candidate. Same prompt,
same question — so primary sources were being reached by luck. Nothing
downstream can repair that: authority-adjusted selection can only rank
what retrieval returned.

This is also the clearest argument yet against raising the source ceiling
from 6 to 12, which was asked for and again not done. Twelve commentary
articles is worse than six.

**The query writer is now told to ask the maintainer.** When a subject is
a named tool, library, framework, service or standard, one of its queries
should target that maintainer's own material. The counter-rule matters as
much: a technique, phenomenon or method has no maintainer, so the same
query spends a search to retrieve nothing.

It is a prompt rule, and the tests prove only that the rule is asked for.
A deterministic version was built first and withdrawn. Injecting
"*subject* official documentation" for every entity produced
"cost-sensitive learning official documentation" on a methods question;
narrowing it to comparisons, and then to single-token capitalised
subjects, still fired on `SMOTE` — an algorithm, not a product. Deciding
in code which subjects have a maintainer is not reliably solvable, which
is the conclusion this project already reached about first-party-docs
detection. The model writing the query can judge it, so it is asked to.
`site:` filters and domain allowlists both remain refused.

**A near miss no longer prints as though it cleared the bar.** A rejected
claim reported `best entailment 0.980 ... is below the 0.98 support
threshold`, a sentence that reads as though the engine cannot compare two
floats — `f"{0.97951:.3f}"` is `"0.980"`. Scores are now truncated rather
than rounded, so a sub-threshold value can never render at or above the
threshold, and the threshold prints at its own precision: at two places a
threshold of 0.985 showed as 0.98 and made a truthful 0.9840 read as
though it were above it.

**A relative clause is no longer a second proposition.** Splitting the
compound noun phrase "a graph of nodes and edges that supports flexible
data flow" at "and" left "edges that supports ...", and `supports` —
whose subject is the relative pronoun — was counted as a second clause's
verb, refusing a true claim. A predicate directly after a *mid-segment*
relative pronoun no longer counts. Restricted to non-initial pronouns, so
"and that requires a separate service" still fires. Same family as the
preposition fix in v1.11, which was narrower than the problem.

**The run-telemetry panel leads with its numbers.** It shipped collapsed,
and the next thing asked of it was for the figures it was already
carrying — a discoverability answer, not a capability one. Duration,
model calls, tokens, cost and sources read are now always visible, with
the full breakdown behind a disclosure. Cost is still marked as a floor
when the engine says it is one.

### Claim yield, and a run-telemetry panel

Three hosted v1.10 runs extracted **85 evidence items and cited 8 — 9.4%**,
publishing **8 of 22** generated claims. The sharpest of them asked *"what
is speculative decoding in LLM inference?"*, retrieved NVIDIA official docs
at 0.98 and three papers at 0.96–0.98 — the best source set the engine has
pulled — and published **two sentences**, with five of six sources never
cited.

That measurement withdrew the obvious fix. Raising the source ceiling from
6 to 12 would widen a funnel already discarding nine tenths of what enters
it, so it was not done. Where the 14 discarded claims died: 7 below the NLI
threshold, **3 on modality**, **2 on atomicity**, 2 on relevance.

**A bare assertion is no longer the weakest modality.** `modality_band`
scored text with no modal marker as band 0 — below every hedge — so flat
evidence was the weakest possible premise and *any* hedged claim exceeded
it. Three claims were refused for being **more cautious** than their own
source, among them "SQLite deployment can consist of copying the database
file" against evidence stating it flatly. The ladder is now hedge 1,
tendency 2, bare assertion 3, necessity 4.

`repair.py` had already found this and fixed it locally with its own
`_BARE_ASSERTION_LEVEL = 3`; `guards.py`, which actually gates publication,
never got the fix. A test now pins the two ladders together, because the
divergence was the defect rather than the band value.

`modality_guard` still exempts a bare claim, and deliberately. Hedge
*deletion* belongs to `hedge_guard`, which scopes it to the sentence that
carries the claim and classifies it unrepairable; reporting it here would
move it into `REPAIRABLE_GUARDS` and send an overclaim to be reworded
instead of refused. Removing the exemption regressed two semantic stress
fixtures.

**A plural noun no longer fakes a compound claim.** "stores", "reads" and
"writes" are noun and verb both, and a coordinated segment counted as its
own clause if *any* token looked like a predicate. So "...gives agents
short-term memory through checkpointers and long-term memory through
stores" was split at "and" and "stores" — the object of "through" — was
read as the second clause's verb. A true claim cited to official docs at
0.92 was refused as compound. A predicate directly after a preposition no
longer counts. Genuine two-clause and multi-sentence compounds still fire.

**Every claim is rendered once.** `summary_claims`, `key_findings`,
`sections[].claims` and `comparison_pairs` were four independent passes over
one claim pool with nothing reconciling them, so four claims printed eight
times — the Summary repeating the first comparison table verbatim and a
section repeating the second. The table keeps the duplicate: it carries the
axis label and attributes each sentence to a subject, so it says strictly
more, and a pair that lost a cell would stop being a contrast. A heading
left with nothing is dropped rather than printed empty.

**The Limitations section no longer contradicts the answer above it.** One
report stated "the evidence did not establish a named dimension along which
they differ" directly beneath a table of two named dimensions, because the
contract's generic `dimension` slot is a *fallback* axis and was counted as
an unmet requirement. It is now silent when a named axis carried a pair, and
still reported when none did. "More than one published claim fills the X
slot; they may repeat each other" described the two sides of a working
contrast as a defect — one claim per subject is what makes a pair a
contrast. The aim was removing false statements, not shortening the list;
real gaps still report.

**New: a run-telemetry panel on the report page.** Collapsed by default,
and assembled only from what the engine measured during the run: wall clock
and per-stage timing, logical calls against billable provider requests,
tokens in and out with cache and reasoning detail, estimated cost with the
engine's own "this is a floor" caveat, temperature, the model bound to each
role, retrieval and dedup counts, verification rates, the budget ceilings
the run was held to, and the commit, prompt and schema versions needed to
reproduce it.

No credential can reach it: `environment.capture` is an explicit allowlist
that reads no environment variables, asserted by test. Host detail the
server does collect — platform, processor, CPU count — is deliberately not
surfaced, also asserted, so broadening that stays a decision rather than a
drift. A field the server did not send is omitted rather than shown as
zero, because "0 cached tokens" and "this build did not record them" are
different facts.

### Output quality

Output quality. v1.9.0 fixed *which sources get read* and *what a question
is held to*, and left the answer thin. Two hosted runs made the reason
measurable, and none of it was a verification failure — the gates behaved
exactly as designed on an input that was starved before it reached them.

*"What types of vector index are used for similarity search?"* read six
accountable sources, extracted 15 evidence items, and published **one**
claim: "Tree-based indexes are one type of vector index."

*"langchain vs langgraph differences?"* extracted 30 items and published
five claims, **four of them about LangGraph alone**. It satisfied three
named axes and still reported that it had not answered — correctly, because
no axis carried a claim about both subjects. Every claim was true, supported
and relevant. Together they were not a comparison.

**The synthesiser is now asked for balance.** For a comparison it must write
one claim per subject on each axis, is told that three claims about one
subject is not a comparison however well evidenced, and is told what to do
when the evidence covers only one side — write about a different axis.

**The claim budget tracks the question, not the contract.** It was
`2 × slots`, and a `list` contract has two slots whatever it asks for — so a
six-dimension question asked for four claims while a comparison that
happened to gain three named axes asked for twelve. It is now the larger of
slot count and planned sub-questions. The arithmetic was also split into
`claims_requested` so a test can drive the real function: the first test
written for this fix reimplemented the calculation and passed with the fix
removed.

**Comparison subjects are matched through the run's own aliases.** A
published claim read "LangGraph's state persists throughout execution,
unlike LCEL's linear flow" — a genuine contrast, where LCEL is LangChain's
own expression language. It contains no "LangChain", so it counted as
speaking about one subject and no pair formed.

**The planner cannot outrun the source budget.** Coverage needs two distinct
sources per sub-question, so six dimensions against six sources cannot all
be covered — which is why a run said "only limited evidence was found" four
times. That was arithmetic, not retrieval. The planner is now capped at
`max_sources // 2`.

**Two source-classification gaps.** Medium publications on their own domains
(`pub.towardsai.net`, `ai.plainenglish.io`) classified as `other` at 0.50 —
*above* a blog at 0.45 — so three were selected for one comparison while the
blog penalty never applied. Project documentation on `.github.io` and
`.readthedocs.io` now reads as official docs. Tutorial and course sites were
deliberately **not** promoted: that tier is for reviewed encyclopaedic
sources, and rating variable-quality tutorial content above a vendor's own
page is guessing at quality rather than classifying provenance.

**The banner carries no maintained numbers.** A hard-coded count went stale,
and the range that replaced it went stale too once runs published seven
claims. It now states the properties that hold.

Known and unfixed: `python.langchain.com` classifies as `other`. Generic
first-party-docs detection on an arbitrary subdomain is not reliably
solvable, and hard-coding individual product domains to fake it was refused.

8/8 mutation checks caught, each by a named test.

## v1.9.0 — 2026-09-30

A quality release driven by hosted failures. Every defect below was
reproduced from a real run before it was fixed, and each is pinned by a
regression test. One paid run validated the result end to end.

**Answer-shape contracts are stable across wording.** The shape a run is
held to came only from the model, so two phrasings of one question got
different contracts: "the main causes of hallucination" was read as `list`
once and `definition` on a near-identical earlier phrasing, and "the context
window size of GPT-4 Turbo" became a `definition` — a figure checked against
a slot asking what the subject is. Explicit question wording now overrides
the model's label and `shape_source` records which decided. The override is
deliberately narrow: it returns nothing for anything that is not an
unambiguous question form, so the model's reading stands everywhere else.
`docs/ANSWER-SHAPES.md` states exactly which forms are corrected, which stay
model-decided, and every fallback.

**Comparison subjects are separated from context.** The analyst's `entities`
field is described to it as the concepts a question names — including the
setting — and coverage treated every one as a side to be contrasted. Asked
how retrieval-augmented generation differs from fine-tuning *for language
models*, a run published three supported, cited, relevant claims and then
reported that it had not answered, because nothing mentioned "language
models". `comparison_subjects` is now read from the question's wording, so a
setting cannot become a side.

**Yes/no causal questions have their own contract.** `candidate_drivers` used
to discharge `causal_evidence`, and the comment justifying it admitted the
alternative was unsafe for "does X cause Y?" with a relevance model as the
only backstop — the association-for-causation substitution the verification
layer exists to refuse. There are now two shapes with no route between them:
`causal` for a yes/no test, whose contract contains no drivers slot at all,
and `causal_drivers` for "what causes X?", where the drivers are the answer.
Association language, mixed sentences, and sentences that name causation in
order to deny it are all refused.

**Comparisons are answered by structure, not by new prose.** A contrast
asserts two things and the atomicity guard refuses compound claims —
correctly. Across nine runs the synthesiser produced one `direct_contrast`
claim and atomicity refused it. Rather than weaken the guard or let repair
delete the unsupported half, a comparison is now assembled from verified
side claims meeting on a *named* axis and rendered as a table that says no
sentence was written to join the cells. `relationship` may stand in only
when its kind explains why a contrast is inappropriate; "both are used with
language models" no longer qualifies. No canonical slot declares a static
alternative any more.

**Query generation is shorter and more direct.** The prompt instructed the
model to "include the specific technical terms an authoritative page would
use", with no length bound. It obeyed, producing eleven-word jargon stacks
that only research papers matched — so a question about the causes of
overfitting was searched as a literature review and read a tweet, a
newsletter and two papers on double descent. The instruction is gone; the
prompt asks for the question's own vocabulary, bounds length, and protects
proper names, versions and dates from that bound. Query style now depends on
the answer shape. Measured on the paid run: median query length fell from 11
words to 5.

**Retrieval candidates are preserved.** A run read six pages from 44
candidates and the other 38 were unrecoverable — `PERSIST_RUNS=false`, and
the search stage streamed only counts — so "was there a better source in the
pool?" could not be answered. Each round now records a bounded, sanitised
manifest of every query and every candidate with its score, class,
authority, the adjustment separately from the total, every sub-question it
serves, the decision, the reason and the fetch outcome. URLs are reduced to
scheme, host and path; no credentials, headers, raw payloads, page bodies or
filesystem paths.

**Source selection is authority-aware and allocated by slot.** Selection
sorted by relevance banded to tenths and consulted authority only within a
band, so a tweet at 0.87 beat a primary source at 0.84 and the authority
term never applied; and a global top-N let one sub-question consume every
slot. Selection now allocates across sub-questions for representation, then
goes best-first. The adjustment is capped so a relevance gap wider than 0.35
cannot be overturned by source class. `SourceType.SOCIAL` and `REFERENCE`
were added because social posts fell through to `other`, whose base quality
is *above* a blog's.

**Coverage requires relevance, not just an exact quote.** `quote_verified`
is a provenance property, and coverage treated it as sufficient: two
exact-match quotes across two sources marked a sub-question covered,
whatever they were about. An item must now also discuss the sub-question's
terms. The check is an explicitly bounded negative prefilter, and the four
levels are named so they cannot be confused — `lexically_plausible`,
`evidence_relevant`, `claim_relevant`, `slot_satisfied`. Aliases are derived
from the question's own text, never a global synonym table.

**The duplicate start event is gone.** The endpoint announced a run before
the graph was built and the runner announced it again; both were forwarded,
so the page showed "Starting research" twice, which reads as a restart.
Suppressed in the web transport only — a CLI caller has no transport-level
start.

**Every sub-question gets a query before any gets a second.** Found by the
paid run: the model proposed two queries each for the first three of five
sub-questions and none for the other two, and assembly stopped in arrival
order, so two sub-questions were never searched. Assembly is now
breadth-first in planner priority order, a sub-question the model omits gets
a query from its own text, and a budget genuinely too small names and logs
what it could not cover.

**Gaps carry explicit causes.** A gap now distinguishes no query issued, no
suitable source found, a selected candidate whose fetch failed, a retrieved
source that did not discuss the sub-question, and insufficient evidence.
Five causes need five different fixes, and they were previously
indistinguishable.

**Paid validation.** One authorised run on `94368bb7` re-asked the question
that had published nothing: *"What are the main causes of overfitting in
machine learning?"* It published **5 claims of 6 generated** and reported the
question answered, against a baseline of 0 of 1 and 1-of-5 coverage. 147.5s
of 240, $0.008809 of a $0.05 ceiling, 14 of 30 provider requests, 6 of 8
Tavily credits, all 41 candidates preserved, one visible start event, one
attempt with no retry. Evidence:
`examples/live-validation/question-shapes/overfitting-20260930-221039/` and
`docs/RELEASE-EVIDENCE-v1.9.0.md`.

**What remains offline-validated only.** Three authorised paid runs were
made in total: the `causal_drivers` question above, a structured comparison
(*"how does retrieval-augmented generation differ from fine-tuning?"* — 2
subjects, 2 complete pairs on named axes, 7 claims, answered), and a yes/no
causal test (*"does label noise cause overfitting?"* — correctly **not**
answered, with the core slot refusing an association claim). So comparison
and the yes/no contract are now live-validated too.

The comparison run found one more defect: it reported the question answered
with two complete pairs and shipped a report with **no contrast table**,
because `runner._render` — the renderer producing the markdown the web
result carries — was never given the pairs, while `finalize` was. Both now
share one rehydration function. That fix, and the breadth-first query fix,
both postdate the runs that revealed them and are structurally verified
only. Relationship-kind discharge is also CI-verified only: run 2 answered
by pairs, so the escape hatch was never exercised. The breadth-first
query fix landed after that run, so it too is structurally tested rather
than live-validated — its guarantee holds regardless, because the node
injects missing queries itself rather than relying on the model. One
successful live question shows the path works, not that research quality is
general.

### Also in this release


Four of the nine answer shapes were unreachable, a stylesheet was never
imported, and a run could not say that a question named something that
does not exist.

**Four answer shapes could not be reached.** `OutputFormat` offered the
analyst five labels while `QuestionType` defined nine, so causal,
numeric, list and multi-part questions had no label to be classified
as. Nothing failed: they were classified `overview` and held to a
*definition* contract, whose core slot asks what the subject is. A
hosted run asked for the main causes of hallucination in language
models, produced four cause claims, and published **0 of 4** — every
one withheld against requirements the question never had. The analyst
was also given no guidance for choosing among the labels it did have,
so `overview` acted as a default rather than as the narrowest shape.

Two call-site gaps in the same area, found while fixing it:
`contract_from_analysis` passed neither `dimensions` nor `parts`, so
`build_contract`'s named-dimension branch was unreachable in
production and tested only where it is declared, and `synthesis` would
have produced an *unusable* contract — which refuses every claim — for
any multi-part question. Both fields now reach the contract, and a
`synthesis` label with no named parts falls back rather than silencing
the run.

**A causal question has two honest readings.** "Does X cause Y?" wants
evidence of causation; "What causes Y?" wants the drivers, and for
that reading the drivers *are* the answer. `candidate_drivers` now
discharges `causal_evidence`, for the reason `relationship` discharges
`direct_contrast`. Stated cost: naming a plausible driver discharges
the core slot of a strict causal question without establishing
causation, with the relevance judgement as the backstop.

**A question can now say that its subject does not exist.** Asked for
the difference between GPT-6 Astra and "GPT-5.5 Sol", the engine
retrieved sources and published nothing. That was correct — the second
model does not exist, and a comparison requires every named subject to
be spoken about — but the page reported "0 of 5 requirements covered",
which describes the engine rather than the question. Coverage now
checks the contract's entities against the retrieved *sources*, not
the published claims, and leads the limitations with the subject no
source mentions. A run that retrieved nothing at all is excluded: that
is a different problem, and blaming the question for it would be
wrong.

**A stylesheet was never imported.** `web/src/index.css` held four rule
blocks and `main.tsx` imports `styles.css` alone, so none of them ever
reached the bundle. The capability chips ran together into
"5 sources2 verified claims" because `.card__caps` was never a flex
container, and the withheld-evidence fallback — the screen a visitor
sees whenever the gate publishes nothing — had been rendering unstyled
since it shipped. The rules are merged into the imported stylesheet
and the orphan is deleted; `--amber`, which it referenced and which
does not exist, is now the `--warn` pair the adjacent notice uses.

**The hourly run cap was the operator's limit, not a visitor's.**
`runs_per_ip_per_hour` was 2 and also the maximum configuration could
request, so `DEMO_RUNS_PER_HOUR` could not raise it. Now 10. The daily
allowance, derived from real provider spend, is unchanged and remains
the cap that bounds cost.

**The absent subject is now shown where the reader is looking.** The
first fix put it in the report's limitations, but "0 of 5 requirements
covered" is rendered in the contract panel, and that panel derives slot
status on the client from the published claims — so it could not know a
subject was missing from the sources, because the browser never
receives source text. The engine's own assessment is now carried out
of the node in `answer_coverage`, through the result payload, to the
panel, which leads with the absent subject and says an unfilled
requirement is the correct outcome in that case. `AnswerCoverage.to_dict`
had no caller in `src/` before this, so it was serialising a value
nothing read.

Recordings predating the new key are backfilled with null rather than
skipped. The schema-version gate skips a recording wholesale on
mismatch, which is right for a change in how a payload should be read
and wrong for an added nullable field: bumping it would have taken all
three demo recordings down until each was re-recorded with a paid run.
The canonical-shape test caught this.

Guards added for the class of defect above: every `OutputFormat` maps
to a `QuestionType`, every `QuestionType` is reachable from some
`OutputFormat`, and the schema's `Literal` offers exactly the members
that exist. Each of the three would have failed before this change.

## v1.8.0 — 2026-09-30

Comparisons can be answered. They could not before, for a structural
reason that took nine runs and four attempts to locate correctly.

Asked how a large language model differs from a neural network, the
engine produced exactly **one** `direct_contrast` claim across nine
hosted runs, and the atomicity guard refused it — rightly: a contrast
asserts two things and every other guard reasons about "the sentence
that supports this claim".

So the synthesiser did the only thing available and wrote claims about
each subject, declaring `dimension`. **The relevance judge then
refused each one** for "describing neural networks, not how LLMs
differ" — applying the *report's* question to a single claim, which no
atomic claim can answer. Comparisons published nothing, and the cause
was a gate enforcing a report-level question at the claim level.

Two changes, at the two levels the question lives at.

**Claim level.** The judge no longer vetoes a claim that structurally
fills an *optional* part of the answer. It keeps full authority over
core slots and over claims declaring no slot or an unrecognised one.
A prompt change asking it to judge against the listed parts was tried
first, in v1.6.1, and **measurably failed** — the very next run
refused three `dimension` claims with the same reasoning — so the
authority is narrowed in code rather than requested in a prompt.

**Report level.** `assess_coverage` will not call a comparison
answered unless the published claims *between them* speak about every
subject the question named.

That second gate is what still stops the original defect, and the
defect was always report-level: five supported, cited claims about one
of two subjects, published as an answer to how they differ. A
one-sided report now publishes its claims and states plainly that it
did not answer. An off-topic claim does not get through either — the
structural check still requires the claim or its evidence to mention
one of the contract's subjects.

**Honest about reach.** Replayed against all ten captured runs, the
coverage half changes nothing: every run that published claims had
already answered by another route, and the rest published nothing. Its
effect depends on the claim-level narrowing, which no captured run
exercised. Unverified on the hosted path.

Also recorded: the narrowing was initially tested only where the
predicate is defined, and removing it from the production path broke
no test. That is the seventh instance in this project of something
tested where it is declared rather than where it is used. It is now
driven through `_check_entailment` with a scripted judge.

No change on the frozen adversarial set.

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

This release also carried two corrections that were labelled
Unreleased when they were written and shipped in this tag. They are
recorded here rather than left under a heading that stopped being
true, because a changelog that mislabels what shipped is the same
class of defect as a report that overstates what it verified.

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
