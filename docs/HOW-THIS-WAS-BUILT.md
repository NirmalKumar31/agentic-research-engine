# How this was built

A record of how the engine's output quality was diagnosed and fixed, written
because the interesting part is not the fixes — it is which measurement
justified each one, and which proposed fixes were withdrawn when the
measurement said no.

Every number below came from a real run. Where a claim is not measured, it
says so.

---

## The short version

The engine published thin answers. Five releases chased that through five
*different* causes, each of which only became visible once the previous one
was fixed:

| Release | What was limiting the answer | How it was found |
| --- | --- | --- |
| v1.10 | **Arithmetic** — the claim budget ignored question breadth | A 6-dimension question asked for 4 claims and published 1 |
| v1.11 | **Two guards refusing true claims** | 3 claims refused for being *more cautious* than their source |
| v1.12 | **Retrieval** — primary sources reached by luck | Same question 6h apart: 4 official docs, then 0 |
| v1.13 | **Claim-to-evidence binding** | 4 of 5 refusals cited the worst source at 0.001–0.004 entailment |
| v1.14 | **A slot label** | 2 verified claims deleted for declaring the wrong slot |

Measured on the same question throughout — *"langchain vs langgraph
differences?"* — so the before/after is comparable:

| | v1.9.0 | v1.10 | v1.11 | v1.12 | v1.13 |
| --- | --- | --- | --- | --- | --- |
| Published claims | 5 | 4 | 4 | 3 | **6** |
| Complete comparison pairs | 0 | 2 | 1 | 0 | **2** |
| Answered the question? | no | partly | yes | **no** | **yes** |
| Evidence cited / extracted | — | 17% | 14% | 14% | **21%** |

v1.12 went *backwards on the answer* while succeeding at what it set out to
do. That is the most useful row in the table, and the next section explains
why.

---

## The chain, stage by stage

### Thin answers were arithmetic, not quality

The first instinct was that the model wrote badly. The measurement said
otherwise: the claim budget was `2 x required slots`, and a `list` contract
has two slots no matter how broad the question is. So a question with six
dimensions asked for four claims and published one, while a comparison that
happened to gain three named axes asked for twelve.

Coverage needed two distinct sources per sub-question, against a six-source
ceiling — so six sub-questions needed twelve source-slots and could not
possibly be covered. "Only limited evidence was found" was unavoidable
before any model ran.

### Two guards were refusing true claims

With the budget fixed, three live runs extracted **85 evidence items and
cited 8 — 9.4%**, publishing 8 of 22 generated claims. Reading the
rejections found two deterministic guards failing in the *safe* direction:

**Modality.** A bare assertion scored band 0 — *below every hedge* — so flat
evidence was the weakest possible premise and any hedged claim exceeded it.
A claim reporting "SQLite deployment **can** consist of copying the database
file" was refused against evidence stating it flatly. Three claims in one
run were refused for being **more cautious** than their own source.

The repair module had already found this and fixed it locally with its own
`_BARE_ASSERTION_LEVEL = 3`. The module that actually gates publication
never got the fix. A test now pins the two ladders together, because the
divergence was the defect — not the band value.

**Atomicity.** `"stores"`, `"reads"` and `"writes"` are both plural nouns and
third-person verbs, and a coordinated segment counted as its own clause if
*any* token looked like a predicate. So

> ...gives agents short-term memory through checkpointers **and** long-term
> memory through stores

was split at "and", and `stores` — the object of `through` — was counted as
the second clause's verb. A true claim cited to official documentation at
0.92 was refused as compound.

### Retrieval was succeeding by luck

Then the same question, six hours apart, with no code change between:

| | run A | run B |
| --- | --- | --- |
| Official docs retrieved | **4** (0.92–0.96) | **0** |
| Quality ceiling | 0.96 | **0.64** |

Run B cited a blog on LangGraph's state model while the documentation that
states it directly was never a candidate. Nothing downstream can repair
that: authority-adjusted selection can only rank what retrieval returned.

The cause was in the query prompt. It said *"prefer the question's own
vocabulary"* — correct, because a query full of specialist terms returns
only specialist papers — but that is also exactly what commentary about a
subject is written to rank for, and nothing pulled the other way. The fix
tells the query writer that a named tool, library or standard has a
maintainer worth asking directly, and that a technique does not.

Result: **4 of 6 sources first-party, ceiling 0.98.**

### Fixing retrieval moved the bottleneck, it did not remove it

The run with the best sources the engine had ever retrieved published
**three claims and refused five**. Four of the five cited the *worst* source
in the set — a listicle at 0.64 — at entailment **0.0013–0.0064**. Those are
not near misses; the quote does not carry the claim at all. The three
official documentation pages were never quoted.

All five failures were contrast-shaped, and the listicle was the only page
whose *title* was a comparison. The analyst was writing the sentence first
and attaching an evidence id to it afterwards.

A structural cause sat underneath: **official documentation is
single-subject, because a vendor does not document its competitor.** A
comparison therefore has to be *assembled* from each side's own docs — which
the engine already does — but the synthesiser saw a flat evidence list and
had to infer which axes had both sides. That inference failed precisely when
retrieval improved.

So the evidence block now states which subjects each axis covers and tags
every item with the subject it names. The next run filled the contrast slot
for the first time in five attempts, with two complete pairs and the highest
claim yield measured.

---

## Three fixes that were built and withdrawn

This section is the point of the document.

### Raising the source ceiling

Requested twice: 6 sources to 12. Not done, because the measurement
contradicts it. Across three runs the engine extracted **85 evidence items
and cited 8**. The sharpest case retrieved NVIDIA official documentation at
0.98 plus three papers at 0.96–0.98 — the best source set it had ever pulled
— and published **two sentences**, with five of six sources uncited.

Widening a funnel that discards nine tenths of what enters it adds
"retrieved, not cited" rows and nothing else. Twelve commentary articles is
worse than six.

### A deterministic primary-source query

The retrieval fix was first built deterministically: inject
`"<subject> official documentation"` for every entity the question named.

It produced **`"cost-sensitive learning official documentation"`** on a
fraud-detection methods question — a technique has no maintainer, so the
query spends a search to retrieve nothing. Caught by an existing dedup test,
not by a new one.

Narrowed to comparisons only: still fired. Narrowed again to single-token
capitalised subjects: still fired on **`SMOTE`**, an algorithm rather than a
product.

Deciding in code which subjects have a maintainer is not reliably solvable —
the same conclusion this project had already reached about first-party-docs
detection. The judgement moved to the model that writes the query, where it
belongs. `site:` filters and domain allowlists both remain refused.

### A lexical gate on claim-to-evidence binding

The binding diagnosis was first proposed as a check *before* entailment, able
to withhold a claim. Withdrawn before shipping: it would have been a sixth
deterministic gate, added in the same release that fixed two of them for
refusing true claims — and a claim whose subject is a pronoun would fail a
lexical check while being genuinely entailed.

It ships as diagnosis only. It is computed after the verdict and cannot
withhold anything, and a test asserts that it removes nothing.

---

## Two hypotheses that were wrong

**"Hedge deletion is an unguarded hole."** Asserted while reading the
modality guard, before reading the module next to it. `hedge_guard` already
caught deletion, scoped to the sentence that carries the claim, using the
same overlap logic that was about to be rewritten. The band-0 exemption was
a documented division of labour, not an oversight — and removing it
regressed two semantic stress fixtures.

**"Refused claims cite off-subject quotes."** The diagnostic built for this
returned **zero** on the next run while six claims failed at entailment
0.0004–0.0044. The quotes were about the right subject and simply did not
assert what the claims asserted — a semantic mismatch inside one topic,
which a lexical check cannot see by design. The hypothesis was too coarse.
The diagnostic is kept because it costs nothing and reads zero honestly; a
zero from it is not evidence that binding is healthy.

---

## The defect class this project keeps producing

A value computed correctly and never handed to the thing that needed it.
The commit history names each instance. Two examples:

- the comparison pairs were assembled, serialised, and then not passed to
  one of the two renderers — found by a paid run, after a test had asserted
  only the other call site;
- the modality fix existed in the repair module and not in the module that
  gates publication.

The tell is the same every time: a test written where the value is
*declared* rather than where it *runs*. It recurred inside the fix for
itself — two mutants survived the tests for the slot-label counter because
every one of them drove the consumer with the count handed in, never the
producer.

---

## How changes are verified

**Mutation testing as the acceptance gate.** Every new guard or counter is
checked by breaking it deliberately and confirming a named test fails. A
surviving mutant is treated as a missing test, except where the mutant
itself was wrong — which has happened, and is recorded rather than quietly
rerun.

**Pass conditions written before the run.** Paid runs are rare, so each one
has its success criterion recorded in advance:

> `>=1 *.langchain.com` source, and a quality ceiling above 0.64

That one passed — 4 first-party sources, ceiling 0.98. The next, for the
claim-binding work, was pre-committed as `off-subject claims <= 1`, and it
passed **vacuously**: the metric read zero because it was measuring the
wrong thing. Recorded as a vacuous pass rather than counted as evidence.

**Local gates match CI exactly.** `ruff check src tests examples`, not
`ruff check .` — the broader command flags a docs file CI does not gate, and
a local invocation that differs from CI's hides real defects in both
directions.

---

## What is not proven

The two most recent releases have not been validated by a live run. Of the
last four fixes, three were prompt changes: those moved output the most and
are the ones no test covers. A green CI run says less about them than it
does about the deterministic work.

The open question is whether a contrast pair now forms on an axis whose two
claims were previously discarded for carrying the wrong slot label. It needs
one run to answer, and the pass condition is already written down.
