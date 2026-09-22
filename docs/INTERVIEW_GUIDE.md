# Interview guide

How to present this project, and accurate answers to the questions it invites.

Every number quoted here is measured. If you are asked something this document
does not cover, the honest answer — "I did not measure that" — is worth more
than a guess, and interviewers can tell the difference.

---

## The 60-second version

> It is a deep-research system built on LangGraph. You give it a research
> question; it decomposes that into sub-questions, runs web searches in
> parallel, deduplicates the results before fetching anything, extracts
> evidence as verbatim quotes checked against the source text, assesses its
> own coverage, and loops for another round if there are gaps — bounded by
> explicit budgets. Then it synthesises a report where every claim carries
> citations, and a verification pass checks that each citation resolves to a
> source the run actually retrieved.
>
> The two parts I would point at are the evidence provenance — you can trace
> any sentence back through claim, evidence, source, query, to the
> sub-question that motivated it — and the model routing, where code requests
> a model by role and configuration decides whether that runs on OpenAI or a
> local Ollama model.

If they only ask one follow-up it is usually "why LangGraph". Have that ready.

---

## Architecture questions

### "Why LangGraph instead of a normal Python pipeline?"

Three things the graph gives that a function pipeline does not, and I needed
all three:

1. **A cyclic, state-carrying loop.** The coverage critic can send the run
   back for another research round. That is a cycle with accumulating state
   across iterations. Expressible with a `while` loop, but then I own the
   state merging.
2. **Fan-out with typed fan-in.** `Send` dispatches one worker per query;
   `defer=True` runs the next node only after all of them settle. With
   `asyncio.gather` I would hand-roll the barrier and the result merging.
3. **Reducers as a concurrency contract.** Eight workers writing to one
   `evidence` channel merge through `operator.add` — declared once on the
   channel rather than coordinated at each call site.

Plus checkpointing: a run that dies in synthesis has already paid for all the
retrieval, and the checkpoint preserves it.

**The honest caveat**, which lands well: for a purely linear pipeline
LangGraph would be overhead. It earns its place here because of the loop and
the fan-out.

### "Why isn't this just RAG?"

RAG is retrieve-then-generate: embed a query, pull top-k chunks from a
pre-built index, stuff them in a prompt. Four differences:

| | RAG | This |
|---|---|---|
| Corpus | Pre-indexed, fixed | The live web, discovered per run |
| Retrieval | One shot | Iterative, gap-driven |
| Query | The user's, embedded | Decomposed into sub-questions, rewritten as searches |
| Verification | Usually none | Quote checking plus citation entailment |

The structural difference is the feedback loop. RAG retrieves once and
generates. This assesses what it gathered, identifies specific gaps, writes
targeted follow-up questions, and searches again — bounded.

There is no vector store, and that is deliberate: evidence is tens of items
scoped to a single run, not a corpus. A dict lookup is the correct data
structure. Adding a vector database would have been resume-driven
development.

### "How do parallel workers update shared state safely?"

They never mutate shared state. Each returns a partial update, and LangGraph
folds it into the channel through that channel's reducer:

```python
evidence: Annotated[list[EvidenceItem], operator.add]
sources:  Annotated[list[SourceDocument], merge_sources]
```

`merge_sources` is custom because a source is written twice — as a stub when
its URL is deduplicated, then filled in by a fetch worker — so it merges by id
with last-write-wins. That also makes a node retry idempotent.

Identifiers are never allocated inside parallel workers. Source ids are
assigned in the single-writer deduplication barrier; evidence ids are
`f"{source_id}-e{n}"`, unique without coordination because exactly one worker
owns a source.

### "Walk me through a request."

Question → `analyze_query` (intent, entities, time-sensitivity) →
`plan_research` (4–6 sub-questions) → `generate_queries` (search strings,
deduplicated against everything already issued) → **fan out** one worker per
query → **barrier**: deduplicate URLs, assign source ids → **fan out** one
fetch per unique URL → **barrier**: content-hash dedup, quality score, select
top N → **fan out** one extraction call per source → **barrier**:
`assess_coverage` → either loop or → `synthesize` → `verify_citations` →
`finalize`.

---

## Design-decision questions

### "Why fan out per stage instead of per researcher?"

This is the best question to be asked, because the obvious design is worse.

Give each researcher a sub-question and have it search, fetch and extract:
easy to draw, and broken in one specific way — **a worker cannot see its
siblings.** Three sub-questions about fraud detection all surface the same
Wikipedia page, and it gets downloaded three times and sent to a model three
times, because no worker knows the others found it.

Splitting at stage boundaries puts a deduplication barrier between search and
fetch, where all results are visible at once.

Be precise about the payoff, because it varies. A unit test pins the
mechanism: three queries returning the same three URLs produce three fetches,
not nine. On the live run in the README, real overlap was low — 2 duplicates
in 48 Tavily results — because six genuinely different sub-questions return
genuinely different pages. So it is cheap insurance, and it matters most
exactly where sub-questions are closely related, which is where the
per-researcher design wastes the most.

Volunteering that the saving was small on that particular run is worth more
than quoting the flattering number from a test fixture.

The cost is two extra synchronisation points, adding latency equal to the
slowest worker per stage. I took that trade because it converts a token-cost
problem into a latency rounding error.

### "How do you prevent infinite research loops?"

Four independent stop conditions, any one sufficient:

```python
if coverage.sufficient:                                 return "synthesize"
if round_number >= budget.max_research_rounds:          return "synthesize"
if len(completed_queries) >= budget.max_search_queries: return "synthesize"
if len(sources) >= budget.max_sources:                  return "synthesize"
```

The point is that **termination does not depend on the critic being
satisfied.** There is a test that rigs the coverage model to always demand
more research; the run still stops at exactly the round cap and still produces
a report. There is also an LLM call ceiling reserved *before* each call, and
`recursion_limit` on the graph as a backstop against a routing bug.

When a budget stops the run, the report still gets written, with the
limitation stated in it.

### "How do you verify citations?"

Two passes, cheapest and strongest first.

**Structural** — free, deterministic, decidable by lookup:
- does every cited id resolve to a source actually retrieved? (the one that
  must never fail)
- does every factual-looking claim carry a citation?
- which retrieved sources were never used?

**Entailment** — one model call per claim, sampled rather than exhaustive,
key findings first: does the cited evidence actually support this claim?

Running structural first means the expensive pass is never spent on a claim
already known to be broken.

If a citation points at a source that was never retrieved, repair strips the
marker and leaves the sentence. That is the conservative choice: the sentence
may well be true and merely mis-cited, but a dangling reference is always
wrong. The stripped claim then shows up as uncited on re-verification, which
is an honest description of its state.

Underneath all of it is quote verification at extraction time: each evidence
item carries a verbatim span checked against the source text, tolerant of
curly quotes and reflowed whitespace, intolerant of paraphrase.

### "How does local/cloud routing work, and is 'hybrid' real?"

Code asks for a model by role — `router.get(ModelRole.SYNTHESIZER)` — and
configuration maps roles to `provider:model`.

The hybrid split follows one rule: **a role goes local when its output is
short, schema-constrained and produced many times per run; it stays in the
cloud when a wrong answer changes the final report.** Evidence extraction runs
once per source, is the highest-volume role, and its job is quotation rather
than judgement — so it goes local. Synthesis runs once and is what the user
reads — so it stays cloud.

That is not a guess. Measured on `qwen3:4b`: 83% quote fidelity when
extracting (it copies text accurately), but as verifier it judged only 33% of
its own claims as supported, which says more about calibration than about the
claims.

### "What did you learn from running on small local models?"

The most transferable thing in the project:

**Prose instructions do not survive a 4B model; schema fields do.** My
synthesis prompt asked for citation markers like `[S3]` in the claim text. The
model produced a complete, well-organised report with **zero** markers. I
moved citations into a required `source_ids` schema field — same model, same
question, 9 citations, all valid.

> If a model must produce something reliably, put it in the schema, not the
> instructions.

Second: **local models do not parallelise.** Ollama serves one model largely
serially, so fanning eight extraction calls at it produced queueing and read
timeouts rather than throughput. That is why there is a separate
`MAX_PARALLEL_LOCAL_LLM_CALLS`, applied only to local providers.

### "How do you control cost?"

Five mechanisms:

1. **Role-based routing** — the highest-volume role can run locally at zero
   marginal token cost.
2. **Deduplication before fetching** — each unique page costs one fetch and
   one extraction call regardless of how many queries found it.
3. **One extraction call per source**, not per (source × sub-question). That
   bounds calls at the number of sources rather than their product.
4. **Search depth escalation** — Tavily `basic` costs 1 credit, `advanced` 2.
   Round 1 is always basic; later rounds escalate, because by then cheap
   search has demonstrably missed something.
5. **Hard budgets** — rounds, queries, sources, LLM calls, all enforced
   before the spend.

Synthesis never sees raw pages, only the curated evidence package. That is
both a cost and a quality decision.

Cost reporting is honest: pricing lives in `pricing.toml` sourced from the
provider's published rates, and a model absent from that table reports cost as
*unavailable* rather than as a plausible guess.

### "What happens when one researcher fails?"

It records the failure into the `errors` channel and the run continues with
whatever succeeded.

This required a specific discovery: **LangGraph's node-level `error_handler`
does not fire for `Send`-dispatched nodes.** It works for ordinary nodes and
silently does not for fan-out workers, so an escaping exception kills the
whole super-step and loses every sibling's completed work. I verified that
against 1.2.x with a standalone script.

It bit me for real, too: an `httpx.ReadTimeout` from Ollama escaped an
extraction worker because the worker caught only `LLMError` and the router
mapped connection errors but not timeouts. Now all transport failures become
`LLMError` subclasses, and every worker has a broad catch as a final net, with
a regression test.

There is a matching trap: a conditional edge returning an empty list of
`Send`s silently skips the downstream node and ends the graph — no error. Every
dispatcher returns a fallback node name instead.

### "Why Tavily?"

It is search-API-shaped for agents: one call returns ranked results *and*
optionally the page content, which often removes a separate fetch. Generous
free tier, and pricing is legible — basic 1 credit, advanced 2 — which makes
the depth-escalation strategy easy to reason about.

But nothing above the provider boundary knows Tavily exists. Everything
normalises to an internal `SearchResult`. I wrote a second provider (Brave)
specifically to test that claim — an interface with one implementation is a
guess about what varies. Adding Brave required no change to the graph, the
state, or the evidence pipeline.

### "Why no gold-answer evaluation?"

Gold answers for open research questions are expensive to produce, go stale
quickly, and mostly measure whether the model agrees with whoever wrote them.

So the metrics measure whether the system did what it claims: citation
validity, claim support, quote fidelity, evidence coverage, source diversity,
duplicate avoidance.

**The limitation I state up front:** none of this measures whether the report
is *true*. The engine can score perfectly while faithfully reporting what a
set of wrong pages said. It verifies faithfulness to retrieved sources, not
correctness about the world. Saying this before being asked is worth more than
any metric in the table.

---

## Production questions

### "What would change at production scale?"

| Now | At scale | Why |
|---|---|---|
| SQLite checkpoints | Postgres | Multiple workers need shared state |
| In-process asyncio | A task queue | Runs are minutes long; HTTP requests should not hold them |
| Per-run semaphores | A distributed rate limiter | Per-process limits do not compose across replicas |
| Full re-fetch each run | A content cache keyed by canonical URL | Popular sources repeat across runs |
| Sampled entailment | Full checking, offline | Cost moves off the critical path |
| Heuristic source quality | Learned, from citation-usefulness feedback | Real signal beats my domain list |

What I would **not** change: state shape, provenance model, or the budget
mechanism. Those were designed for this.

### "What is the weakest part?"

Pick one and be specific — vagueness here reads as not knowing.

**Source quality scoring.** It is a hand-written weighted heuristic over
document type, search rank, length and recency. The weights are reasoned about
but not tuned against anything, because I had no ground truth for "good
source". It is used only for *ordering* — which pages are worth an extraction
call — never as a claim about truth, and it reports its reasons so a reader
can disagree. The honest fix is feedback: track which sources end up cited in
verified claims and learn from that.

Runner-up: the factual-claim detector for citation coverage is a keyword
heuristic and will misclassify some framing sentences. It only drives
warnings, never deletion, so a false positive costs a warning rather than
content.

### "What would you do next?"

PDF extraction is the biggest real gap — a lot of good academic sources are
PDFs and they are currently skipped as `UNSUPPORTED_TYPE`. After that, a
content cache keyed by canonical URL across runs, and cross-encoder reranking
of evidence before it reaches synthesis.

---

## Questions to ask them

Signals that you think about this as engineering, not a demo:

- "How do you evaluate LLM features where there is no gold answer?"
- "Where do you draw the line between prompt engineering and schema
  enforcement?" (You have a measured story here.)
- "How do you budget and monitor token spend in production?"
- "Do you run any models locally, or is it all vendor APIs?"

---

## Traps

**Do not claim the system verifies truth.** It verifies that claims are
faithful to retrieved sources. Say it that precisely.

**Do not call the parallelism a speedup you have not measured.** What is
measured is that searches run concurrently (six live Tavily queries all
returned within ~1s of each other) and that deduplication works. A wall-clock
speedup number would need an A/B run, and I have not done one.

**Do not oversell hybrid mode.** It routes by role with a stated rule and
measured justification. It does not dynamically choose a model per token or
per difficulty.

**Do not hide that local models are worse at some things.** You measured
exactly where — that is the interesting part, and pretending otherwise
invites the one follow-up you cannot answer.

**Know your own numbers**, from the live run in the README (real Tavily
search, `qwen3:4b` running locally, zero API cost):

- 6 sub-questions, 6 queries, 48 results, 5 sources, 5 distinct domains
- 30 evidence items, **30/30 quotes verified**
- 11 citations, **100% valid**, 100% coverage, 60% entailment-supported
- 20 model calls, 21.5k in / 6.0k out tokens, $0.00, 638s
- 0 separate page fetches — all five sources reused content the search
  provider already returned
- 0 → 9 citations from moving citations into the schema
- 199 hermetic tests in under 10 seconds
