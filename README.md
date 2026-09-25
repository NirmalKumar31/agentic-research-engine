# Agentic Research Engine

A LangGraph research system that decomposes a question, searches the web in
parallel, extracts evidence as exact-normalized source quotes, and publishes
only the atomic claims a dedicated entailment classifier scored as supported
by one of their own cited quotes. Everything else is withheld.

---

## The problem

Ask one model a broad research question and you get fluent prose with
confident, frequently fabricated citations. Three failures are bundled
together: no decomposition, no retrieval discipline, and no verification.
This project separates them and measures whether the result holds.

## What it does

- **Decomposes** the question into 4–6 researchable dimensions.
- **Searches in parallel**, one query per dimension, rewriting queries each
  round so follow-ups do not repeat earlier searches.
- **Deduplicates before fetching.** A page found by four sub-questions costs
  one fetch and one extraction call.
- **Extracts evidence as exact-normalized source quotes**, each checked
  against the retrieved text. Only a match after whitespace and
  punctuation normalisation can ground a citation.
- **Reads PDFs page by page**, so a citation renders `[S4, p. 5]` when
  page-aware PDF extraction succeeded.
- **Assesses its own coverage** and loops on specific gaps, under hard
  limits on rounds, queries, sources and model calls.
- **Writes atomic claims** — one verifiable proposition each, with the
  source's own modality, scope, quantities and time frame preserved.
- **Checks every claim against each of its own quotes separately**, using a
  pinned NLI classifier plus deterministic guards, and removes everything
  that does not clear the bar before publishing.
- **Falls back to exact source excerpts** when nothing passes, rather than
  publishing an empty report or relaxing the bar.

## How a claim gets published

Two different models, doing two different jobs. The generative model never
decides whether its own claims are supported.

| Stage | Who | What |
|---|---|---|
| Planning, search, extraction, synthesis | `qwen3:4b` locally, or a cloud model | Decomposes the question, writes queries, pulls exact quotes, writes atomic claims |
| Deterministic guards | Plain Python | Refuse narrow, high-confidence overclaims: a figure not in the quote, a hedge promoted to a requirement, an invented ranking, causation from association, invented exclusivity |
| Semantic entailment | `DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, pinned to revision `b3546ea6` | Scores each claim against each cited quote separately and returns probabilities only |
| Publication gate | Plain Python | Publishes only when one guard-passing quote entails the claim at ≥ 0.98 |

A claim publishes when **a single cited quote carries it on its own**.
Quotes are never concatenated: assembling a broad claim out of several
partial ones is the failure this gate exists to prevent. Anything else is
withheld — below threshold, guard failure, unresolved evidence, a
non-citable quote, or a classifier that could not be reached. Every failure
path withholds, and none falls back to asking a generative model.

The threshold is calibrated, not guessed, and the model and revision are
pinned because a checkpoint that moves silently invalidates every number
here. All three are recorded on every judgment.

**What this does not mean.** The classifier is a learned model and can be
wrong. It is conservative by construction and by threshold, so its usual
error is withholding a true claim, but "withheld" and "published" are not
proofs. This is not zero hallucinations, not perfect factuality, not general
entailment correctness — it is a measured, fail-closed filter whose
behaviour on the cases tested is written down below.

## Architecture

```mermaid
graph TD
    START([START]) --> AQ[analyze_query] --> PR[plan_research] --> GQ[generate_queries]
    GQ -.Send xN.-> SW[search_worker] --> DS[dedupe_sources<br/>defer]
    DS -.Send xM.-> FW[fetch_worker] --> RS[register_sources<br/>defer]
    RS -.Send xM.-> EW[extract_worker] --> AC[assess_coverage<br/>defer]
    AC -->|gaps and budget left| GF[generate_followups] --> GQ
    AC -->|sufficient| SY[synthesize] --> VC[verify_citations] --> FIN[finalize] --> END([END])
    style DS fill:#e8f0fe
    style RS fill:#e8f0fe
    style AC fill:#e8f0fe
    style GF fill:#fff4e5
```

Blue nodes are fan-in barriers. The orange node is the only back-edge, and
it is budget-guarded. Design rationale, including rejected alternatives:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Provenance

A claim references **evidence ids**, not source ids. The engine resolves
those to sources itself:

```python
Claim(
    text="Precision-recall is the better metric under heavy imbalance",
    evidence_ids=["S3-e1"],  # supplied by the model
    citation_ids=["S3"],  # derived by the engine
    kind=ClaimKind.FACTUAL,
)

EvidenceItem(
    id="S3-e1",
    source_id="S3",
    sub_question_id="SQ2",
    discovery=DiscoveryRef(query_id="Q5", sub_question_id="SQ2"),  # or None
    cross_attributed=False,
    quote="precision-recall curves are a more informative evaluation than ROC AUC",
    quote_match=QuoteMatch.EXACT_NORMALIZED,
    page=14,
)
```

That ordering is the point. Letting a model emit both invites the two to
disagree. An evidence id that does not resolve, or resolves to a quote that
never aligned to its source, is dropped and reported.

The chain has two halves with different guarantees:

- **Guaranteed.** Every citation resolves to an exact evidence item, its
  exact-normalized source quote, its source, and its page when page-aware
  PDF extraction succeeded for that source.
- **Conditional.** The link back to a query exists only where that source
  was genuinely retrieved for that sub-question. Much of the evidence —
  20.8% to 83.3% across the three recorded runs — is reused across
  sub-questions and is marked `cross_attributed` with no query id, rather
  than borrowing an unrelated one.

## Install

Python 3.11 or 3.12.

```bash
git clone https://github.com/NirmalKumar31/agentic-research-engine.git
cd agentic-research-engine
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Web search needs a provider key in every mode.
[Tavily](https://app.tavily.com) has a sufficient free tier:

```bash
# .env
TAVILY_API_KEY=tvly-...
```

## Quickstart

```bash
agentic-research check                      # verify configuration
agentic-research research "your question"   # run
agentic-research show latest                # re-display a stored run
```

## Modes

| Mode | Models | Notes |
|---|---|---|
| `local` | Ollama (`qwen3:4b`) | No paid LLM usage. ~15 minutes per run on an M-series laptop. Still needs a search provider. |
| `cloud` | OpenAI | Substantially faster, at metered cost. No clean-corpus cloud run is currently published, so no timing is quoted here. |
| `hybrid` | Extraction local, reasoning cloud | Extraction is the highest-volume role and is mechanical. |

Set `LLM_MODE` and, for cloud or hybrid, `OPENAI_API_KEY`.
`ALLOW_CLOUD_FALLBACK` defaults to false: choosing local mode should not
quietly start spending money if Ollama is unreachable.

## Web interface

React and Vite in front of FastAPI, streaming the same graph events the CLI
consumes over SSE. The interface has two modes, chosen by the server.

**Replay** (`LIVE_RESEARCH_ENABLED=false`, the default) serves research runs
recorded earlier and committed inside the package. It needs no API key, no
search provider and no model.

**Live** (`LIVE_RESEARCH_ENABLED=true`) additionally accepts a visitor's own
question, under the demo ceilings.

Replay is the default for the public deployment because the daily run cap
lives in process memory, and a host that sleeps when idle resets it on every
cold start — so it cannot bound an account-level quota. Per-run request and
spend ceilings are unaffected. The alternative was a persistent quota store
whose only purpose would be letting strangers spend an API budget.

```bash
pip install -e ".[web]"
cd web && npm install && npm run build && cd ..
uvicorn agentic_research.web.api:get_asgi_app --factory
```

`render.yaml` deploys the replay site and declares no secrets.
`deploy/render-live.yaml` deploys the live variant and prompts for two keys.
Render's free tier sleeps when idle, so the first visit after a quiet period
takes about a minute; replay makes that cheap, because waking the service
spends nothing.

## Measured results

Three recorded local `qwen3:4b` runs, one round each, verified by the
pinned DeBERTa NLI classifier at threshold 0.98. Every substantive claim
was checked against each of its own quotes — `checked == checkable` in all
three — and only claims one quote carried on its own were published.

| | RAG comparison | NIST framework | Fraud detection |
|---|---|---|---|
| Research rounds | 1 | 1 | 1 |
| Search queries | 6 | 6 | 6 |
| Unique sources (usable) | 5 (5) | 5 (5) | 5 (5) |
| Evidence items | 30 | 30 | 30 |
| Generated substantive claims | 28 | 12 | 14 |
| Exact duplicates removed | 4 | 3 | 8 |
| Unique candidates checked | 24/24 | 9/9 | 6/6 |
| Supported | 11 | 8 | 4 |
| Partially supported | 3 | 0 | 0 |
| Unsupported | 10 | 1 | 2 |
| Never checked | 0 | 0 | 0 |
| **Withheld** | 13 | 1 | 2 |
| **Published** | **11** | **8** | **4** |
| Evidence-only excerpts | 0 | 0 | 0 |
| Quote fidelity (exact) | 90% | 80% | 93% |
| Evidence integrity | 100% | 100% | 100% |
| Page-cited evidence | 5 | 0 | 0 |
| Duration | 846s | 923s | 547s |

39 unique candidates across the three runs, all checked, **23 published
and 16 withheld** — a 59% publication rate. The previous generative
verifier published nothing at all from a comparable set.

### Release validation

Publishing something is easy; publishing only what the evidence supports
is the claim being made. So all 23 published claims were read against
the exact quote the gate selected. That review is recorded per candidate
in [`examples/release-audit/`](examples/release-audit/), alongside every
pairwise NLI score and guard result.

**Result: 22 of 23 supported, 1 not.** One claim published
*"some vector databases **lack** standardized encryption features"* from
a source saying they *"**might** lack"* them. The hedge was dropped, and
neither layer caught it — the classifier scored it 0.9946, and the
modality guard compares strength bands where "no modality" is the
weakest band, so removing a hedge reads as weakening rather than
strengthening.

That is a real defect, stated here rather than rounded away. It is
[tracked in LIMITATIONS](docs/LIMITATIONS.md) and the release is not
tagged while it stands.

These are product artifacts, served by the demo. They are not a
benchmark: n=1 each, one model, one configuration.

### Attribution experiment

Three extraction strategies over one frozen six-source corpus, three
repeats each, `qwen3:4b`.

| | all-open | retrieved-only | adjacent |
|---|---|---|---|
| Evidence coverage | **100%** | 16.7% | 50.0% |
| Citable evidence | 27.3 | 17.7 | 18 |
| Cross-attributed | 79.6% | 0% | 59.1% |

No variation was observed across the three repeats of any arm. Three
draws cannot establish that sampling noise is absent; the defensible
point is structural. On this corpus each source was retrieved for a mean
of 1.17 sub-questions, so retrieved-only extraction mechanically limits
how many sub-questions can reach a two-source coverage bar. Production
uses all-open because this project prioritises multi-source coverage
over complete query-level lineage. Scope and caveats:
[`examples/attribution-experiment/`](examples/attribution-experiment/).

An earlier local-versus-cloud comparison used a corpus whose source text had
been stripped and is excluded from current results.

## Security

Relevant because this fetches attacker-influenced URLs and feeds
attacker-influenced text to a model.

- **SSRF.** Deny-by-default on resolved addresses: loopback, private,
  link-local, multicast, reserved and unspecified ranges are blocked across
  IPv4 and IPv6 including IPv4-mapped forms. Every resolved address must be
  safe, not merely one of them.
- **DNS rebinding.** The connection goes to the address that was validated,
  with the hostname carried in the `Host` header and TLS SNI. A target with
  no validated address fails closed; failover never re-resolves; every
  redirect hop is revalidated and re-pinned. Certificate verification is
  preserved, and a wrong SNI is refused — tested against a real TLS server
  with a real certificate.
- **Spend.** Ceilings are reserved before each provider request rather
  than counted after it. The provider-request and output-token ceilings
  are exact. The input-token and cost ceilings are *estimated* before
  dispatch — input size is approximated and prices come from a local
  table — so they bound the expected cost, not the invoice. For a public
  deployment the provider account's own spend limit is the durable
  backstop, and the deployment docs say so.
- **Prompt injection.** Retrieved text is framed as untrusted data, the
  extractor has no tools, and a claim invented from a page references no
  evidence so it fails resolution.
- **Secrets.** gitleaks runs over full history with rules for the providers
  used here, verified by a positive control that plants randomly generated
  credentials each run.

## Evaluation

```bash
agentic-research evaluate -n 3       # benchmark questions
agentic-research freeze "question"   # capture an evidence corpus
agentic-research compare -a local=ollama:qwen3:4b -a cloud=openai:gpt-6-luna
agentic-research attribution -r 3    # extraction strategy experiment
```

No gold answers. Every metric asks whether the system did what it claims,
not whether the world agrees.

| Metric | What it measures |
|---|---|
| `evidence_integrity` | Share of referenced evidence ids that exist and are citable |
| `citation_coverage` | Share of evidence-owing claims carrying a citation |
| `claim_support` | Share of checked claims fully entailed by their own evidence |
| `quote_fidelity` | Share of quotes found verbatim after normalisation |
| `evidence_coverage` | Share of sub-questions with ≥2 exact-match items from ≥2 sources |
| `source_diversity` | 1 − share held by the largest domain |

`citation_integrity` is a structural invariant, not a quality signal: the
engine derives citations from already-resolved evidence, so anything below
100% is an engine bug.

## Limitations

The short version: it verifies faithfulness to retrieved evidence, not truth
about the world. Exact quote matching proves textual alignment, not
entailment. Published figures come from single runs.

Full list: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Testing

```bash
pip install -e ".[dev]"
pytest              # hermetic: no network, no credentials, no cost
```

Every external boundary is faked, while the real state machine, reducers,
deduplication and verification execute. The TLS tests run a real local HTTPS
server with a certificate from an in-process CA, because certificate
verification cannot be asserted against a mock.

## Repository

```
src/agentic_research/
    config.py       typed settings, role→model resolution, budgets
    models.py       domain types (the provenance chain)
    graph/          state, reducers, workflow, routing, prompts, nodes
    llm/            role router, usage accounting, pricing
    search/         provider contract, Tavily, Brave
    retrieval/      URL safety, fetcher, PDF and content extraction
    evidence/       deduplication, quality, store, quote verification
    citations/      resolution, verification, publication gate
    evaluation/     metrics, benchmark, A/B, attribution experiment
    cli/            Typer commands, terminal progress rendering
    web/            FastAPI, recorded runs
web/                React + Vite frontend
tests/              unit (hermetic) · integration (opt-in)
examples/           attribution experiment, archived artifacts
docs/               ARCHITECTURE.md · LIMITATIONS.md
```

## Technology

LangGraph 1.2 · LangChain 1.6 · Pydantic 2 · FastAPI · React 19 + Vite ·
Tavily · trafilatura · pypdf · httpx · Typer · structlog · pytest, ruff,
mypy strict.

## License

MIT — see [LICENSE](LICENSE).
