# Agentic Research Engine

A LangGraph research system that decomposes a question, searches the web in
parallel, extracts evidence as exact-normalized source quotes, and publishes
only the atomic claims a dedicated entailment classifier scored as supported
by one of their own cited quotes. Everything else is withheld.

Supported is not the same as relevant, so both are checked. A claim can be
entailed by its quote at 0.996, cited correctly, and answer nothing that was
asked — a deployed run published five of those before the answer contract
existed. Claims are held to a contract built from the question before
retrieval starts, and one that fills no part of it is withheld with its own
reason rather than reported as unsupported, which it is not.

> **[How this was built](docs/HOW-THIS-WAS-BUILT.md)** — how the output
> quality work was diagnosed: which measurement justified each fix, three
> proposed fixes that were withdrawn when the measurement contradicted
> them, and two hypotheses that turned out to be wrong.
>
> **[Validation history](docs/VALIDATION-HISTORY.md)** — the full
> release-by-release measurement record behind the one-paragraph summary
> in "Measured results" below.

---

## The problem

Ask one model a broad research question and you get fluent prose with
confident, frequently fabricated citations. Three failures are bundled
together: no decomposition, no retrieval discipline, and no verification.
This project separates them and measures whether the result holds.

## What it does

- **Builds an answer contract** from the question before retrieving
  anything — the canonical slots an answer of that kind must fill — and
  **refuses rather than guesses**: a comparison naming fewer than two
  entities produces an unusable contract with no slots at all.
- **Decomposes** the question into 4–6 researchable dimensions and
  **searches in parallel**, one query per dimension.
- **Extracts evidence as exact-normalized source quotes**, each checked
  against the retrieved text, with page-aware PDF citations where
  extraction succeeds.
- **Assesses its own coverage** and loops on specific gaps, under hard
  limits on rounds, queries, sources and model calls.
- **Writes atomic claims** — one verifiable proposition each — and
  **checks every claim against each of its own quotes separately**,
  using a pinned NLI classifier plus deterministic guards, removing
  everything that does not clear the bar before publishing.
- **Checks relevance separately from support**, so a true, well-cited
  claim that answers nothing is withheld for that reason and labelled
  as such, rather than reported alongside what did answer the question.
- **Reports what it did not answer**, and falls back to exact source
  excerpts when nothing passes, instead of publishing an empty report
  or relaxing the bar.

## How a claim gets published

Two different models, doing two different jobs. The generative model never
decides whether its own claims are supported.

| Stage | Who | What |
|---|---|---|
| Planning, search, extraction, synthesis | `qwen3:4b` locally, or a cloud model | Decomposes the question, writes queries, pulls exact quotes, writes atomic claims |
| Deterministic guards | Plain Python | Refuse narrow, high-confidence overclaims: a figure not in the quote, a hedge promoted to a requirement, an invented ranking, causation from association, invented exclusivity |
| Semantic entailment | `DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, pinned to revision `b3546ea6` | Scores each claim against each cited quote separately and returns probabilities only |
| Publication gate | Plain Python | Publishes only when one guard-passing quote entails the claim at ≥ 0.98 |

Claims must be **atomic** — enforced before scoring, because a fused claim
defeats every other guard. A claim publishes when **a single cited quote
carries it on its own**; quotes are never concatenated. Every failure
path withholds, and none falls back to asking a generative model.

The threshold is calibrated, not guessed, and the model and revision are
pinned and recorded on every judgment. **What this does not mean:** the
classifier is a learned model and can be wrong; it is conservative by
construction, so its usual error is withholding a true claim. This is not
zero hallucinations or general entailment correctness — it is a measured,
fail-closed filter whose behaviour on the cases tested is in
[validation history](docs/VALIDATION-HISTORY.md).

## Does it answer the question?

A separate axis from support, and for one release only the first was
being asked: a deployed run published five claims that every gate above
passed — entailed, cited, atomic — and that collectively did not address
what was asked. "Supported" had been standing in for "an answer".

| Stage | Who | What |
|---|---|---|
| Answer contract | Plain Python | Turns the question into the slots an answer must fill; refuses when the question cannot be given a shape |
| Proposition decomposition | Plain Python | Splits a claim on a new subject with a finite verb, not on the word "and" |
| Relevance judgement | The **critic** model, batched | Does this claim help answer the question? Asked of the critic, not the synthesiser — a model marking its own homework finds its work relevant |
| Bounded repair | The critic, then every gate again | One rewrite for claims refused on wording alone; may not launder — refused if it adds a number, a subject, causation, or states the claim more strongly |
| Coverage | Plain Python | Which core slots the published report actually filled |

**Relevance fails closed:** if the judgement cannot be obtained, the
claims it would have covered are withheld. Full mechanism detail,
including the specific failure this section exists to prevent, in
[validation history](docs/VALIDATION-HISTORY.md).

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

A claim references **evidence ids**, resolved to sources by the engine,
not supplied by the model — so the two cannot disagree. The chain is
**guaranteed** from citation to exact quote to source (and page, where
PDF extraction succeeded), and **conditional** for the query-attribution
link: evidence reused across sub-questions is marked `cross_attributed`
rather than borrowing an unrelated query id. Full code example and the
guaranteed/conditional split: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

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
| `cloud` | OpenAI | Substantially faster, at metered cost. |
| `hybrid` | Extraction local, reasoning cloud | Extraction is the highest-volume role and is mechanical. |

Set `LLM_MODE` and, for cloud or hybrid, `OPENAI_API_KEY`.
`ALLOW_CLOUD_FALLBACK` defaults to false: choosing local mode should not
quietly start spending money if Ollama is unreachable.

A frozen-corpus local-vs-cloud comparison is published:
[`evaluations/phase_b/RESULTS.md`](evaluations/phase_b/RESULTS.md) (cloud
8.84-28.51s per run; local 54.50-118.21s completed, 120.02-122.78s
timeout — n=2 per question, no significance claimed).

## Web interface

React and Vite in front of FastAPI, streaming the same graph events the CLI
consumes over SSE.

**Replay** (`LIVE_RESEARCH_ENABLED=false`, the default blueprint) serves
recorded runs committed inside the package — no API key, no search
provider, no model. **Live** (`LIVE_RESEARCH_ENABLED=true`) additionally
accepts a visitor's own question, under demo ceilings. The live demo URL
below currently runs live mode — check `/api/config` and `/api/readiness`
on that URL for the instance's actual current state rather than assuming
this document describes it; a deployment's configuration can change
independently of the blueprint default.

```bash
pip install -e ".[web]"
cd web && npm install && npm run build && cd ..
uvicorn agentic_research.web.api:get_asgi_app --factory
```

`render.yaml` deploys the replay site with no secrets. `deploy/render-live.yaml`
deploys the live variant, which prompts for three
credentials (OpenAI, Tavily, and a Hugging Face token for the
private verifier endpoint) and provisions the Key Value store that holds
the daily cap. The daily cap survives a web restart but not a restart of
the Key Value store itself on Render's free tier; the financial backstop
there is the spend limit on the provider account, not this counter.

**Live demo:** <https://agentic-research-engine-live.onrender.com> — free
tier, so the first request wakes it (~1 minute). Start with the recorded
runs: they cost nothing and are the better demonstration. Live research on
the demo is an **experimental, fail-closed integration, not a general
research assistant** — scope and measured behavior in
[validation history](docs/VALIDATION-HISTORY.md).

```bash
pip install -e ".[nli-local]"     # includes the local semantic verifier
cp .env.example .env              # add a search provider key
agentic-research check            # verifies the whole path before spending
agentic-research research "your question"
```

`check` exits non-zero if the selected path cannot complete a verified
run — a missing search key, an unreachable model, or a verifier that
cannot answer.

## Local reproducibility

A fresh live research run is **not** bit-for-bit reproducible: search
results change over time, a model tag like `qwen3:4b` can move to
different weights, and hardware differences affect generation. What *is*
reproducible is bounded and explicit, in three levels:

| Level | What it proves | Command | Needs |
|---|---|---|---|
| **1. Replay** | A committed recorded run re-renders identically, forever | `agentic-research replay <id>` | nothing — no network, no credentials |
| **2. Frozen-corpus engine determinism** | The engine's own machinery is deterministic end to end | `agentic-research verify-reproducible` | nothing — scripted model, scripted search/fetch |
| **3. Local environment** | What's installed matches what you set up with | `./scripts/bootstrap-local.sh` then `./scripts/verify-local.sh` | Python, optionally Ollama and Node |

Level 2 validates **engine behaviour, not model quality**: a scripted
model always returns the same structured output, so a pass says the
state machine and publication gate behave consistently — nothing about a
real model's judgement.

```bash
agentic-research replay list                 # see what is committed
agentic-research verify-reproducible          # engine determinism, offline
```

`requirements-lock.txt` hash-pins core dependencies for CI's exact
platform (linux/cp312) — dependency integrity, not a hermetic build or
supply-chain attestation. `nli-local` (torch) is deliberately unpinned
(platform/CUDA-specific wheels): `pip install -e ".[nli-local]"`.

For Ollama: `ollama pull qwen3:4b` pulls a **tag**, not an immutable
digest. `bootstrap-local.sh` records the digest it observes;
`verify-local.sh` compares against it if you export
`EXPECTED_OLLAMA_DIGEST` yourself — without that export, a changed
digest is a real limitation this script will not catch.

## Measured results

Three recorded local `qwen3:4b` runs: 21 unique candidates checked, **11
published, 10 withheld**. An independent blinded human review of every
candidate (verdict hidden) found **precision 1.00, recall 0.58 — zero
unsupported or uncertain claims published**. On a frozen adversarial set
(eleven cases, seventeen claims) across two releases, irrelevant
published claims went from 6 to 0, at the cost of published claims
falling from 14 to 6 and one correct claim now wrongly withheld.

Across eight live hosted runs on one question shape plus four more
question shapes added later: definitional and factual-lookup questions
produce short, correct, cited answers in 50-120s for under a cent;
comparison and procedural questions stay weak, and the bottleneck is
measured as synthesis quality (58% of refusals), not the verification
machinery. **Every refusal inspected, across all runs, was the correct
refusal.**

Full release-by-release numbers, every table, and what each hosted run
found: **[`docs/VALIDATION-HISTORY.md`](docs/VALIDATION-HISTORY.md)**.

## Security

Relevant because this fetches attacker-influenced URLs and feeds
attacker-influenced text to a model.

- **SSRF.** Deny-by-default on resolved addresses (loopback, private,
  link-local, multicast, reserved, unspecified; IPv4 and IPv6).
- **DNS rebinding.** Connects to the validated address; hostname carried
  in `Host`/SNI; every redirect hop revalidated and re-pinned; a wrong
  SNI is refused, tested against a real TLS server with a real
  certificate.
- **Spend.** Ceilings reserved before each provider request. Provider-
  request and output-token ceilings are exact; input-token and cost
  ceilings are estimated before dispatch, so they bound expected cost,
  not the invoice. A public deployment's own provider spend limit is the
  durable backstop.
- **Prompt injection.** Retrieved text is framed as untrusted data, the
  extractor has no tools, and a claim invented from a page references no
  evidence so it fails resolution.
- **Secrets.** gitleaks runs over full history, verified by a positive
  control that plants randomly generated credentials each run.

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

**Scope:** verifies faithfulness to retrieved evidence, not truth about
the world. Exact quote matching proves textual alignment, not entailment.
Published figures come from single runs. Full list:
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

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
    evaluation/     metrics, benchmark, A/B, attribution experiment, frozen-corpus determinism
    cli/            Typer commands, terminal progress rendering
    web/            FastAPI, recorded runs
web/                React + Vite frontend
tests/              unit (hermetic) · integration (opt-in)
examples/           benchmark questions, quality-eval, attribution experiment, live validation
evaluations/        Benchmark Phase B's evidentiary record (see evaluations/phase_b/README.md)
scripts/            bootstrap-local.sh · verify-local.sh · generate-lock.py
docs/               see docs/README.md for the full index
requirements-lock.txt   hash-pinned core deps, linux/cp312 (see "Local reproducibility")
```

## Technology

LangGraph 1.2 · LangChain 1.6 · Pydantic 2 · FastAPI · React 19 + Vite ·
Tavily · trafilatura · pypdf · httpx · Typer · structlog · pytest, ruff,
mypy strict.

## License

MIT — see [LICENSE](LICENSE).
