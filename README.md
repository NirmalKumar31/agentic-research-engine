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

> **[How this was built](docs/HOW-THIS-WAS-BUILT.md)** — which
> measurement justified each output-quality fix. **[Validation
> history](docs/VALIDATION-HISTORY.md)** — the full release-by-release
> record behind "Measured results" below.

## The problem

Ask one model a broad research question and you get fluent prose with
confident, fabricated citations — no decomposition, no retrieval
discipline, no verification. This project separates the three.

## What it does

- **Builds an answer contract** from the question before retrieving
  anything, and **refuses rather than guesses**: a comparison naming
  fewer than two entities produces an unusable contract with no slots.
- **Decomposes** the question into 4–6 dimensions and **searches in
  parallel**, one query per dimension.
- **Extracts evidence as exact-normalized source quotes**, with
  page-aware PDF citations where extraction succeeds.
- **Assesses its own coverage** and loops on specific gaps, under hard
  limits on rounds, queries, sources and model calls.
- **Writes atomic claims** and **checks each against its own quotes
  separately**, using a pinned NLI classifier plus deterministic guards.
- **Checks relevance separately from support**: a true, well-cited claim
  that answers nothing is withheld for that reason and labelled as such.
- **Reports what it did not answer**, falling back to exact source
  excerpts when nothing passes rather than relaxing the bar.

## How a claim gets published

Two different models, doing two different jobs: a generative model writes
claims and quotes, and a separate, deterministic pipeline — guards, a
pinned semantic-entailment classifier, then a publication gate — decides
whether each one is supported. Full pipeline table and what "the
classifier can be wrong" does and does not mean:
[`docs/ARCHITECTURE.md` §7.6](docs/ARCHITECTURE.md), measured behaviour in
[validation history](docs/VALIDATION-HISTORY.md).

## Does it answer the question?

A separate axis from support (the five-claims incident above). A
relevance judgement, from a critic model rather than the synthesiser,
and one bounded repair pass run against every claim; **relevance fails
closed** — if the judgement cannot be obtained, the claims it would
have covered are withheld. Mechanism detail:
[`docs/ANSWER-SHAPES.md`](docs/ANSWER-SHAPES.md).

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

Blue nodes are fan-in barriers; the orange node is the only back-edge,
and it is budget-guarded. A claim references **evidence ids**, resolved
to sources by the engine rather than supplied by the model, so the two
cannot disagree — citation to exact quote to source is **guaranteed**,
the query-attribution link is **conditional**. Design rationale and full
code example: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Install

Python 3.11 or 3.12. Needs a search provider key in every mode —
[Tavily](https://app.tavily.com) has a sufficient free tier.

```bash
git clone https://github.com/NirmalKumar31/agentic-research-engine.git
cd agentic-research-engine
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
# then set TAVILY_API_KEY=tvly-... in .env

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
`ALLOW_CLOUD_FALLBACK` defaults to false, so local mode never silently
starts spending money if Ollama is unreachable. Frozen-corpus
local-vs-cloud comparison published:
[`evaluations/phase_b/RESULTS.md`](evaluations/phase_b/RESULTS.md) (cloud
8.84-28.51s per run; local 54.50-118.21s completed, 120.02-122.78s
timeout — n=2 per question, no significance claimed).

## Web interface

React and Vite in front of FastAPI, streaming the same graph events the
CLI consumes over SSE. **Replay** (`LIVE_RESEARCH_ENABLED=false`, the
default blueprint) serves recorded runs committed inside the package —
no API key, no search provider, no model. **Live**
(`LIVE_RESEARCH_ENABLED=true`) additionally accepts a visitor's own
question, under demo ceilings. Full mode comparison and the
shared-quota-store caveats: [`docs/ARCHITECTURE.md` §2.1](docs/ARCHITECTURE.md),
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). Check `/api/config` on a
running instance for its actual current state.

```bash
pip install -e ".[web]"
cd web && npm install && npm run build && cd ..
uvicorn agentic_research.web.api:get_asgi_app --factory
```

`render.yaml` deploys the replay site with no secrets. `deploy/render-live.yaml`
deploys the live variant, which prompts for three
credentials (OpenAI, Tavily, and a Hugging Face token for the
private verifier endpoint).

**Live demo:** <https://agentic-research-engine-live.onrender.com> — free
tier, so the first request wakes it (~1 minute). Start with the recorded
runs: they cost nothing and are the better demonstration. Live research on
the demo is an **experimental, fail-closed integration, not a general
research assistant** — scope and measured behavior in
[validation history](docs/VALIDATION-HISTORY.md).

```bash
pip install -e ".[nli-local]"     # includes the local semantic verifier
cp .env.example .env              # add a search provider key
agentic-research check            # exits non-zero if the path can't complete a verified run
agentic-research research "your question"
```

## Local reproducibility

A fresh live research run is **not** bit-for-bit reproducible: search
results change over time and a model tag like `qwen3:4b` can move to
different weights. What *is* reproducible is bounded and explicit, in
three levels:

| Level | What it proves | Command | Needs |
|---|---|---|---|
| **1. Replay** | A committed recorded run re-renders identically, forever | `agentic-research replay <id>` | nothing — no network, no credentials |
| **2. Frozen-corpus engine determinism** | The engine's own machinery is deterministic end to end | `agentic-research verify-reproducible` | nothing — scripted model, scripted search/fetch |
| **3. Local environment** | What's installed matches what you set up with | `./scripts/bootstrap-local.sh` then `./scripts/verify-local.sh` | Python, optionally Ollama and Node |

Level 2 validates **engine behaviour, not model quality**: a scripted
model always returns the same output, so a pass says the state machine
and publication gate behave consistently — nothing about real judgement.

```bash
agentic-research replay list                 # see what is committed
agentic-research verify-reproducible          # engine determinism, offline
```

Dependency-pinning scope (`requirements-lock.txt`, the unpinned
`nli-local` extra) and the Ollama tag-vs-digest caveat:
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md#operations).

## Measured results

Three recorded local `qwen3:4b` runs: 21 unique candidates checked, **11
published, 10 withheld**. A blinded self-review by the project author
(verdict hidden) found **precision 1.00, recall 0.58 — zero unsupported
or uncertain claims published**; this was a self-review, not an
independent benchmark — the reviewer built the system. On a frozen
adversarial set (eleven cases, seventeen claims) across two releases,
irrelevant published claims went from 6 to 0, at the cost of published
claims falling from 14 to 6 and one correct claim now wrongly withheld.

Across eight live hosted runs on one question shape plus four more
question shapes added later: definitional and factual-lookup questions
produce short, correct, cited answers in 50-120s for under a cent;
comparison and procedural questions stay weak. In the original
eight-run study, the bottleneck was measured as synthesis quality (58%
of refusals), not the verification machinery, and **every refusal
inspected in that study was the correct refusal.**

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
- **Spend.** Ceilings reserved before each request: provider-request and
  output-token ceilings are exact, input-token and cost ceilings are
  estimated before dispatch (bounding expected cost, not the invoice). A
  deployment's own provider spend limit is the durable backstop.
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

`citation_integrity` is a structural invariant, not a quality signal:
anything below 100% is an engine bug, since citations derive from
already-resolved evidence.

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

Every external boundary is faked, while the real state machine, reducers
and verification execute. The TLS tests run a real local HTTPS server
with a certificate from an in-process CA — certificate verification
cannot be asserted against a mock.

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
evaluations/        Benchmark Phase B's public, redacted benchmark record (see evaluations/phase_b/README.md)
scripts/            bootstrap-local.sh · verify-local.sh · generate-lock.py
docs/               see docs/README.md for the full index
requirements-lock.txt   hash-pinned core deps, linux/cp312 (see "Local reproducibility")
```

## Technology and license

LangGraph 1.2 · LangChain 1.6 · Pydantic 2 · FastAPI · React 19 + Vite ·
Tavily · trafilatura · pypdf · httpx · Typer · structlog · pytest, ruff,
mypy strict. MIT — see [LICENSE](LICENSE).
