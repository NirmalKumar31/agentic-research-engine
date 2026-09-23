# Sample run

Unedited artifacts from one real execution on 2026-09-23, after the
provenance rework: live Tavily search, `qwen3:4b` through Ollama, no cloud
model and no external LLM API spend.

**Question:** *Compare modern approaches for detecting fraud in highly
imbalanced transaction datasets, including how such models should be
evaluated.*

| File | What to look at |
|---|---|
| `report.md` | The output and its verification footer, including the claims its own verifier flagged |
| `evidence.json` | The provenance chain: `evidence_ids`, `discovery`, `quote_match`, `page`, `cross_attributed` |
| `sources.json` | Every source with `content_origin`, quality score and the discovery paths that found it |
| `metrics.json` | The full metric set plus the captured environment |
| `run.json` | Plan, queries, coverage history, errors |

Worth opening `evidence.json` specifically. Each item carries the exact
quote, its match class, and the discovery path — which is what makes a
claim in `report.md` traceable to a span rather than to a URL.

## The uncomfortable numbers

Published as measured, not as hoped:

- **Quote fidelity 74%** (20 of 27 exact). Two quotes were reworded and
  five could not be located; all seven are excluded from citation. Under
  the previous looser definition this read 100%.
- **78% of evidence is cross-attributed** — it answers a sub-question whose
  queries never retrieved that source. Provenance to the source holds; the
  evidence→query link does not.
- **Claim support is sampled**, 10 of 17 eligible claims. Benchmark runs
  check every claim; this was an interactive run and is labelled as such.
- **Citation integrity 100% means nothing here.** Citations are derived
  from already-resolved evidence, so it is an invariant. Evidence
  integrity, also 100%, is the figure that measures the model.

Runtime was 1,096s on a 10-core arm64 laptop, nearly all of it local
inference. See `metrics.json` → `environment` for the full capture.
