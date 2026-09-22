# Sample run

Unedited artifacts from one real execution on 2026-09-22: live Tavily search,
`qwen3:4b` running locally through Ollama, no cloud model and no API cost.

**Question:** *Compare modern approaches for detecting fraud in highly
imbalanced transaction datasets, including how such models should be
evaluated.*

| File | What to look at |
|---|---|
| `report.md` | The output, including the verification footer. Note that it lists the four claims its own verifier flagged as unsupported rather than hiding them. |
| `evidence.json` | The provenance chain. Every item has `source_id`, `sub_question_id`, `query_id`, the verbatim `quote` and `quote_verified`. |
| `sources.json` | Every source with its quality score, the reasons behind that score, and which queries found it. |
| `metrics.json` | The full measured metric set. |

Worth noticing:

- **30/30 quotes verified.** Every extracted quote was located in its source
  text.
- **11/11 citations valid**, 0 sources retrieved but never cited.
- **`pages_fetched: 0`, `provider_content_reused: 5`.** Tavily returned page
  content with the search results, so no separate HTTP fetch was needed for
  any source.
- **60% entailment support.** The weakest number, and it is a 4B local model
  grading its own report. This is why hybrid mode keeps verification in the
  cloud.
- **2 duplicate URLs out of 48 results.** Real overlap between genuinely
  different sub-questions is low; deduplication is cheap insurance rather
  than a large constant saving.

## Benchmark

`benchmark-local.json` is the output of `agentic-research evaluate -n 3` from
the same session: three benchmark questions, same local model, same live
search. 3/3 succeeded, 58 model calls, $0.00, mean 971s per question.

Citation validity was 100% on all three. Quote fidelity averaged 86% and
claim support 70% — both properties of a 4B model rather than of the
pipeline, and both are why hybrid mode exists.
