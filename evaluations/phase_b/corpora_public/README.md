# Redacted public corpora

The files in this directory are Benchmark Phase B's 12 frozen corpora
with `SourceDocument.text` and `EvidenceItem.quote` removed -- the
verbatim scraped third-party web text, whose redistribution rights this
project has not established. Everything else is unchanged: source URLs,
titles, domains, `content_hash`, timestamps, quality scores, the
model's own `claim` paraphrases, relevance scores, and citability.

**These files will not reproduce the `corpus_hash()` values recorded in
`../manifest.json`** -- that hash covers the original, full-text corpus,
kept locally for the record and not redistributed here.

## Reproduction procedure

To obtain an equivalent (not necessarily byte-identical -- live web
content can change) corpus for independent verification:

```
agentic-research freeze "<question text from examples/benchmark/questions.json>" \
    -o <output path> --mode local
```

using the pinned Ollama model (`qwen3:4b`, digest
`359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`) and
the same search provider (Tavily) this run used. Results will differ to
the extent live search results have changed since this corpus was
originally frozen (see each corpus's own `captured_at` timestamp).
