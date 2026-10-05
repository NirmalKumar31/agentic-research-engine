"""Strip verbatim scraped web text from the 12 frozen corpora before
public redistribution.

`SourceDocument.text` (the full fetched page) and `EvidenceItem.quote`
(a verbatim excerpt from that page) are third-party web content whose
redistribution rights this project has not established -- unlike
`claim` (the model's own paraphrase) or `content_hash`/metadata, which
are not the original copyrighted text itself. Rather than guess at fair
use for an arbitrary mix of blog posts, docs pages and forum threads
pulled by live search, the conservative default is to not redistribute
the verbatim text in a public repository at all.

This produces a metadata-only rendition of each corpus: every field
except `text` and `quote` is preserved (sources, URLs, timestamps,
`content_hash`, quality scores, claims, relevance, citability), so the
*structure* of what was retrieved and cited remains fully inspectable.
What is lost is exactly the ability to recompute `corpus_hash()` against
this file, since that hash covers the original including the stripped
text -- the hash in `manifest.json` refers to the original full-text
corpus, not to this redacted rendition. Plaintext originals are not
retained in the repository or working tree; see RESULTS.md's
redistribution note for the encrypted-archive record. A reproduction
procedure (re-run `agentic-research freeze` against the same question
text) is the documented path to an equivalent corpus for independent
verification, not a guarantee of byte-identical text from live web
sources that can change or disappear.

Run once, by hand, before committing corpora publicly. Does not modify
the original files in `evaluations/phase_b/corpora/` -- writes to
`evaluations/phase_b/corpora_public/`, which is what actually gets
committed; the originals stay local for the record.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "evaluations" / "phase_b" / "corpora"
DST = ROOT / "evaluations" / "phase_b" / "corpora_public"


def redact_source(source: dict) -> dict:
    redacted = dict(source)
    original_len = len(redacted.get("text", ""))
    redacted["text"] = (
        f"[REDACTED: {original_len} chars of retrieved source text, not "
        "redistributed here -- see content_hash and the reproduction "
        "procedure in evaluations/phase_b/corpora_public/README.md]"
    )
    return redacted


def redact_evidence(item: dict) -> dict:
    redacted = dict(item)
    original_len = len(redacted.get("quote", ""))
    redacted["quote"] = (
        f"[REDACTED: {original_len} chars of verbatim source quote, not redistributed here]"
    )
    return redacted


README = """# Redacted public corpora

The files in this directory are Benchmark Phase B's 12 frozen corpora
with `SourceDocument.text` and `EvidenceItem.quote` removed -- the
verbatim scraped third-party web text, whose redistribution rights this
project has not established. Everything else is unchanged: source URLs,
titles, domains, `content_hash`, timestamps, quality scores, the
model's own `claim` paraphrases, relevance scores, and citability.

**These files will not reproduce the `corpus_hash()` values recorded in
`../manifest.json`** -- that hash covers the original, full-text corpus.
Plaintext originals are not retained in the repository or working tree:
two independently stored encrypted archives were round-trip verified
(decrypted, every file checked against its SHA-256) before the
plaintext was deleted -- see `../RESULTS.md`'s redistribution note.

## Reproduction procedure

To obtain an equivalent (not necessarily byte-identical -- live web
content can change) corpus for independent verification:

```
agentic-research freeze "<question text from examples/benchmark/questions.json>" \\
    -o <output path> --mode local
```

using the pinned Ollama model (`qwen3:4b`, digest
`359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`) and
the same search provider (Tavily) this run used. Results will differ to
the extent live search results have changed since this corpus was
originally frozen (see each corpus's own `captured_at` timestamp).
"""


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    corpora = sorted(SRC.glob("*.json"))
    if not corpora:
        raise SystemExit(f"no corpora found under {SRC}")
    for path in corpora:
        data = json.loads(path.read_text())
        data["sources"] = [redact_source(s) for s in data["sources"]]
        data["evidence"] = [redact_evidence(e) for e in data["evidence"]]
        (DST / path.name).write_text(json.dumps(data, indent=2), encoding="utf-8")
    (DST / "README.md").write_text(README, encoding="utf-8")
    print(f"Redacted {len(corpora)} corpora into {DST}")


if __name__ == "__main__":
    main()
