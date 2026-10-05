"""A documentation claim about what the committed recordings publish
must match what they actually publish -- today, not when it was written.

LIMITATIONS.md once said "all three recorded demos publish nothing",
written on 2026-09-24 about a run under a verifier replaced the next
day and an atomicity rule rewritten two days after that. The sentence
was never wrong when written; it became wrong silently when the three
recordings were regenerated under the fixed engine and nobody revisited
the sentence describing them. Found during a closeout accuracy audit,
not by this test -- this test exists so the next drift is found by CI
instead of by a person reading carefully.

The per-recording "Measured results" table moved from README.md to
docs/VALIDATION-HISTORY.md during a documentation curation pass (the
README now states only the aggregate); this test's detailed check
followed it there, so it still checks the real table, not a stale
reference to where that table used to live. A second check confirms
README's new aggregate ("11 published, 10 withheld") still sums to the
same per-recording counts.

Two things are checked: the committed recordings' real published-claim
counts agree with what VALIDATION-HISTORY.md states about them, and no
future prose in these documents claims "publish nothing" (or an
unscoped zero) about "the three" canonical recordings without naming
the historical scope that claim needs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDINGS_DIR = ROOT / "src" / "agentic_research" / "web" / "recorded_runs"
README = ROOT / "README.md"
VALIDATION_HISTORY = ROOT / "docs" / "VALIDATION-HISTORY.md"
LIMITATIONS = ROOT / "docs" / "LIMITATIONS.md"

# id -> the README table's column label, in the order README lists them.
RECORDINGS = {
    "rag-vector-vs-search": "RAG comparison",
    "nist-ai-risk-framework": "NIST framework",
    "fraud-detection-imbalanced": "Fraud detection",
}


def _published_claim_count(payload: dict) -> int:
    result = payload.get("result", payload)
    report = result.get("report") or {}
    summary = len(report.get("summary_claims", []))
    sections = sum(len(s.get("claims", [])) for s in report.get("sections", []))
    findings = len(report.get("key_findings", []))
    return summary + sections + findings


def _load(recording_id: str) -> dict:
    return json.loads((RECORDINGS_DIR / f"{recording_id}.json").read_text())


class TestReadmesPublishedCountsMatchTheCommittedRecordings:
    def test_every_recording_in_the_table_exists(self) -> None:
        for recording_id in RECORDINGS:
            assert (RECORDINGS_DIR / f"{recording_id}.json").is_file(), recording_id

    def test_the_validation_history_published_row_matches_reality(self) -> None:
        """Parses the "Measured results" table's own Published row and
        checks each column against the recording it names.

        Written against the table's actual text rather than a hand-kept
        list of numbers, so a regenerated recording with a different
        count fails this test instead of quietly outdating the table.
        """
        text = VALIDATION_HISTORY.read_text()
        header_match = re.search(
            r"\|\s*\|\s*" + r"\s*\|\s*".join(re.escape(c) for c in RECORDINGS.values()) + r"\s*\|",
            text,
        )
        assert header_match, (
            "docs/VALIDATION-HISTORY.md's measured-results header no longer "
            "matches the expected columns"
        )
        published_match = re.search(r"\|\s*\*\*Published\*\*\s*\|([^\n]+)\|", text)
        assert published_match, (
            "docs/VALIDATION-HISTORY.md has no **Published** row in the measured-results table"
        )
        counts = [c.strip().strip("*") for c in published_match.group(1).split("|") if c.strip()]
        assert len(counts) == len(RECORDINGS), counts

        for recording_id, count_str in zip(RECORDINGS, counts, strict=True):
            payload = _load(recording_id)
            actual = _published_claim_count(payload)
            assert actual == int(count_str), (
                f"VALIDATION-HISTORY.md says {recording_id} publishes {count_str}, "
                f"the committed recording actually publishes {actual}"
            )

    def test_the_readme_aggregate_matches_the_sum_of_real_counts(self) -> None:
        """README no longer states per-recording counts (see above test),
        only the aggregate "11 published, 10 withheld" -- this must still
        sum to the same real counts, not just look plausible."""
        total_published = sum(_published_claim_count(_load(r)) for r in RECORDINGS)
        text = README.read_text()
        match = re.search(r"\*\*(\d+)\s+published,\s+(\d+)\s+withheld\*\*", text)
        assert match, "README's aggregate published/withheld claim is missing or reworded"
        assert int(match.group(1)) == total_published, (
            f"README claims {match.group(1)} published, the three recordings "
            f"actually total {total_published}"
        )


class TestNoUnscopedZeroPublicationClaim:
    """Guards the specific mistake, not just today's fixed text: a future
    edit must not re-assert "the three recordings publish nothing"
    without naming which historical run it is talking about.
    """

    _UNSCOPED_ZERO = re.compile(
        r"(all three|the three) recorded demos? publish(es)? nothing", re.IGNORECASE
    )
    # A historical claim is allowed, but only if it is anchored: a date or
    # a commit hash in the same paragraph, not just the word "historical".
    _HISTORICAL_ANCHOR = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b|\b[0-9a-f]{7,40}\b")

    def test_limitations_does_not_make_the_claim_unscoped(self) -> None:
        """The anchor must be in this sentence or the very next one.

        Two weaker versions of this check were tried first, and both
        passed on a reintroduced copy of the original bug. A ~400-char
        window either side reached past the unscoped sentence into a
        later one that happens to carry a date. Splitting on sentence
        boundaries fixed that, but naive splitting never recognises
        "...nothing.**" as a sentence end at all -- the `**` sits
        between the period and the whitespace -- so the match silently
        merged into the next sentence anyway, the same failure in a new
        shape. `\\*{0,2}` in the split pattern is the fix for that part.

        The next sentence is allowed to carry the anchor, not only the
        matching one: a bold lead-in legitimately hands the date to the
        sentence right after it, which is how the real fix below is
        written. Still far narrower than a character window -- this
        fails if the anchor is two sentences away, in a different
        paragraph, or absent.
        """
        text = LIMITATIONS.read_text()
        sentences = re.split(r"(?<=[.!?])\*{0,2}\s+", text)
        for i, sentence in enumerate(sentences):
            if not self._UNSCOPED_ZERO.search(sentence):
                continue
            nearby = sentence + " " + (sentences[i + 1] if i + 1 < len(sentences) else "")
            assert self._HISTORICAL_ANCHOR.search(nearby), (
                f"unscoped claim with no date/commit in its own or the next sentence: "
                f"{sentence!r} -- this is the exact shape of the bug this test exists to catch"
            )

    def test_readme_makes_no_such_claim_at_all(self) -> None:
        """README is reader-facing and current-state only; it should never
        need a historical caveat in the first place."""
        text = README.read_text()
        assert not self._UNSCOPED_ZERO.search(text)


class TestTheHistoricalClaimNamesItsOwnScope:
    """Non-vacuity for the two tests above: confirms the *current* wording
    is actually the kind of statement they are built to allow, not an
    accidentally-passing test with nothing to check."""

    def test_the_current_limitations_text_does_contain_a_scoped_historical_claim(self) -> None:
        text = LIMITATIONS.read_text()
        assert "Historical:" in text
        assert "ae22fcc" in text
        assert "2026-09-24" in text

    def test_the_current_limitations_text_also_states_the_present_counts(self) -> None:
        text = LIMITATIONS.read_text()
        assert "Current: the three committed recordings publish claims" in text
        for recording_id in RECORDINGS:
            assert recording_id in text
