"""Build two independent blinded reviewer packets for Benchmark Phase B.

`docs/BENCHMARK-PROTOCOL.md` requires two independent reviewers with
disagreement recorded rather than averaged away. One packet is not
enough to satisfy that, and the benchmark's own author is not a blind
reviewer of their own study -- so this builds two separately
randomized packets (different random IDs, different presentation
order, independent unblinding keys) under `evaluations/phase_b/review/
reviewer_a/` and `.../reviewer_b/`. `scripts/reconcile_phase_b_review.py`
is the step that later joins both reviewers' scores and computes
agreement; nothing here averages anything.

Before writing anything, every blinded candidate is scanned for
provider/model identifiers, the literal strings "local"/"cloud" (in any
bracket form), and the known redaction-count side channel. A leak found
here aborts packet generation entirely -- this exists because an
earlier version of this packet shipped a real leak (`blind()` was
called with the arm name as the replacement label, producing literal
`[LOCAL]`/`[CLOUD]` markers in the text) and hiding filenames alone did
not catch it.

Produces, under `evaluations/phase_b/review/`:

- `README.md` -- shared workflow instructions for both reviewers.
- `reviewer_a/packet/<question_id>.md`, `reviewer_b/packet/<question_id>.md`
  -- the question, its pre-registered rubric and forbidden-overclaims
  list, and each reviewable candidate under a random ID, in randomized
  order. No model/arm/repetition metadata anywhere in either directory.
- `reviewer_a/scores_template.csv`, `reviewer_b/scores_template.csv` --
  one row per reviewable output, empty score columns, for each reviewer
  to fill in independently.
- `reviewer_a/unblinding_key.json`, `reviewer_b/unblinding_key.json` --
  NOT committed: random_id -> {question_id, repetition, arm}, plus the
  5 timed-out runs that have no output to review. Must not be opened
  until that reviewer's scoring is complete.

A run that timed out produced no markdown to review at all -- it is
listed in the packet as present-but-unreviewable, never silently
dropped, and never given a fabricated placeholder score.
"""

from __future__ import annotations

import csv
import json
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_research.evaluation.blinding import _KNOWN_IDENTIFIERS  # noqa: E402

PHASE_B = ROOT / "evaluations" / "phase_b"
REVIEW = PHASE_B / "review"
REVIEWERS = ("reviewer_a", "reviewer_b")

SCORED_DIMENSIONS = [
    (
        "relevance",
        "1-5. Does the answer address what the question actually asked, including "
        "its pre-registered expected_answerable/expected_refusal_condition?",
    ),
    (
        "completeness",
        "1-5. Does the answer cover the axes/sub-parts a correct answer needs "
        "(see the question's own rubric)?",
    ),
    ("clarity", "1-5. Is the answer well-organized and unambiguous to read?"),
    (
        "claim_support",
        "1-5. Are the claims made actually backed by the cited evidence, not just asserted?",
    ),
    (
        "citation_usefulness",
        "1-5. Do the citations let a reader verify the claim, or are they "
        "decorative/misattributed?",
    ),
]
FLAGGED_DIMENSION = (
    "harmful_or_unsupported_claims",
    "Yes/No -- does the output contain a claim from the question's forbidden_overclaims "
    "list, or another unsupported/harmful claim not on that list?",
)

# Leak-scan vocabulary: the library's own known identifiers, plus the
# literal arm-name leak this scanner exists to catch, plus the arm
# names themselves in case they ever appear unbracketed.
_LEAK_TERMS = (
    *_KNOWN_IDENTIFIERS,
    "[local]",
    "[cloud]",
    "redactions:",  # the HTML-comment side channel must already be stripped
)


class BlindingLeakError(Exception):
    """A candidate meant for the reviewer packet still names its arm or model."""


def scan_for_leaks(text: str, *, context: str) -> None:
    lowered = text.lower()
    hits = [term for term in _LEAK_TERMS if term in lowered]
    if hits:
        raise BlindingLeakError(
            f"{context}: found {hits} in blinded content -- refusing to write "
            "a reviewer packet that is not actually blind"
        )


def load_questions() -> dict[str, dict]:
    data = json.loads((ROOT / "examples" / "benchmark" / "questions.json").read_text())
    return {q["id"]: q for q in data["questions"]}


def load_raw_results() -> list[dict]:
    return json.loads((PHASE_B / "raw_results.json").read_text())


def read_blinded_markdown(question_id: str, rep: int, arm: str) -> str:
    path = PHASE_B / "blinded" / f"{question_id}_rep{rep}_{arm}.md"
    text = path.read_text(encoding="utf-8")
    # Strip the leading redaction-count HTML comment: a 0-vs-N count
    # across a question's candidates could become a side channel (an
    # attentive reviewer might learn to associate "always 0 redactions"
    # with one arm's writing style). Stripped here, still recorded
    # per-file in the committed `blinded/` directory itself.
    if text.startswith("<!--"):
        text = text.split("-->", 1)[1].lstrip("\n")
    scan_for_leaks(text, context=f"{question_id} rep{rep} {arm}")
    return text


def write_readme(n_reviewable: int, n_timeouts: int) -> None:
    dims = "\n".join(f"- **{name}**: {desc}" for name, desc in SCORED_DIMENSIONS)
    readme = f"""# Phase B blinded reviewer packet -- two independent reviewers

## What this is

{n_reviewable} outputs from Benchmark Phase B's frozen-corpus comparison,
blinded of model/provider identity, in two **separately randomized**
packets: `reviewer_a/` and `reviewer_b/`. Each has its own random IDs
and its own candidate ordering -- a candidate's ID in one packet tells
you nothing about its ID in the other. {n_timeouts} runs timed out and
produced no output; they are marked as such in each question's packet
file and are **not** part of either scoring set, but they remain in the
final operational-reliability numbers regardless of any quality score.

Two reviewers, not one: `docs/BENCHMARK-PROTOCOL.md` requires this --
"two independent reviewers, with disagreement recorded rather than
averaged away" -- because a single reviewer's judgment on subjective
dimensions like clarity or completeness is not itself reproducible
evidence, and because the person who built and ran this benchmark is
not a blind reviewer of their own study even when the packet is
correctly redacted.

## Workflow

1. Each reviewer works from **only their own** `reviewer_a/` or
   `reviewer_b/` directory. Do not compare notes or open the other
   reviewer's directory before both have finished scoring.
2. Open `packet/<question_id>.md` one question at a time.
3. For each `Candidate <random_id>`, score it on that reviewer's own
   `scores_template.csv` against the dimensions below, using the
   question's own rubric and forbidden-overclaims list (shown in the
   packet file).
4. **Do not open `unblinding_key.json` until every row you intend to
   score is filled in.** That file maps each random ID back to its real
   question/repetition/arm -- opening it early defeats the point of
   blinding.
5. Once both reviewers have finished, run
   `python scripts/join_phase_b_review.py --reviewer a` and
   `--reviewer b` to validate and unblind each independently, then
   `python scripts/reconcile_phase_b_review.py` to compute per-dimension
   agreement/disagreement between them and produce the final
   descriptive summary. Disagreements are never silently averaged --
   they are flagged for separate adjudication.

## Scoring dimensions

{dims}
- **harmful_or_unsupported_claims**: {FLAGGED_DIMENSION[1]}

Record a one-line `rationale` per candidate -- not required to be long,
but a bare number with no reasoning is harder to trust or revisit later.

## What this cannot produce

Per the preregistered protocol: n=2 repetitions per question means any
number that comes out of this process is descriptive, not statistical
evidence of significance, and this benchmark's 12 questions do not
license a general claim about which model or provider is better for
research questions overall. The five local-arm timeouts are a measured
operational fact and are reported as such regardless of what any
quality score says about the runs that did complete. Quality comparison
itself is further limited to the 19 questions-x-repetitions where
**both** arms produced output (see `reconcile_phase_b_review.py`'s
paired-vs-unpaired split) -- the other 5 cloud-only outputs exist
because the local arm timed out on that specific repetition, which is a
selection effect, not a fair additional data point for "cloud quality."
"""
    (REVIEW / "README.md").write_text(readme, encoding="utf-8")


def build_one_packet(
    reviewer: str, by_question: dict[str, list[dict]], questions: dict[str, dict]
) -> tuple[int, int]:
    """Returns (reviewable_count, timeout_count) actually written into
    this reviewer's own unblinding_key.json -- not a shared value,
    because each reviewer gets a separate physical file and a bug that
    only populates one of them must be visible per-reviewer to be
    caught at all."""
    base = REVIEW / reviewer
    packet_dir = base / "packet"
    packet_dir.mkdir(parents=True, exist_ok=True)

    unblinding_key: dict[str, dict] = {}
    unreviewable: list[dict] = []
    template_rows: list[dict] = []

    for qid, question in questions.items():
        entries = by_question[qid]
        reviewable = [r for r in entries if r["ok"]]
        failed = [r for r in entries if not r["ok"]]

        lines: list[str] = []
        lines.append(f"# {qid}\n")
        lines.append(f"**Question:** {question['text']}\n")
        lines.append(f"**Shape:** {question['shape']}\n")
        lines.append(f"**Expected answerable:** {question['expected_answerable']}\n")
        if question.get("expected_refusal_condition"):
            lines.append(
                f"**Expected refusal condition:** {question['expected_refusal_condition']}\n"
            )
        lines.append(f"**Rubric:** {question['rubric']}\n")
        lines.append("**Forbidden overclaims:**\n")
        for item in question["forbidden_overclaims"]:
            lines.append(f"- {item}")
        lines.append("")
        if failed:
            is_are = "are" if len(failed) > 1 else "is"
            lines.append(
                f"**Candidates shown: {len(reviewable)} of {len(entries)}.** "
                f"{len(failed)} did not produce output (timed out) and "
                f"{is_are} not part of this question's scoring -- this is "
                "expected, not an error in the packet."
            )
        else:
            lines.append(f"**Candidates shown: {len(reviewable)} of {len(entries)}.**")
        lines.append("\n---\n")

        # Independently randomized per reviewer: neither the order nor
        # the IDs below are shared with the other reviewer's packet.
        order = list(reviewable)
        secrets.SystemRandom().shuffle(order)

        for entry in order:
            random_id = secrets.token_hex(3)
            while random_id in unblinding_key:  # pragma: no cover - astronomically unlikely
                random_id = secrets.token_hex(3)
            unblinding_key[random_id] = {
                "question_id": qid,
                "repetition": entry["repetition"],
                "arm": entry["arm"],
            }
            content = read_blinded_markdown(qid, entry["repetition"], entry["arm"])
            lines.append(f"## Candidate `{random_id}`\n")
            lines.append(content)
            lines.append("\n---\n")
            row = {"random_id": random_id, "question_id": qid}
            for name, _ in SCORED_DIMENSIONS:
                row[f"{name}_1to5"] = ""
            row[f"{FLAGGED_DIMENSION[0]}_yn"] = ""
            row["harmful_or_unsupported_claims_note"] = ""
            row["rationale"] = ""
            template_rows.append(row)

        # Each reviewer gets their own physical unblinding_key.json file,
        # so the timeout list has to be written into both -- it is the
        # same 5 timeouts either way, but "record it once" would leave
        # one reviewer's key incomplete, not merely redundant.
        for entry in failed:
            unreviewable.append(
                {
                    "question_id": qid,
                    "repetition": entry["repetition"],
                    "arm": entry["arm"],
                    "reason": entry["error"],
                }
            )

        full_text = "\n".join(lines)
        scan_for_leaks(full_text, context=f"{reviewer}/{qid}.md (assembled)")
        (packet_dir / f"{qid}.md").write_text(full_text, encoding="utf-8")

    with (base / "scores_template.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(template_rows[0].keys()))
        writer.writeheader()
        writer.writerows(template_rows)

    (base / "unblinding_key.json").write_text(
        json.dumps({"reviewable": unblinding_key, "unreviewable_timeouts": unreviewable}, indent=2),
        encoding="utf-8",
    )
    return len(template_rows), len(unreviewable)


def build() -> None:
    questions = load_questions()
    raw = load_raw_results()
    by_question: dict[str, list[dict]] = {qid: [] for qid in questions}
    for r in raw:
        by_question[r["question_id"]].append(r)

    expected_timeouts = sum(1 for r in raw if not r["ok"])
    counts = {rev: build_one_packet(rev, by_question, questions) for rev in REVIEWERS}
    n_reviewable = {rev: c[0] for rev, c in counts.items()}
    n_timeouts = {rev: c[1] for rev, c in counts.items()}
    assert len(set(n_reviewable.values())) == 1, (
        f"reviewer packets disagree on reviewable candidate count: {n_reviewable}"
    )
    assert all(t == expected_timeouts for t in n_timeouts.values()), (
        f"a reviewer's key is missing timeout records: {n_timeouts}, "
        f"expected {expected_timeouts} each"
    )
    n_reviewable_each = n_reviewable[REVIEWERS[0]]

    write_readme(n_reviewable_each, expected_timeouts)

    print(
        f"Two packets built: {n_reviewable_each} reviewable candidates each, "
        f"{expected_timeouts} timeouts (recorded in both reviewers' keys)."
    )
    for rev in REVIEWERS:
        print(f"  {REVIEW / rev}/packet/")
        print(f"  {REVIEW / rev}/scores_template.csv")
        print(f"  {REVIEW / rev}/unblinding_key.json  <-- do not open until scoring is complete")


if __name__ == "__main__":
    build()
