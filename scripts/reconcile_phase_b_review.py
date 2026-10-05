"""Reconcile both reviewers' Phase B scores: agreement, pairing, summary.

Run only after `scripts/join_phase_b_review.py --reviewer a` and
`--reviewer b` have both produced `joined_results.csv`. This is the
only place the two reviewers' scores are combined, and it never
collapses a disagreement into a single averaged number: every
dimension's two raw scores are preserved side by side, the absolute
difference is computed, and anything at or above the disagreement
threshold is flagged with an empty `adjudicated_*` column for a human
to resolve separately. Nothing here resolves a disagreement on its own.

Also separates the quality comparison into two groups, because
averaging them together is a selection-bias trap: the local arm timed
out on 5 of 24 repetitions, concentrated on relationship/comparison/
causal/ambiguous question shapes, so the 24 cloud outputs and the 19
local outputs that exist are not comparable samples of the same
population.

- **Paired** (19 question-repetitions where both arms produced output):
  the fair quality comparison.
- **Unpaired** (5 cloud-only outputs whose local counterpart timed out):
  reported descriptively, separately, explicitly labeled as selection-
  biased -- not pooled into the paired comparison.

Operational reliability (local 19/24 vs cloud 24/24 completed) is
computed directly from `raw_results.json`-derived status, independent
of anything a reviewer scored, and is restated unconditionally.

Produces `evaluations/phase_b/review/reconciliation.csv` (one row per
one of the 48 runs, both reviewers' raw scores, per-dimension abs diff
and disagreement flag, blank adjudication columns) and
`evaluations/phase_b/review/SUMMARY.md`.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PHASE_B = ROOT / "evaluations" / "phase_b"
REVIEW = PHASE_B / "review"

SCORE_COLUMNS = [
    "relevance_1to5",
    "completeness_1to5",
    "clarity_1to5",
    "claim_support_1to5",
    "citation_usefulness_1to5",
]
DISAGREEMENT_THRESHOLD = 2  # abs diff on a 1-5 scale at or above this is flagged


def load_joined(reviewer: str) -> dict[tuple[str, int, str], dict[str, str]]:
    path = REVIEW / f"reviewer_{reviewer}" / "joined_results.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_key = {(r["question_id"], int(r["repetition"]), r["arm"]): r for r in rows}
    if len(by_key) != len(rows):
        raise SystemExit(f"{path}: duplicate (question_id, repetition, arm) key found")
    return by_key


def reconcile() -> list[dict[str, Any]]:
    a = load_joined("a")
    b = load_joined("b")

    if set(a) != set(b):
        only_a = set(a) - set(b)
        only_b = set(b) - set(a)
        raise SystemExit(
            f"reviewers scored different run sets -- not bijective. "
            f"only in reviewer_a: {sorted(only_a)[:5]}; only in reviewer_b: {sorted(only_b)[:5]}"
        )
    if len(a) != 48:
        raise SystemExit(f"expected 48 runs, found {len(a)}")

    rows: list[dict[str, Any]] = []
    for key in sorted(a):
        qid, rep, arm = key
        ra, rb = a[key], b[key]
        if ra["status"] != rb["status"]:
            raise SystemExit(
                f"{key}: reviewer_a status={ra['status']!r} but reviewer_b "
                f"status={rb['status']!r} -- both reviewers score the same "
                "underlying runs, this should never happen"
            )
        row: dict[str, Any] = {
            "question_id": qid,
            "repetition": rep,
            "arm": arm,
            "status": ra["status"],
            "timed_out": ra["timed_out"],
            "duration_s": ra["duration_s"],
            "known_cost_usd": ra["known_cost_usd"],
        }
        if ra["status"] == "SCORED":
            for col in SCORE_COLUMNS:
                va, vb = int(ra[col]), int(rb[col])
                row[f"reviewer_a_{col}"] = va
                row[f"reviewer_b_{col}"] = vb
                row[f"abs_diff_{col}"] = abs(va - vb)
                row[f"disagreement_{col}"] = abs(va - vb) >= DISAGREEMENT_THRESHOLD
                row[f"adjudicated_{col}"] = ""
            row["reviewer_a_harmful_yn"] = ra["harmful_or_unsupported_claims_yn"]
            row["reviewer_b_harmful_yn"] = rb["harmful_or_unsupported_claims_yn"]
            row["harmful_disagreement"] = (
                ra["harmful_or_unsupported_claims_yn"] != rb["harmful_or_unsupported_claims_yn"]
            )
            row["adjudication_notes"] = ""
        else:
            for col in SCORE_COLUMNS:
                row[f"reviewer_a_{col}"] = ""
                row[f"reviewer_b_{col}"] = ""
                row[f"abs_diff_{col}"] = ""
                row[f"disagreement_{col}"] = ""
                row[f"adjudicated_{col}"] = ""
            row["reviewer_a_harmful_yn"] = ""
            row["reviewer_b_harmful_yn"] = ""
            row["harmful_disagreement"] = ""
            row["adjudication_notes"] = ""
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    path = REVIEW / "reconciliation.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def paired_and_unpaired(
    rows: list[dict[str, Any]],
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    scored_reps: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        if row["status"] == "SCORED":
            scored_reps.setdefault((row["question_id"], row["repetition"]), set()).add(row["arm"])
    paired = {k for k, arms in scored_reps.items() if arms == {"local", "cloud"}}
    unpaired = {k for k, arms in scored_reps.items() if arms != {"local", "cloud"}}
    return paired, unpaired


def write_summary(rows: list[dict[str, Any]]) -> None:
    paired_reps, unpaired_reps = paired_and_unpaired(rows)

    lines = ["# Phase B blinded review -- reconciled descriptive summary\n"]
    lines.append(
        "**Reviewers: two independent Codex (AI) sessions, not the human "
        "reviewers docs/BENCHMARK-PROTOCOL.md specifies -- a disclosed "
        "deviation, not an implied substitute. n=2 repetitions per "
        "question. All numbers below are descriptive means over scored "
        "outputs, never blended into a single pooled number without "
        "showing both raw scores. This is not statistical significance "
        "evidence.**\n"
    )

    lines.append("## Paired quality comparison (both arms produced output)\n")
    lines.append(
        f"{len(paired_reps)} question-repetitions where both arms completed "
        f"({len(paired_reps) * 2} scored outputs, {len(paired_reps)} per arm). "
        "This is the only fair arm-vs-arm quality comparison in this document.\n"
    )
    lines.append("| Dimension | local: a | local: b | cloud: a | cloud: b |")
    lines.append("|---|---|---|---|---|")
    for col in SCORE_COLUMNS:
        cells = []
        for arm in ("local", "cloud"):
            for reviewer in ("a", "b"):
                values = [
                    row[f"reviewer_{reviewer}_{col}"]
                    for row in rows
                    if row["arm"] == arm
                    and row["status"] == "SCORED"
                    and (row["question_id"], row["repetition"]) in paired_reps
                ]
                cells.append(
                    f"{statistics.mean(values):.2f} (n={len(values)})" if values else "n/a"
                )
        lines.append(f"| {col} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

    disagreements = [
        (row["question_id"], row["repetition"], row["arm"], col)
        for row in rows
        if row["status"] == "SCORED"
        for col in SCORE_COLUMNS
        if row[f"disagreement_{col}"] is True
    ]
    lines.append(
        f"\n**Inter-rater disagreements (|diff| >= {DISAGREEMENT_THRESHOLD}): "
        f"{len(disagreements)}.** Not averaged away -- see `reconciliation.csv`'s "
        "`adjudicated_*` columns, left blank pending manual resolution."
    )
    for qid, rep, arm, col in disagreements:
        lines.append(f"  - {qid} rep{rep} {arm}: {col}")

    harmful_disagreements = [
        row for row in rows if row["status"] == "SCORED" and row["harmful_disagreement"] is True
    ]
    if harmful_disagreements:
        lines.append(
            f"\n**Harmful/unsupported-claims flag disagreement: {len(harmful_disagreements)}.** "
            "One reviewer flagged, the other did not -- needs adjudication, not averaging."
        )
        for row in harmful_disagreements:
            lines.append(f"  - {row['question_id']} rep{row['repetition']} {row['arm']}")

    lines.append("\n## Unpaired cloud-only outputs (selection-biased, reported separately)\n")
    lines.append(
        f"{len(unpaired_reps)} question-repetition(s) where the cloud arm produced "
        "output but the local arm timed out on that specific repetition. These "
        "exist only because of that timeout -- including them in the paired "
        "comparison above would silently compare cloud's performance on the "
        "full question set against local's performance on an easier subset. "
        "Reported here, separately, not averaged into the paired table.\n"
    )
    for col in SCORE_COLUMNS:
        values = [
            row[f"reviewer_a_{col}"]
            for row in rows
            if row["arm"] == "cloud"
            and row["status"] == "SCORED"
            and (row["question_id"], row["repetition"]) in unpaired_reps
        ] + [
            row[f"reviewer_b_{col}"]
            for row in rows
            if row["arm"] == "cloud"
            and row["status"] == "SCORED"
            and (row["question_id"], row["repetition"]) in unpaired_reps
        ]
        if values:
            lines.append(
                f"- {col}: mean {statistics.mean(values):.2f} "
                f"(n={len(values)}, both reviewers pooled)"
            )

    lines.append("\n## Operational reliability (unconditional, independent of any quality score)\n")
    local_total = sum(1 for r in rows if r["arm"] == "local")
    local_timeout = sum(1 for r in rows if r["arm"] == "local" and r["timed_out"] == "True")
    cloud_total = sum(1 for r in rows if r["arm"] == "cloud")
    cloud_timeout = sum(1 for r in rows if r["arm"] == "cloud" and r["timed_out"] == "True")
    lines.append(
        f"- local: {local_total - local_timeout}/{local_total} completed, "
        f"{local_timeout}/{local_total} timed out"
    )
    lines.append(
        f"- cloud: {cloud_total - cloud_timeout}/{cloud_total} completed, "
        f"{cloud_timeout}/{cloud_total} timed out"
    )
    lines.append(
        "\nThese counts hold regardless of this review's quality scores -- a "
        "timeout produced no output to score and is not reinterpreted as a "
        "quality judgment of any kind."
    )

    (REVIEW / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REVIEW / 'SUMMARY.md'}")


def main() -> None:
    rows = reconcile()
    write_csv(rows)
    write_summary(rows)


if __name__ == "__main__":
    main()
