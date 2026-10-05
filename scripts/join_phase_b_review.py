"""Validate, unblind and join one reviewer's completed Phase B scores.

Run only after `evaluations/phase_b/review/reviewer_<a|b>/scores_template.csv`
has been filled in by that reviewer. This is the one point where that
reviewer's random IDs are resolved back to question/repetition/arm --
everything before this script runs stays blind. Each reviewer is
joined independently; `scripts/reconcile_phase_b_review.py` is the
separate step that combines both reviewers and computes agreement.

Validation is strict and fails closed on any of:
- a score field that is missing, non-integer, or outside 1-5
- a harmful_or_unsupported_claims_yn value that is not exactly Yes/No
- an empty rationale
- a duplicate random_id
- the CSV's random_id set not matching the key's reviewable set exactly
  (bijective: no extra IDs, none missing)
- a row's question_id not matching what the key recorded for that
  random_id (catches a reviewer copy-pasting into the wrong row)

Produces `evaluations/phase_b/review/reviewer_<x>/joined_results.csv`:
one row per one of the 48 preregistered runs (43 scored + 5 timeouts).
The 5 timeouts are included with blank score columns and an explicit
`status=TIMEOUT_NO_OUTPUT` marker -- never dropped, never given a
fabricated score, regardless of what the scored rows show.
"""

from __future__ import annotations

import argparse
import csv
import json
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


class ReviewValidationError(Exception):
    """The reviewer's score sheet is not yet valid to join."""


def load_raw_results() -> dict[tuple[str, int, str], dict]:
    raw = json.loads((PHASE_B / "raw_results.json").read_text())
    return {(r["question_id"], r["repetition"], r["arm"]): r for r in raw}


def validate_and_load(reviewer_dir: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    scores_path = reviewer_dir / "scores_template.csv"
    key_path = reviewer_dir / "unblinding_key.json"
    with scores_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    key = json.loads(key_path.read_text())
    reviewable_key = key["reviewable"]

    problems: list[str] = []

    seen_ids: set[str] = set()
    for i, row in enumerate(rows):
        rid = row.get("random_id", "").strip()
        label = rid or f"row {i}"
        if not rid:
            problems.append(f"{label}: empty random_id")
            continue
        if rid in seen_ids:
            problems.append(f"{rid}: duplicate random_id")
        seen_ids.add(rid)

        for col in SCORE_COLUMNS:
            value = row.get(col, "").strip()
            if not value:
                problems.append(f"{rid}: {col} is blank")
                continue
            try:
                parsed = int(value)
            except ValueError:
                problems.append(f"{rid}: {col}={value!r} is not an integer")
                continue
            if not (1 <= parsed <= 5):
                problems.append(f"{rid}: {col}={parsed} is outside 1-5")

        flag = row.get("harmful_or_unsupported_claims_yn", "").strip()
        if flag.lower() not in ("yes", "no"):
            problems.append(
                f"{rid}: harmful_or_unsupported_claims_yn={flag!r} must be exactly Yes or No"
            )

        if not row.get("rationale", "").strip():
            problems.append(f"{rid}: rationale is empty")

        mapping = reviewable_key.get(rid)
        if mapping is None:
            problems.append(f"{rid}: not present in unblinding_key.json at all")
        elif mapping["question_id"] != row.get("question_id", "").strip():
            problems.append(
                f"{rid}: question_id={row.get('question_id')!r} in the CSV does not "
                f"match {mapping['question_id']!r} recorded in the key -- "
                "check for a copy-paste into the wrong row"
            )

    # Not `expected_ids - seen_ids` *and* `seen_ids - expected_ids`: an ID
    # in the sheet but not the key is already caught per-row above
    # (`mapping is None`), so checking it again here would be dead code
    # that can never add a problem the per-row loop has not already
    # raised. Only "the key expected this ID and it never appeared at
    # all" needs a separate pass, since the per-row loop only sees rows
    # that exist.
    missing = set(reviewable_key) - seen_ids
    if missing:
        problems.append(
            f"{len(missing)} expected ID(s) missing from the score sheet: {sorted(missing)[:5]}"
        )

    if problems:
        raise ReviewValidationError(
            f"{len(problems)} problem(s) in {scores_path}:\n  - " + "\n  - ".join(problems)
        )

    return rows, key


def join(reviewer: str) -> list[dict[str, Any]]:
    reviewer_dir = REVIEW / f"reviewer_{reviewer}"
    rows, key = validate_and_load(reviewer_dir)
    raw_by_key = load_raw_results()

    joined: list[dict[str, Any]] = []
    for row in rows:
        mapping = key["reviewable"][row["random_id"]]
        raw = raw_by_key[(mapping["question_id"], mapping["repetition"], mapping["arm"])]
        joined.append(
            {
                "question_id": mapping["question_id"],
                "repetition": mapping["repetition"],
                "arm": mapping["arm"],
                "random_id": row["random_id"],
                "status": "SCORED",
                **{c: row[c] for c in SCORE_COLUMNS},
                "harmful_or_unsupported_claims_yn": row["harmful_or_unsupported_claims_yn"]
                .strip()
                .title(),
                "harmful_or_unsupported_claims_note": row["harmful_or_unsupported_claims_note"],
                "rationale": row["rationale"],
                "duration_s": raw["duration_s"],
                "timed_out": raw["timed_out"],
                "known_cost_usd": raw["known_cost_usd"],
                "automated_metrics": raw["metrics"],
            }
        )

    for entry in key["unreviewable_timeouts"]:
        raw = raw_by_key[(entry["question_id"], entry["repetition"], entry["arm"])]
        joined.append(
            {
                "question_id": entry["question_id"],
                "repetition": entry["repetition"],
                "arm": entry["arm"],
                "random_id": "",
                "status": "TIMEOUT_NO_OUTPUT",
                **dict.fromkeys(SCORE_COLUMNS, ""),
                "harmful_or_unsupported_claims_yn": "",
                "harmful_or_unsupported_claims_note": "",
                "rationale": "",
                "duration_s": raw["duration_s"],
                "timed_out": raw["timed_out"],
                "known_cost_usd": raw["known_cost_usd"],
                "automated_metrics": {},
            }
        )

    expected = len(key["reviewable"]) + len(key["unreviewable_timeouts"])
    if len(joined) != expected:
        raise ReviewValidationError(
            f"the key describes {expected} runs but joining produced {len(joined)} rows"
        )
    return joined


def write_csv(reviewer: str, joined: list[dict[str, Any]]) -> Path:
    path = REVIEW / f"reviewer_{reviewer}" / "joined_results.csv"
    fieldnames = [k for k in joined[0] if k != "automated_metrics"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in joined:
            writer.writerow({k: v for k, v in row.items() if k != "automated_metrics"})
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", choices=["a", "b"], required=True)
    args = parser.parse_args()
    joined = join(args.reviewer)
    path = write_csv(args.reviewer, joined)
    scored = sum(1 for r in joined if r["status"] == "SCORED")
    timeouts = sum(1 for r in joined if r["status"] == "TIMEOUT_NO_OUTPUT")
    print(f"Wrote {path} ({scored} scored, {timeouts} timeouts, {len(joined)} total)")


if __name__ == "__main__":
    main()
