"""Select the support threshold from stored NLI scores.

Pure function of the artifacts written by nli_calibrate.py, so the
published table can be recomputed without any model. The selection rule
is fixed in advance and is not negotiable against the result:

    among thresholds with zero supported false positives,
    take the highest supported recall.

A false positive here is a claim the reviewer judged partially supported
or unsupported that the gate would publish. That is the failure this
whole exercise exists to remove, so it is a hard constraint rather than
one term in a blended score.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(50)]  # 0.50 .. 0.99


@dataclass
class Row:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        published = self.tp + self.fp
        return self.tp / published if published else 1.0

    @property
    def recall(self) -> float:
        positives = self.tp + self.fn
        return self.tp / positives if positives else 0.0


def publishes(case: dict, threshold: float) -> tuple[bool, str | None, float]:
    """Publication rule: some guard-passing quote entails at/above threshold."""
    valid = [e for e in case["evidence"] if e["guards_passed"]]
    if not valid:
        return False, None, max((e["entailment"] for e in case["evidence"]), default=0.0)
    best = max(valid, key=lambda e: e["entailment"])
    return best["entailment"] >= threshold, best["evidence_id"], best["entailment"]


def sweep(cases: list[dict]) -> list[Row]:
    rows = []
    for threshold in THRESHOLDS:
        tp = fp = fn = tn = 0
        for case in cases:
            published, _, _ = publishes(case, threshold)
            positive = case["human_label"] == "supported"
            if published and positive:
                tp += 1
            elif published:
                fp += 1
            elif positive:
                fn += 1
            else:
                tn += 1
        rows.append(Row(threshold, tp, fp, fn, tn))
    return rows


def select(rows: list[Row]) -> Row | None:
    clean = [r for r in rows if r.fp == 0]
    if not clean:
        return None
    # Highest recall; ties broken toward the higher threshold, which
    # keeps the larger safety margin for the same measured result.
    return max(clean, key=lambda r: (r.recall, r.threshold))


def main() -> None:
    report: dict[str, dict] = {}
    for path in sorted(HERE.glob("nli-*-scores.json")):
        artifact = json.loads(path.read_text())
        name = path.stem.replace("nli-", "").replace("-scores", "")
        rows = sweep(artifact["scores"])
        chosen = select(rows)

        print(f"\n=== {name}  ({artifact['model_id']})")
        print(f"    revision {artifact['model_revision']}")
        print("    thr   TP  FP  FN  TN   prec   recall")
        shown = {r.threshold for r in rows[::5]} | ({chosen.threshold} if chosen else set())
        for r in rows:
            if r.threshold in shown:
                mark = " <- selected" if chosen and r.threshold == chosen.threshold else ""
                print(
                    f"   {r.threshold:.2f}  {r.tp:2d}  {r.fp:2d}  {r.fn:2d}  {r.tn:2d}"
                    f"   {r.precision:.2f}   {r.recall:.2f}{mark}"
                )
        if chosen is None:
            print("    NO THRESHOLD REACHES ZERO FALSE POSITIVES -- model rejected")
        else:
            print(
                f"    selected threshold {chosen.threshold:.2f}: "
                f"TP={chosen.tp} FP={chosen.fp} FN={chosen.fn} recall={chosen.recall:.2f}"
            )

        report[name] = {
            "model_id": artifact["model_id"],
            "model_revision": artifact["model_revision"],
            "runtime": artifact["runtime"],
            "memory_mb": artifact["memory_mb"],
            "sweep": [
                {
                    "threshold": r.threshold,
                    "tp": r.tp,
                    "fp": r.fp,
                    "fn": r.fn,
                    "tn": r.tn,
                    "precision": round(r.precision, 4),
                    "recall": round(r.recall, 4),
                }
                for r in rows
            ],
            "selected_threshold": chosen.threshold if chosen else None,
            "selected": (
                {
                    "tp": chosen.tp,
                    "fp": chosen.fp,
                    "fn": chosen.fn,
                    "tn": chosen.tn,
                    "precision": round(chosen.precision, 4),
                    "recall": round(chosen.recall, 4),
                }
                if chosen
                else None
            ),
        }

        if chosen is not None:
            fps, fns = [], []
            for case in artifact["scores"]:
                published, best_id, best = publishes(case, chosen.threshold)
                positive = case["human_label"] == "supported"
                entry = {
                    "case_id": case["case_id"],
                    "claim": case["claim"],
                    "human_label": case["human_label"],
                    "best_evidence_id": best_id,
                    "best_entailment": round(best, 4),
                }
                if published and not positive:
                    fps.append(entry)
                elif positive and not published:
                    entry["failed_guards"] = sorted(
                        {
                            name
                            for e in case["evidence"]
                            for name, g in e["guards"].items()
                            if not g["passed"]
                        }
                    )
                    fns.append(entry)
            report[name]["false_positives"] = fps
            report[name]["false_negatives"] = fns

    (HERE / "nli-calibration.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {(HERE / 'nli-calibration.json').name}")
    if not report:
        sys.exit("no score artifacts found")


if __name__ == "__main__":
    main()
