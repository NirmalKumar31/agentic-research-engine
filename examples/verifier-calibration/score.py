"""Score a verifier's verdicts against the human labels in cases.json.

Run:
    python examples/verifier-calibration/score.py [cases.json]

Reports exact agreement, supported-precision and -recall, false
positives and negatives, and the confusion matrix.

A false positive -- verifier says supported, human says not -- is the
expensive direction: it means an unsupported claim reached the report. A
false negative costs a true claim its place, which is a worse demo and a
safer failure.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

LABELS = ("supported", "partially_supported", "unsupported")


def score(cases: list[dict]) -> dict:
    labelled = [c for c in cases if c.get("human_label")]
    if not labelled:
        return {"labelled": 0}

    agree = sum(1 for c in labelled if c["human_label"] == c["verifier_verdict"])
    # "Supported" is the only verdict that publishes, so precision and
    # recall are defined against it rather than averaged over all three.
    tp = sum(
        1
        for c in labelled
        if c["verifier_verdict"] == "supported" and c["human_label"] == "supported"
    )
    fp = sum(
        1
        for c in labelled
        if c["verifier_verdict"] == "supported" and c["human_label"] != "supported"
    )
    fn = sum(
        1
        for c in labelled
        if c["verifier_verdict"] != "supported" and c["human_label"] == "supported"
    )

    matrix = Counter((c["human_label"], c["verifier_verdict"]) for c in labelled)
    return {
        "labelled": len(labelled),
        "unlabelled": len(cases) - len(labelled),
        "agreement": round(agree / len(labelled), 4),
        "supported_precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "supported_recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "false_positives": fp,
        "false_negatives": fn,
        "matrix": matrix,
    }


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "cases.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    result = score(doc["cases"])

    if not result["labelled"]:
        print(f"{len(doc['cases'])} cases, none labelled yet.")
        print("Set human_label on each case, then re-run.")
        return 1

    print(f"labelled            {result['labelled']} of {len(doc['cases'])}")
    print(f"exact agreement     {result['agreement']:.1%}")
    print(f"supported precision {result['supported_precision']}")
    print(f"supported recall    {result['supported_recall']}")
    print(f"false positives     {result['false_positives']}  (unsupported claim published)")
    print(f"false negatives     {result['false_negatives']}  (true claim removed)")
    print("\nconfusion matrix (human -> verifier):")
    for human in LABELS:
        row = "  ".join(f"{result['matrix'].get((human, v), 0):>3}" for v in LABELS)
        print(f"  {human:>20}  {row}")
    print(f"  {'':>20}  " + "  ".join(f"{v[:3]:>3}" for v in LABELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
