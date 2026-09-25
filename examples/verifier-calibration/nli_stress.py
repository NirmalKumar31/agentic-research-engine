"""Run the adversarial stress suite through the real NLI models.

Separate from the development calibration on purpose: the calibration
cases shaped three verifier designs and cannot measure whether a fix
generalises. These cases are synthetic and were written from the
transformation classes, not from any observed failure.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nli_calibrate import CANDIDATES  # noqa: E402

from agentic_research.citations.nli import NLIVerifier  # noqa: E402
from agentic_research.citations.semantic import verify_claim  # noqa: E402

HERE = Path(__file__).parent
STRESS = ROOT / "tests" / "fixtures" / "semantic_stress.json"


def main() -> None:
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.98
    cases = json.loads(STRESS.read_text())["cases"]
    report: dict[str, object] = {"support_threshold": threshold, "models": {}}

    for name, (model_id, revision) in CANDIDATES.items():
        verifier = NLIVerifier(model_id, revision, batch_size=8)
        results, started = [], time.perf_counter()
        for case in cases:
            verdict = verify_claim(
                case["hypothesis"],
                [("E1", case["premise"])],
                verifier,
                support_threshold=threshold,
            )
            actual = "publish" if verdict.publishable else "withhold"
            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "expected": case["expected"],
                    "actual": actual,
                    "correct": actual == case["expected"],
                    "entailment": round(verdict.per_evidence[0].entailment, 4)
                    if verdict.per_evidence
                    else None,
                    "failed_guards": verdict.per_evidence[0].failed_guard_names
                    if verdict.per_evidence
                    else [],
                    "reason": verdict.reason,
                }
            )
        elapsed = time.perf_counter() - started

        unsafe = [r for r in results if r["expected"] == "withhold" and r["actual"] == "publish"]
        strict = [r for r in results if r["expected"] == "publish" and r["actual"] == "withhold"]
        report["models"][name] = {  # type: ignore[index]
            "model_id": model_id,
            "model_revision": revision,
            "cases": len(results),
            "correct": sum(1 for r in results if r["correct"]),
            "unsafe_publishes": unsafe,
            "over_withheld": strict,
            "seconds": round(elapsed, 2),
            "results": results,
        }
        print(
            f"{name}: {sum(1 for r in results if r['correct'])}/{len(results)} correct, "
            f"{len(unsafe)} unsafe publishes, {len(strict)} over-withheld ({elapsed:.1f}s)"
        )
        for r in unsafe:
            print(f"    UNSAFE {r['id']} ({r['category']}) ent={r['entailment']}")

    out = HERE / "nli-stress.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
