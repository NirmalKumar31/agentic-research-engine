"""Score the development calibration cases with each candidate NLI model.

Writes raw per-pair probabilities and guard results, so the threshold
sweep is a pure function of a stored artifact rather than something that
needs a model rerun -- and so a reviewer can recompute the published
table without downloading 2GB of weights.

Run:  python examples/verifier-calibration/nli_calibrate.py
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agentic_research.citations.guards import run_guards  # noqa: E402
from agentic_research.citations.nli import NLIVerifier  # noqa: E402

HERE = Path(__file__).parent
CANDIDATES = {
    "deberta-v3-large": (
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",
    ),
    "deberta-v3-base": (
        "cross-encoder/nli-deberta-v3-base",
        "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
    ),
    "minilm2-l6-h768": (
        "cross-encoder/nli-MiniLM2-L6-H768",
        "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d",
    ),
}


def _rss_mb() -> float:
    """Resident set size. ru_maxrss is bytes on macOS, kilobytes on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def _current_rss_mb() -> float:
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def main() -> None:
    cases = json.loads((HERE / "blind_cases.json").read_text())["cases"]
    key = sys.argv[1] if len(sys.argv) > 1 else None
    targets = {key: CANDIDATES[key]} if key else CANDIDATES

    for name, (model_id, revision) in targets.items():
        baseline_rss = _current_rss_mb()
        verifier = NLIVerifier(model_id, revision, batch_size=8)

        load_start = time.perf_counter()
        verifier.score([("warm up the model", "warm up the model")])
        load_seconds = time.perf_counter() - load_start
        loaded_rss = _current_rss_mb()

        pairs: list[tuple[str, str]] = []
        index: list[tuple[str, str]] = []
        for case in cases:
            for item in case["evidence"]:
                pairs.append((item["quote"], case["claim"]))
                index.append((case["case_id"], item["evidence_id"]))

        infer_start = time.perf_counter()
        predictions = verifier.score(pairs)
        infer_seconds = time.perf_counter() - infer_start
        peak_rss = _rss_mb()

        by_case: dict[str, list[dict[str, object]]] = {}
        for (case_id, evidence_id), prediction in zip(index, predictions, strict=True):
            by_case.setdefault(case_id, []).append(
                {
                    "evidence_id": evidence_id,
                    "entailment": round(prediction.scores.entailment, 6),
                    "neutral": round(prediction.scores.neutral, 6),
                    "contradiction": round(prediction.scores.contradiction, 6),
                }
            )

        records = []
        for case in cases:
            scored = by_case[case["case_id"]]
            quotes = {e["evidence_id"]: e["quote"] for e in case["evidence"]}
            for entry in scored:
                guards = run_guards(case["claim"], quotes[str(entry["evidence_id"])])
                entry["guards"] = {g.name: {"passed": g.passed, "detail": g.detail} for g in guards}
                entry["guards_passed"] = all(g.passed for g in guards)
            records.append(
                {
                    "case_id": case["case_id"],
                    "recording": case["recording"],
                    "claim": case["claim"],
                    "human_label": case["human_label"],
                    "evidence": scored,
                }
            )

        artifact = {
            "model_id": model_id,
            "model_revision": revision,
            "cases": len(cases),
            "pairs": len(pairs),
            "runtime": {
                "model_load_seconds": round(load_seconds, 2),
                "inference_seconds": round(infer_seconds, 2),
                "seconds_per_pair": round(infer_seconds / len(pairs), 4),
            },
            "memory_mb": {
                "baseline_rss": round(baseline_rss, 1),
                "after_model_load_rss": round(loaded_rss, 1),
                "peak_rss": round(peak_rss, 1),
            },
            "scores": records,
        }
        out = HERE / f"nli-{name}-scores.json"
        out.write_text(json.dumps(artifact, indent=2))
        print(
            f"{name}: {len(pairs)} pairs in {infer_seconds:.1f}s "
            f"(load {load_seconds:.1f}s, peak RSS {peak_rss:.0f}MB) -> {out.name}"
        )


if __name__ == "__main__":
    main()
