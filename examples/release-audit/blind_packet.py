"""Build a blinded review packet, and later join the labels back.

The previous audits were reviewed with the automated verdict visible.
That is how confirmation bias gets in: a claim the system published
reads as supported, and a claim it withheld reads as an overreach,
before the evidence is examined at all.

So the packet shows only what a reviewer needs to judge the pair --
the claim, the exact quote the gate selected, and who published it --
and hides the verdict, the score, the guard results and whether it was
published. Cases are shuffled under a recorded seed so the order
carries no signal either.

This is still not an independent benchmark: the reviewer is the same
party that built the system. It is a blinded release audit, and should
be called that.

    python examples/release-audit/blind_packet.py build
    python examples/release-audit/blind_packet.py join
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
AUDIT = HERE / "candidate-audit.json"
PACKET = HERE / "blind-packet.json"
LABELS = HERE / "blind-labels.json"
JOINED = HERE / "blind-audit.json"
SEED = 20260926


def build() -> None:
    audit = json.loads(AUDIT.read_text())
    cases = []
    for run_id, run in audit["runs"].items():
        for index, candidate in enumerate(run["candidates"]):
            best = candidate.get("best_evidence_id")
            chosen = next(
                (e for e in candidate["evidence"] if e["evidence_id"] == best),
                (candidate["evidence"] or [{}])[0],
            )
            cases.append(
                {
                    "case_id": f"{run_id}-{index}",
                    "claim": candidate["claim"],
                    "claim_kind": candidate.get("kind"),
                    "evidence_id": chosen.get("evidence_id"),
                    "quote": chosen.get("quote"),
                    "source_id": chosen.get("source_id"),
                    "page": chosen.get("page"),
                    # Everything the system decided is withheld here and
                    # rejoined by case_id afterwards.
                }
            )
    random.Random(SEED).shuffle(cases)
    PACKET.write_text(
        json.dumps(
            {
                "description": (
                    "Blinded release audit packet. Each case shows a generated claim and the "
                    "exact evidence quote the verifier selected for it, with no automated "
                    "verdict, score, guard result or publication decision. Label each "
                    "supported / unsupported / uncertain, then run `join`."
                ),
                "instructions": (
                    "supported: this quote, by itself, states or directly implies the claim "
                    "with its scope, modality, quantities and attribution intact. "
                    "unsupported: it does not. uncertain: you cannot tell from this quote alone."
                ),
                "shuffle_seed": SEED,
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {PACKET.name}: {len(cases)} cases, seed {SEED}")


def join() -> None:
    audit = json.loads(AUDIT.read_text())
    packet = json.loads(PACKET.read_text())
    labels = json.loads(LABELS.read_text())

    flat = {}
    for run_id, run in audit["runs"].items():
        for index, candidate in enumerate(run["candidates"]):
            flat[f"{run_id}-{index}"] = (run_id, candidate)

    rows: list[dict[str, object]] = []
    matrix = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    unsupported_published: list[dict[str, str]] = []
    uncertain_published: list[dict[str, str]] = []
    for case in packet["cases"]:
        case_id = case["case_id"]
        run_id, candidate = flat[case_id]
        human = labels[case_id]
        published = bool(candidate["publishable"])
        # Release-negative deliberately includes uncertain: a claim a
        # reviewer cannot confirm from its own quote is not one the
        # system should be publishing either.
        positive = human == "supported"
        if published and positive:
            matrix["tp"] += 1
        elif published and not positive:
            matrix["fp"] += 1
            (unsupported_published if human == "unsupported" else uncertain_published).append(
                {"case_id": case_id, "claim": candidate["claim"], "human": human}
            )
        elif positive:
            matrix["fn"] += 1
        else:
            matrix["tn"] += 1
        rows.append(
            {
                "case_id": case_id,
                "run": run_id,
                "claim": candidate["claim"],
                "evidence_id": candidate.get("best_evidence_id"),
                "human_label": human,
                "published": published,
                "diagnostic_verdict": candidate.get("diagnostic_verdict"),
                "best_entailment": candidate.get("best_entailment"),
                "support_threshold": candidate.get("support_threshold"),
                "failed_guards": sorted(
                    {
                        name
                        for e in candidate.get("nli_scores", [])
                        for name in e.get("failed_guards", [])
                    }
                ),
                "withhold_reason": candidate.get("withhold_reason"),
            }
        )

    precision = matrix["tp"] / (matrix["tp"] + matrix["fp"]) if matrix["tp"] + matrix["fp"] else 1.0
    recall = matrix["tp"] / (matrix["tp"] + matrix["fn"]) if matrix["tp"] + matrix["fn"] else 0.0
    JOINED.write_text(
        json.dumps(
            {
                "description": (
                    "Fresh blinded release audit. Labels were assigned without sight of the "
                    "automated verdict, score, guard result or publication decision, then "
                    "joined by case_id. The reviewer is not independent of the system, so "
                    "this measures release-audit agreement, not model accuracy."
                ),
                "shuffle_seed": packet["shuffle_seed"],
                "confusion": matrix,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "unsupported_published": unsupported_published,
                "uncertain_published": uncertain_published,
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"confusion: {matrix}  precision={precision:.2f} recall={recall:.2f}")
    print(f"UNSUPPORTED PUBLISHED = {len(unsupported_published)}")
    print(f"UNCERTAIN PUBLISHED   = {len(uncertain_published)}")
    print(f"wrote {JOINED.name}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": build, "join": join}[command]()
