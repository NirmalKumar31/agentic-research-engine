"""Build a packet for a second, genuinely independent reviewer, and
join the returned labels back afterwards.

The blinded audit in this directory hides the automated decision, which
removes one bias. It does not remove the reviewer: the same person
built the system and assigned the labels, so the result is a blinded
self-review and is called that everywhere it appears.

This produces a packet for somebody else. It carries only what is
needed to judge a claim against its evidence, and deliberately omits
the automated verdict, the score, the guard outcomes, the publication
decision, the previous human label and any explanation of a previous
audit -- all of which would anchor a fresh reviewer just as effectively
as the verdict did.

    python examples/release-audit/reviewer_packet.py build
    python examples/release-audit/reviewer_packet.py join

`build` writes reviewer-packet.json alongside an empty
reviewer-labels.json for the reviewer to fill. `join` reads the filled
labels back, validates them, and joins them to the hidden publication
outcome by `case_id` -- mirroring `blind_packet.py`'s join, with one
difference: this reviewer is independent of the system, so the result
is release validation, not a self-review, and the join enforces that
every case was actually labelled before it touches any hidden field.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
AUDIT = HERE / "candidate-audit.json"
PACKET = HERE / "reviewer-packet.json"
LABELS = HERE / "reviewer-labels.json"
JOINED = HERE / "reviewer-audit.json"
RECORDED_RUNS = HERE.parents[1] / "src" / "agentic_research" / "web" / "recorded_runs"

VALID_LABELS = {"supported", "unsupported", "uncertain"}

DEFINITIONS = {
    "supported": (
        "This quote, on its own, states or directly implies the claim with its "
        "scope, modality, quantities and attribution intact."
    ),
    "unsupported": (
        "It does not. The claim says more than the quote, or something the quote does not say."
    ),
    "uncertain": "You cannot tell from this quote alone.",
}


def _source_metadata(run_id: str) -> dict[str, dict[str, Any]]:
    """`source_id` -> {"title", "domain"} for one recorded run.

    candidate-audit.json's evidence records never carried this metadata
    -- `chosen.get("source_title")` always returned `None`, silently,
    for every case in both packets this script has ever built. The real
    values live in the recorded run's own `result.sources[]`, keyed by
    the same `source_id` evidence already references.
    """
    recording = json.loads((RECORDED_RUNS / f"{run_id}.json").read_text())
    result = recording.get("result", recording)
    return {
        s["id"]: {"title": s.get("title"), "domain": s.get("domain")} for s in result["sources"]
    }


def build() -> None:
    audit = json.loads(AUDIT.read_text())
    cases = []
    for run_id, run in audit["runs"].items():
        source_meta = _source_metadata(run_id)
        for index, candidate in enumerate(run["candidates"]):
            best = candidate.get("best_evidence_id")
            chosen = next(
                (e for e in candidate["evidence"] if e["evidence_id"] == best),
                (candidate["evidence"] or [{}])[0],
            )
            meta = source_meta.get(chosen.get("source_id"), {})
            cases.append(
                {
                    "case_id": f"{run_id}-{index}",
                    "claim": candidate["claim"],
                    "quote": chosen.get("quote"),
                    "source_title": meta.get("title"),
                    "source_domain": meta.get("domain"),
                    "source_id": chosen.get("source_id"),
                    "page": chosen.get("page"),
                }
            )

    PACKET.write_text(
        json.dumps(
            {
                "description": (
                    "Independent review packet for the Agentic Research Engine release audit. "
                    "Each case is one generated claim and the single evidence quote the system "
                    "selected as its strongest support. Judge whether that quote supports that "
                    "claim. Nothing about what the system decided is included."
                ),
                "instructions": (
                    "Label every case supported / unsupported / uncertain using the definitions "
                    "below. Judge the claim against its quote only -- not against what you know "
                    "to be true about the subject. Write your labels into "
                    "reviewer-labels.json as {case_id: label}."
                ),
                "label_definitions": DEFINITIONS,
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )
    if not LABELS.exists():
        LABELS.write_text(json.dumps({c["case_id"]: None for c in cases}, indent=2) + "\n")

    leaked = {"publishable", "best_entailment", "nli_scores", "human_review", "withhold_reason"}
    body = PACKET.read_text()
    assert not [k for k in leaked if k in body], "the packet leaks an automated decision"
    print(f"wrote {PACKET.name}: {len(cases)} cases, no automated decision included")
    print(f"wrote {LABELS.name}: awaiting a second reviewer")


class ValidationError(Exception):
    """Raised when the returned labels cannot be safely joined.

    Deliberately checked before the join touches `candidate-audit.json`'s
    hidden fields at all: a count mismatch or a stray label value is a
    data-integrity problem, not something to silently coerce or drop.
    """


def validate_labels(packet_case_ids: set[str], labels: dict[str, object]) -> None:
    if len(packet_case_ids) != 21:
        raise ValidationError(f"expected 21 cases in the packet, found {len(packet_case_ids)}")

    label_ids = set(labels)
    if label_ids != packet_case_ids:
        missing = sorted(packet_case_ids - label_ids)
        extra = sorted(label_ids - packet_case_ids)
        raise ValidationError(
            f"returned labels do not match the packet's case_ids exactly "
            f"-- missing {missing}, unexpected {extra}"
        )

    invalid = {cid: v for cid, v in labels.items() if v not in VALID_LABELS}
    if invalid:
        raise ValidationError(
            f"every label must be one of {sorted(VALID_LABELS)}; "
            f"got invalid or unfilled values: {invalid}"
        )


def join() -> None:
    audit = json.loads(AUDIT.read_text())
    packet = json.loads(PACKET.read_text())
    labels = json.loads(LABELS.read_text())

    # Validate before any hidden field (publication outcome, diagnostic
    # verdict, entailment score) is read at all.
    validate_labels({c["case_id"] for c in packet["cases"]}, labels)

    flat: dict[str, tuple[str, dict[str, object]]] = {}
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
        # `present_in_published_report` is the actual outcome a reader saw,
        # not `publishable` (the gate's own internal diagnostic flag) --
        # `gate_disagreements` in candidate-audit.json exists precisely to
        # catch the two ever differing, so the real outcome is the one
        # worth validating against.
        published = bool(candidate["present_in_published_report"])
        # Release-negative deliberately includes uncertain: a claim an
        # independent reviewer cannot confirm from its own quote is not
        # one the system should have published either.
        positive = human == "supported"
        if published and positive:
            matrix["tp"] += 1
        elif published and not positive:
            matrix["fp"] += 1
            (unsupported_published if human == "unsupported" else uncertain_published).append(
                {"case_id": case_id, "claim": str(candidate["claim"]), "human": str(human)}
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
                "withhold_reason": candidate.get("withhold_reason"),
            }
        )

    precision = matrix["tp"] / (matrix["tp"] + matrix["fp"]) if matrix["tp"] + matrix["fp"] else 1.0
    recall = matrix["tp"] / (matrix["tp"] + matrix["fn"]) if matrix["tp"] + matrix["fn"] else 0.0

    JOINED.write_text(
        json.dumps(
            {
                "description": (
                    "Independent release-validation join. reviewer-labels.json was produced "
                    "from reviewer-packet.json, which structurally cannot carry an automated "
                    "verdict, score, guard result, publication decision or prior label -- "
                    "joined here, by case_id, against the hidden candidate-audit.json outcome."
                ),
                "process_attestation": {
                    "reviewer_independence": (
                        "A second reviewer, independent of the person and process that built "
                        "the verifier -- unlike the self-review recorded in blind-audit.json."
                    ),
                    "blinding": (
                        "reviewer-packet.json cannot contain an automated verdict, score, "
                        "guard result, publication decision or prior label: reviewer_packet.py "
                        "asserts their absence before the packet is ever written. This describes "
                        "the packet only -- the automated outcome already existed, publicly, "
                        "elsewhere in this repository (candidate-audit.json, blind-audit.json) "
                        "by the time the packet was built. Blinding here is procedural, not "
                        "cryptographic: it depends on the reviewer following the packet's own "
                        "instruction not to look elsewhere, the same limitation disclosed for "
                        "Phase B's blinded review."
                    ),
                    "basis": (
                        "Each case is labelled supported / unsupported / uncertain against its "
                        "own shown quote alone, per the packet's own instructions -- not "
                        "against outside knowledge of the subject."
                    ),
                },
                "validation_note": (
                    "One external reviewer over this fixed 21-case set is independent release "
                    "validation, not a statistical benchmark: n=1 reviewer, no inter-rater "
                    "agreement and no significance claim is made or implied."
                ),
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
