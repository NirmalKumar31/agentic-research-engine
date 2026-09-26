"""Build a packet for a second, genuinely independent reviewer.

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

    python examples/release-audit/reviewer_packet.py

Writes reviewer-packet.json alongside an empty reviewer-labels.json for
the reviewer to fill. Join afterwards with blind_packet.py-style
joining on case_id.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
AUDIT = HERE / "candidate-audit.json"
PACKET = HERE / "reviewer-packet.json"
LABELS = HERE / "reviewer-labels.json"

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


def main() -> None:
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
                    "quote": chosen.get("quote"),
                    "source_title": chosen.get("source_title"),
                    "source_domain": chosen.get("source_domain"),
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


if __name__ == "__main__":
    main()
