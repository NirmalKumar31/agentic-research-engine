"""A second, independent adjudicator for exactly one disputed case.

The independent reviewer joined in `reviewer_packet.py` found one
published claim it labelled unsupported: `nist-ai-risk-framework-5`
(see issue #32). One reviewer's disagreement on one case is a finding
worth investigating, not evidence by itself -- it needs a second,
independent opinion before anyone decides what it means.

This gives a second person only the claim and the single quote the
system selected for it. It carries no label from the first reviewer, no
NLI score, no guard outcome and no publication result -- the same
blinding discipline as `reviewer_packet.py`, scoped to one case because
only one case is in dispute.

    python examples/release-audit/second_opinion.py build
    python examples/release-audit/second_opinion.py join

`build` writes second-opinion-packet.json alongside an empty
second-opinion-label.json for the adjudicator to fill. `join` reads the
returned label, validates it, and records it next to the first
reviewer's original label -- agreement or disagreement, never erasing
either one. It does not decide anything: per issue #32, what happens
next (open an investigation into the claim-generation path, or record
an unresolved disagreement) is a human call made after reading the
comparison, not an automatic action this script takes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
AUDIT = HERE / "candidate-audit.json"
FIRST_LABELS = HERE / "reviewer-labels.json"
PACKET = HERE / "second-opinion-packet.json"
LABEL = HERE / "second-opinion-label.json"
RESULT = HERE / "second-opinion-result.json"

# The one case in dispute (issue #32). Deliberately not a list: widening
# scope to "whichever cases look disputed" is a different, larger task
# than adjudicating this one finding.
CASE_ID = "nist-ai-risk-framework-5"

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


def _find_candidate() -> dict[str, Any]:
    audit = json.loads(AUDIT.read_text())
    run_id, index_str = CASE_ID.rsplit("-", 1)
    candidate: dict[str, Any] = audit["runs"][run_id]["candidates"][int(index_str)]
    return candidate


def build() -> None:
    candidate = _find_candidate()
    evidence = candidate["evidence"]
    best = candidate.get("best_evidence_id")
    chosen = next((e for e in evidence if e["evidence_id"] == best), evidence[0])

    PACKET.write_text(
        json.dumps(
            {
                "description": (
                    "Second-opinion adjudication for one disputed case from the "
                    "Agentic Research Engine release audit (issue #32). A first "
                    "independent reviewer labelled this claim unsupported; the system "
                    "published it. Judge whether the quote supports the claim -- "
                    "nothing about the first reviewer's label, the system's score, its "
                    "guard results or its publication decision is included here."
                ),
                "instructions": (
                    "Label the one case below supported / unsupported / uncertain using "
                    "the definitions provided. Judge the claim against its quote only -- "
                    "not against what you know to be true about the subject. Write your "
                    "label into second-opinion-label.json as {case_id: label}."
                ),
                "label_definitions": DEFINITIONS,
                "cases": [
                    {
                        "case_id": CASE_ID,
                        "claim": candidate["claim"],
                        "quote": chosen.get("quote"),
                        "source_title": chosen.get("source_title"),
                        "source_domain": chosen.get("source_domain"),
                        "source_id": chosen.get("source_id"),
                        "page": chosen.get("page"),
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )
    if not LABEL.exists():
        LABEL.write_text(json.dumps({CASE_ID: None}, indent=2) + "\n")

    # Checked against the case data only, not the whole packet: the label
    # words themselves ("unsupported" etc.) legitimately appear in
    # label_definitions, which is instructions, not a decision about this
    # case.
    leaked = {
        "publishable",
        "present_in_published_report",
        "best_entailment",
        "nli_scores",
        "diagnostic_verdict",
        "human_review",
        "withhold_reason",
    }
    case_body = json.dumps(json.loads(PACKET.read_text())["cases"])
    leaked_found = [k for k in leaked if k in case_body]
    assert not leaked_found, f"the packet leaks an automated field: {leaked_found}"
    first_labels = json.loads(FIRST_LABELS.read_text())
    assert first_labels[CASE_ID] not in case_body, (
        "the first reviewer's label leaked into the packet"
    )
    print(f"wrote {PACKET.name}: 1 case ({CASE_ID}), no automated decision included")
    print(f"wrote {LABEL.name}: awaiting a second adjudicator")


class ValidationError(Exception):
    """Raised when the returned second-opinion label cannot be joined safely."""


def validate_label(label: dict[str, object]) -> None:
    if set(label) != {CASE_ID}:
        raise ValidationError(
            f"second-opinion-label.json must contain exactly one key, {CASE_ID!r}; "
            f"got {sorted(label)}"
        )
    if label[CASE_ID] not in VALID_LABELS:
        raise ValidationError(
            f"label must be one of {sorted(VALID_LABELS)}; got invalid or unfilled value: "
            f"{label[CASE_ID]!r}"
        )


def join() -> None:
    label = json.loads(LABEL.read_text())
    validate_label(label)
    second = label[CASE_ID]

    first_labels = json.loads(FIRST_LABELS.read_text())
    first = first_labels[CASE_ID]

    # The branch is on the second adjudicator's own verdict, not on exact
    # string equality with the first label: the first reviewer's label
    # here is "unsupported", and an "uncertain" second label is a
    # different string that still corroborates "not clearly supported" --
    # instruction #4 explicitly groups unsupported and uncertain together.
    if second == "supported":
        next_step = (
            "The second adjudicator supports the claim: record this as an unresolved "
            "reviewer disagreement. Do not erase the first reviewer's label or claim "
            "the case is resolved."
        )
    else:
        next_step = (
            "The second adjudicator's label is unsupported or uncertain, corroborating "
            "the first reviewer: inspect the claim-generation and verification path for "
            "this case, then propose a narrow regression test and fix in a separate PR."
        )

    RESULT.write_text(
        json.dumps(
            {
                "description": (
                    "Second-opinion adjudication result for one disputed release-audit "
                    "case (issue #32). Both labels are recorded as given; neither is "
                    "erased, averaged or treated as authoritative over the other."
                ),
                "case_id": CASE_ID,
                "first_reviewer_label": first,
                "second_adjudicator_label": second,
                "exact_label_agreement": first == second,
                "next_step": next_step,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"first={first!r} second={second!r}")
    print(f"wrote {RESULT.name}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": build, "join": join}[command]()
