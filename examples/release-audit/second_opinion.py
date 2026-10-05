"""A second, independent adjudicator for exactly one disputed case.

The independent reviewer joined in `reviewer_packet.py` found one
published claim it labelled unsupported: `nist-ai-risk-framework-5`
(see issue #32, maintainer-internal; never shown to the adjudicator).
One reviewer's disagreement on one case is a finding worth
investigating, not evidence by itself -- it needs a second, independent
opinion before anyone decides what it means.

This gives a second person only the claim and the single quote the
system selected for it, under a neutral random identifier -- not the
repo's own case naming, which would itself hint that this came from a
specific run in a specific audit. It carries no label from the first
reviewer, no NLI score, no guard outcome, no publication result, no
mention of the first reviewer's verdict or of issue #32, and no model
or provider identity.

A prior version of this script's generated packet leaked exactly that:
its hand-written `description` field stated outright that "a first
independent reviewer labelled this claim unsupported; the system
published it" -- narrating the blinding instead of enforcing it. Caught
before any human received it. `FORBIDDEN_TERMS` and the build-time
assertion below exist specifically so a human-written sentence like
that fails the same way a leaked field name would, rather than only
being caught by a human re-reading the generated JSON by hand.

    python examples/release-audit/second_opinion.py build
    python examples/release-audit/second_opinion.py join

`build` writes evidence-review-packet.json alongside an empty
evidence-review-label.json for the adjudicator to fill. `join` reads the
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
PACKET = HERE / "evidence-review-packet.json"
LABEL = HERE / "evidence-review-label.json"
RESULT = HERE / "evidence-review-result.json"

# The real, internal case identifier -- used only to look the candidate
# up in candidate-audit.json and the first reviewer's label up in
# reviewer-labels.json. Never written to the outward-facing packet or
# label file: `nist-ai-risk-framework-5` would itself tell an adjudicator
# which run and which audit this came from.
CASE_ID = "nist-ai-risk-framework-5"

# The identifier the adjudicator actually sees. Fixed, not regenerated
# per run, so a label returned against one `build` still joins correctly
# even if `build` is re-run before the label comes back.
EXTERNAL_CASE_ID = "EXR-45d21543"

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

# Checked against the packet's description, instructions and case data --
# deliberately not against label_definitions, which legitimately contains
# "unsupported" and "supported" as instructions to the adjudicator, not
# as a decision about this case. A term appearing here means the packet
# is narrating something about the first reviewer, the system, or the
# verification machinery instead of presenting a bare claim-and-quote.
FORBIDDEN_TERMS = (
    "first reviewer",
    "independent reviewer",
    "unsupported",
    "published",
    "publish",
    "issue #32",
    "guard",
    "nli",
    "score",
    "threshold",
    "openai",
    "qwen",
    "gpt",
    "deberta",
    CASE_ID,
    # The run name alone (without the "-5" index) would still tell an
    # adjudicator which of the three canonical recordings this came
    # from -- a narrower leak than the full case_id, but still one.
    CASE_ID.rsplit("-", 1)[0],
)


def _find_candidate() -> dict[str, Any]:
    audit = json.loads(AUDIT.read_text())
    run_id, index_str = CASE_ID.rsplit("-", 1)
    candidate: dict[str, Any] = audit["runs"][run_id]["candidates"][int(index_str)]
    return candidate


def _assert_not_leaking(packet: dict[str, Any]) -> None:
    # Only `description` and `cases` are checked. `instructions` and
    # `label_definitions` both legitimately name all three label words
    # ("choose supported / unsupported / uncertain") as a generic menu of
    # options -- that is not a leak about this case's verdict, and
    # scanning those fields would make every valid packet fail.
    checked = json.dumps([packet["description"], packet["cases"]]).lower()
    found = [term for term in FORBIDDEN_TERMS if term.lower() in checked]
    assert not found, f"the packet leaks: {found}"


def build() -> None:
    candidate = _find_candidate()
    evidence = candidate["evidence"]
    best = candidate.get("best_evidence_id")
    chosen = next((e for e in evidence if e["evidence_id"] == best), evidence[0])

    packet = {
        "description": (
            "Please review one claim-and-evidence pair. Judge only whether the "
            "supplied quote supports the claim."
        ),
        "instructions": (
            "Do not use outside knowledge, search the web, or infer what anyone else "
            "decided. Label the case below supported / unsupported / uncertain using "
            "the definitions provided, then write your label into "
            "evidence-review-label.json as {case_id: label}, replacing the single "
            "null value, and return the file unchanged otherwise."
        ),
        "label_definitions": DEFINITIONS,
        "cases": [
            {
                "case_id": EXTERNAL_CASE_ID,
                "claim": candidate["claim"],
                "quote": chosen.get("quote"),
                "source_title": chosen.get("source_title"),
                "source_domain": chosen.get("source_domain"),
                "source_id": chosen.get("source_id"),
                "page": chosen.get("page"),
            }
        ],
    }
    _assert_not_leaking(packet)

    PACKET.write_text(json.dumps(packet, indent=2) + "\n")
    if not LABEL.exists():
        LABEL.write_text(json.dumps({EXTERNAL_CASE_ID: None}, indent=2) + "\n")

    print(f"wrote {PACKET.name}: 1 case ({EXTERNAL_CASE_ID}), no automated decision included")
    print(f"wrote {LABEL.name}: awaiting a second adjudicator")


class ValidationError(Exception):
    """Raised when the returned second-opinion label cannot be joined safely."""


def validate_label(label: dict[str, object]) -> None:
    if set(label) != {EXTERNAL_CASE_ID}:
        raise ValidationError(
            f"evidence-review-label.json must contain exactly one key, "
            f"{EXTERNAL_CASE_ID!r}; got {sorted(label)}"
        )
    if label[EXTERNAL_CASE_ID] not in VALID_LABELS:
        raise ValidationError(
            f"label must be one of {sorted(VALID_LABELS)}; got invalid or unfilled value: "
            f"{label[EXTERNAL_CASE_ID]!r}"
        )


def join() -> None:
    label = json.loads(LABEL.read_text())
    validate_label(label)
    second = label[EXTERNAL_CASE_ID]

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
