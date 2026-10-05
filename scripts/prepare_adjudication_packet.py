"""Build the blinded adjudication packet for Phase B's 6 unresolved
reviewer disagreements (2 dimension-score, 4 harmful-claim-flag).

This script builds the packet and the private key, then stops. It does
not adjudicate anything and must not be extended to -- the whole point
of this step is a human decision, not a third AI opinion dressed up as
one. `scripts/apply_adjudication.py` is the separate script that runs
once a human has actually filled in the decisions.

Excluded from the packet, deliberately: arm, model, provider,
repetition, either original reviewer's identity or scores/rationale,
latency, cost, and any filename that could reveal identity. Included:
the question, its pre-registered rubric and forbidden_overclaims, the
disputed field, and the candidate's full blinded report (already
stripped of model/provider identifiers by `blind()` when it was built
for the original review). IDs here are freshly randomized and distinct
from both original reviewers' IDs, so the packet can't be cross-
referenced against either reviewer's original materials.

Run once. Produces, under `evaluations/phase_b/adjudication/`:

- `packet.md` -- the 6 disputed candidates, in randomized order, under
  fresh random IDs.
- `adjudication_template.csv` -- blank, one row per case.
- `adjudication_key.json` -- NOT for the adjudicator. Maps each random
  ID back to (question_id, repetition, arm, dispute_type,
  disputed_field). Kept out of the handoff copy entirely, the same
  discipline as the original review's unblinding keys.
"""

from __future__ import annotations

import csv
import json
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_research.evaluation.blinding import _KNOWN_IDENTIFIERS  # noqa: E402

PHASE_B = ROOT / "evaluations" / "phase_b"
ADJ = PHASE_B / "adjudication"

_LEAK_TERMS = (*_KNOWN_IDENTIFIERS, "[local]", "[cloud]", "redactions:")


class LeakError(Exception):
    pass


def scan_for_leaks(text: str, *, context: str) -> None:
    lowered = text.lower()
    hits = [t for t in _LEAK_TERMS if t in lowered]
    if hits:
        raise LeakError(f"{context}: found {hits} -- refusing to write a non-blind packet")


DISPUTES: list[tuple[str, int, str, str, str]] = [
    # (question_id, repetition, arm, dispute_type, disputed_field)
    ("Q3-procedural", 1, "local", "score_dimension", "claim_support_1to5"),
    ("Q3-procedural", 2, "local", "score_dimension", "claim_support_1to5"),
    ("Q10-long-tail", 1, "cloud", "harmful_claim_flag", "harmful_or_unsupported_claims_yn"),
    ("Q10-long-tail", 2, "cloud", "harmful_claim_flag", "harmful_or_unsupported_claims_yn"),
    ("Q10-long-tail", 2, "local", "harmful_claim_flag", "harmful_or_unsupported_claims_yn"),
    (
        "Q11-adversarial-evidence-shape",
        1,
        "local",
        "harmful_claim_flag",
        "harmful_or_unsupported_claims_yn",
    ),
]

_PROMPTS = {
    "score_dimension": (
        "Score **{field}** for this candidate yourself, 1-5, independent of any "
        "prior score. Do not try to reconstruct or split the difference between "
        "unseen prior scores -- none are shown to you on purpose."
    ),
    "harmful_claim_flag": (
        "Decide, independently: does this output contain a claim from the "
        "question's own forbidden_overclaims list, or another unsupported/"
        "harmful claim not on that list? Answer Yes or No."
    ),
}


def load_questions() -> dict[str, dict]:
    data = json.loads((ROOT / "examples" / "benchmark" / "questions.json").read_text())
    return {q["id"]: q for q in data["questions"]}


def read_blinded_markdown(question_id: str, rep: int, arm: str) -> str:
    path = PHASE_B / "blinded" / f"{question_id}_rep{rep}_{arm}.md"
    text = path.read_text(encoding="utf-8")
    if text.startswith("<!--"):
        text = text.split("-->", 1)[1].lstrip("\n")
    scan_for_leaks(text, context=f"{question_id} rep{rep} {arm}")
    return text


def build() -> None:
    ADJ.mkdir(parents=True, exist_ok=True)
    questions = load_questions()

    order = list(DISPUTES)
    secrets.SystemRandom().shuffle(order)

    key: dict[str, dict] = {}
    template_rows: list[dict] = []
    lines: list[str] = [
        "# Phase B -- blinded adjudication packet (6 disputed cases)\n",
        "Each case below is one of the 6 unresolved reviewer disagreements from "
        "the Phase B blinded review. No arm, model, provider, repetition, or "
        "either original reviewer's score/rationale is shown -- decide each "
        "independently, from the question and the candidate's own report.\n",
        "Do not compare cases to guess a pattern across them; each is an independent decision.\n",
        "\n---\n",
    ]

    for qid, rep, arm, dispute_type, field in order:
        question = questions[qid]
        random_id = secrets.token_hex(4)
        while random_id in key:  # pragma: no cover
            random_id = secrets.token_hex(4)
        key[random_id] = {
            "question_id": qid,
            "repetition": rep,
            "arm": arm,
            "dispute_type": dispute_type,
            "disputed_field": field,
        }

        content = read_blinded_markdown(qid, rep, arm)

        lines.append(f"## Case `{random_id}`\n")
        lines.append(f"**Question:** {question['text']}\n")
        lines.append(f"**Rubric:** {question['rubric']}\n")
        lines.append("**Forbidden overclaims:**\n")
        for item in question["forbidden_overclaims"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append(f"**What is disputed:** {_PROMPTS[dispute_type].format(field=field)}\n")
        lines.append("### Candidate report\n")
        lines.append(content)
        lines.append("\n---\n")

        template_rows.append(
            {
                "blinded_id": random_id,
                "dispute_type": dispute_type,
                "disputed_field": field,
                "final_decision": "",
                "rationale": "",
                "confidence": "",
                "abstain_insufficient_evidence": "",
            }
        )

    full_text = "\n".join(lines)
    scan_for_leaks(full_text, context="packet.md (assembled)")
    (ADJ / "packet.md").write_text(full_text, encoding="utf-8")

    with (ADJ / "adjudication_template.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(template_rows[0].keys()))
        writer.writeheader()
        writer.writerows(template_rows)

    (ADJ / "adjudication_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")

    readme = """# Phase B adjudication packet -- instructions

6 disputed cases from the blinded review, for one independent human
adjudicator. Not a third AI opinion -- if no independent human is
available, these 6 stay unresolved; that is the correct, honest state,
not a defect to paper over.

## Workflow

1. Open `packet.md`. Read each `## Case` in order.
2. For each case, fill one row in `adjudication_template.csv` (matched
   by `blinded_id`):
   - `final_decision`: for a `score_dimension` case, an integer 1-5.
     For a `harmful_claim_flag` case, exactly `Yes` or `No`.
   - `rationale`: your reasoning, required.
   - `confidence`: `High`, `Medium`, or `Low`.
   - `abstain_insufficient_evidence`: `Yes` if you cannot decide from
     what is shown, otherwise `No`. An abstain is a valid, honest
     answer -- it is not scored as a failure.
3. All 6 rows, all fields, before returning the sheet.
4. Do not open `adjudication_key.json` -- it is not included in your
   copy of this packet at all.
5. Save `adjudication_template.csv` in place and return it.
"""
    (ADJ / "README.md").write_text(readme, encoding="utf-8")

    print(f"Packet built: {len(template_rows)} disputed cases.")
    print(f"  {ADJ / 'packet.md'}")
    print(f"  {ADJ / 'adjudication_template.csv'}")
    print(f"  {ADJ / 'README.md'}")
    print(f"  {ADJ / 'adjudication_key.json'}  <-- do NOT hand to the adjudicator")


if __name__ == "__main__":
    build()
