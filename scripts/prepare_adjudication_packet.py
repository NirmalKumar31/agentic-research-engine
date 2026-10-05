"""Build the blinded adjudication packet for Phase B's 6 unresolved
reviewer disagreements (2 dimension-score, 4 harmful-claim-flag).

This script builds the packet and stops. It does not adjudicate
anything and must not be extended to -- the whole point of this step
is an independent human decision, not a third AI opinion dressed up as
one. There is deliberately no companion script that reads back
completed decisions yet: that only gets written once a human has
actually returned a filled-in sheet, against the real one, not a
guess at its shape in advance.

Excluded from the packet, deliberately: arm, model, provider,
repetition, either original reviewer's identity or scores/rationale,
latency, cost, and any filename that could reveal identity. Included:
the question, its pre-registered rubric and forbidden_overclaims, the
disputed field, and the candidate's full blinded report (already
stripped of model/provider identifiers by `blind()` when it was built
for the original review). IDs here are freshly randomized and distinct
from both original reviewers' IDs, so the packet can't be cross-
referenced against either reviewer's original materials.

Blinding here is procedural, not cryptographic: the same candidate
text is also reachable elsewhere in this public repository under
arm-labelled filenames (`evaluations/phase_b/blinded/`,
`evaluations/phase_b/review/reviewer_{a,b}/packet/`). An adjudicator
who searches the repository, or opens either original reviewer's
directory, can recover the arm that way -- the packet's own redaction
does not prevent that. The instructions this script writes tell the
adjudicator not to do that; it is a discipline, the same as the
original reviewers not opening the unblinding key, not a guarantee
enforced by the file format.

Builds through a temporary staging directory and only moves the
result into place once everything -- including the leak scan -- has
succeeded, and refuses outright if the destination already exists and
is non-empty (a packet, a key, or a possibly-in-progress or completed
score sheet). Run again over an existing result by moving or deleting
`evaluations/phase_b/adjudication/` first, deliberately, not by
silently overwriting it.

Produces, under `evaluations/phase_b/adjudication/`:

- `packet.md`, `adjudication_template.csv`, `README.md` -- also
  duplicated, unchanged, into `handoff/` (exactly these three files,
  automatically -- not a manually-assembled copy).
- `adjudication_key.json`, mode `0600`, kept out of `handoff/`
  entirely. Maps each random ID back to (question_id, repetition, arm,
  dispute_type, disputed_field). Not for the adjudicator.
"""

from __future__ import annotations

import csv
import json
import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_research.evaluation.blinding import _KNOWN_IDENTIFIERS  # noqa: E402

PHASE_B = ROOT / "evaluations" / "phase_b"
ADJ = PHASE_B / "adjudication"

_LEAK_TERMS = (*_KNOWN_IDENTIFIERS, "[local]", "[cloud]", "redactions:")

HANDOFF_FILES = ("packet.md", "adjudication_template.csv", "README.md")


class LeakError(Exception):
    pass


class DestinationExistsError(Exception):
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

README_TEMPLATE = """# Phase B adjudication packet -- instructions

6 disputed cases from the blinded review, for one independent human
adjudicator. Not a third AI opinion -- if no independent human is
available, these 6 stay unresolved; that is the correct, honest state,
not a defect to paper over.

## Before you start: procedural blinding, not cryptographic

This packet's text is also reachable elsewhere in this public
repository under filenames that do name the arm (model/provider) --
`evaluations/phase_b/blinded/`, and each original reviewer's own
`packet/` directory. Redacting identifiers from the text in front of
you does not stop a search of the repository from recovering them.

**Do not, before you have filled in all 6 decisions:**

- browse or search this repository for the exact candidate passages
  shown below,
- open `evaluations/phase_b/blinded/`, either reviewer's directory
  under `evaluations/phase_b/review/`, or any other benchmark artifact,
- open `adjudication_key.json` -- it is not included in your copy of
  this packet at all.

Decide from what is on the page in front of you, nothing else.

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
4. Save `adjudication_template.csv` in place and return it.
"""


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


def _build_into(staging: Path) -> int:
    """Writes the full packet (plus its handoff copy) under `staging`.
    Returns the number of disputed cases written. Raises before
    anything is written to the real destination if a leak is found --
    `build()` only moves `staging`'s contents into place after this
    returns successfully."""
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

    (staging / "packet.md").write_text(full_text, encoding="utf-8")

    with (staging / "adjudication_template.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(template_rows[0].keys()))
        writer.writeheader()
        writer.writerows(template_rows)

    (staging / "README.md").write_text(README_TEMPLATE, encoding="utf-8")

    key_path = staging / "adjudication_key.json"
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    os.chmod(key_path, 0o600)

    handoff = staging / "handoff"
    handoff.mkdir()
    for name in HANDOFF_FILES:
        shutil.copyfile(staging / name, handoff / name)

    return len(template_rows)


def build() -> None:
    if ADJ.exists() and any(ADJ.iterdir()):
        raise DestinationExistsError(
            f"{ADJ} already exists and is not empty -- refusing to overwrite a packet, "
            "key, or possibly-completed score sheet. If you really mean to rebuild from "
            "scratch, move or delete it first, deliberately."
        )

    with tempfile.TemporaryDirectory(prefix="phase_b_adjudication_") as tmp:
        staging = Path(tmp)
        count = _build_into(staging)

        ADJ.mkdir(parents=True, exist_ok=True)
        for item in staging.iterdir():
            shutil.move(str(item), str(ADJ / item.name))

    print(f"Packet built: {count} disputed cases.")
    print(f"  {ADJ / 'handoff'}/  <-- give this whole folder to the adjudicator")
    print(f"  {ADJ / 'adjudication_key.json'}  <-- do NOT hand to the adjudicator (mode 0600)")


if __name__ == "__main__":
    build()
