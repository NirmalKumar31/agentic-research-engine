"""Label the blind calibration set, one case at a time.

    python examples/verifier-calibration/label.py

Shows the claim and its complete evidence, and nothing else. The
verifier's verdict is not in this file and is not displayed: a reviewer
who sees the model's answer first is anchored by it, and the resulting
labels measure agreement rather than correctness.

Progress is written after every answer, so you can stop with Ctrl-C and
resume where you left off.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

BLIND = Path(__file__).parent / "blind_cases.json"

CHOICES = {
    "s": "supported",
    "p": "partially_supported",
    "u": "unsupported",
}


def wrap(text: str, indent: str = "    ", width: int = 88) -> str:
    return textwrap.fill(
        " ".join(str(text).split()), width, initial_indent=indent, subsequent_indent=indent
    )


def show_definitions(doc: dict) -> None:
    print("\nLabel definitions\n" + "-" * 88)
    for label, meaning in doc["label_definitions"].items():
        print(f"\n  {label.upper()}")
        print(wrap(meaning, indent="    "))
    print("\n" + "-" * 88)
    print(wrap(doc["instructions"], indent="  "))
    print("-" * 88)


def show_case(case: dict, index: int, total: int) -> None:
    print("\n" + "=" * 88)
    print(f"Case {index} of {total}   [{case['case_id']}]")
    print("=" * 88)
    print("\nCLAIM")
    print(wrap(case["claim"]))
    print(f"\nEVIDENCE ({len(case['evidence'])} item(s))")
    for item in case["evidence"]:
        page = f", p. {item['page']}" if item.get("page") else ""
        print(
            f"\n  [{item['evidence_id']}] {item.get('domain', '?')} "
            f"· {item.get('site_category', '?')}{page}"
        )
        title = item.get("source_title")
        if title:
            print(wrap(title, indent="      "))
        print(wrap(f'"{item["quote"]}"', indent="      "))


def prompt(case: dict) -> bool | None:
    """Label one case.

    True to advance, False to quit, None to re-show the definitions.
    """
    while True:
        answer = (
            input(
                "\n  [s]upported  [p]artial  [u]nsupported  "
                "[r]ationale  [k]skip  [d]efinitions  [q]uit > "
            )
            .strip()
            .lower()
        )
        if answer == "q":
            return False
        if answer == "k":
            return True
        if answer == "d":
            return None  # caller re-shows definitions
        if answer == "r":
            case["human_rationale"] = input("  rationale: ").strip() or None
            continue
        if answer in CHOICES:
            case["human_label"] = CHOICES[answer]
            if not case.get("human_rationale"):
                note = input("  rationale (optional, Enter to skip): ").strip()
                case["human_rationale"] = note or None
            return True
        print("  Unrecognised. Use s, p, u, r, k, d or q.")


def main() -> int:
    if not BLIND.is_file():
        print(f"No blind case file at {BLIND}", file=sys.stderr)
        return 1

    doc = json.loads(BLIND.read_text(encoding="utf-8"))
    cases = doc["cases"]
    show_definitions(doc)

    done = sum(1 for c in cases if c["human_label"])
    print(f"\n{done} of {len(cases)} already labelled.\n")

    for index, case in enumerate(cases, 1):
        if case["human_label"]:
            continue
        show_case(case, index, len(cases))
        result = prompt(case)
        while result is None:
            show_definitions(doc)
            show_case(case, index, len(cases))
            result = prompt(case)
        # Written after every answer so Ctrl-C never loses work.
        BLIND.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        if result is False:
            break

    labelled = sum(1 for c in cases if c["human_label"])
    print(f"\n{labelled} of {len(cases)} labelled. Saved to {BLIND.name}.")
    if labelled == len(cases):
        print("\nAll done. Score against the verifier with:")
        print("    python examples/verifier-calibration/score.py")
    else:
        print("Re-run this command to continue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
