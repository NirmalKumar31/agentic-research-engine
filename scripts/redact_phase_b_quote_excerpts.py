"""Strip verbatim source quotes from the 3 "Source excerpts" fallback
outputs, in place, in both `runs/*.json` and `blinded/*.md`.

`report.py`'s "## Source excerpts" section (rendered when no synthesized
claim passed verification) prints `item.quote` -- a verbatim, often
multi-sentence excerpt from a third-party web page -- directly into the
report. Three of this benchmark's 48 runs hit that path (Q2, Q6, Q10,
all rep1 local), and all three ended up with substantial verbatim
third-party text committed in both the raw run record and the blinded
reviewer copy. Same redistribution-rights concern as the frozen
corpora, smaller in scope.

Only the quote text inside `- "<quote>" — **[Sx]** title` lines is
replaced; the attribution, every other section (Summary, Limitations,
Sources, Citation verification), and every measurement (`ok`,
`timed_out`, `duration_s`, cost, tokens, metrics) are untouched. A
backup of the original unredacted files is kept locally (not committed)
at `evaluations/phase_b/_local_only_unredacted_backup/` before this
runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE_B = ROOT / "evaluations" / "phase_b"

AFFECTED = ["Q2-numeric-lookup_rep1_local", "Q6-causal_rep1_local", "Q10-long-tail_rep1_local"]

# `- "<quote>" — **[S1]** Some Title` -- capture the quote (group 1) and
# the attribution that follows it (group 2), replace only the quote.
_QUOTE_LINE = re.compile(r'^- "(.*)" (— \*\*\[.+)$', re.MULTILINE)


def redact_markdown(text: str) -> tuple[str, int]:
    count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        quote, attribution = match.group(1), match.group(2)
        placeholder = f"[REDACTED: {len(quote)} chars of verbatim source quote, not redistributed]"
        return f'- "{placeholder}" {attribution}'

    redacted = _QUOTE_LINE.sub(_replace, text)
    return redacted, count


def main() -> None:
    for name in AFFECTED:
        run_path = PHASE_B / "runs" / f"{name}.json"
        data = json.loads(run_path.read_text())
        redacted_markdown, n = redact_markdown(data["markdown"])
        if n == 0:
            raise SystemExit(
                f"{run_path}: expected quote lines to redact, found none -- check the pattern"
            )
        data["markdown"] = redacted_markdown
        run_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        blinded_path = PHASE_B / "blinded" / f"{name}.md"
        blinded_text = blinded_path.read_text()
        redacted_blinded, n2 = redact_markdown(blinded_text)
        if n2 == 0:
            raise SystemExit(f"{blinded_path}: expected quote lines to redact, found none")
        blinded_path.write_text(redacted_blinded, encoding="utf-8")

        print(f"{name}: redacted {n} quote(s) in runs/*.json, {n2} in blinded/*.md")


if __name__ == "__main__":
    main()
