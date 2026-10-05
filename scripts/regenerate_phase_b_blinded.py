"""Regenerate `evaluations/phase_b/blinded/*.md` from the already-committed
`evaluations/phase_b/runs/*.json` raw markdown.

Exists because the first blinding pass called `blind(arm.upper(), ...)`,
so every redacted identifier was replaced with the literal string
`[LOCAL]` or `[CLOUD]` -- a more direct arm-identity leak than the
original string. `runs/*.json`'s `markdown` field is the untouched
original output (blinding was only ever applied on the way to
`blinded/`), so this is a pure post-processing fix: no provider call,
no rerun, same 48 recorded outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_research.evaluation.blinding import blind  # noqa: E402

PHASE_B = ROOT / "evaluations" / "phase_b"


def main() -> None:
    runs = sorted((PHASE_B / "runs").glob("*.json"))
    if not runs:
        raise SystemExit(f"no run files found under {PHASE_B / 'runs'}")
    rewritten = 0
    for path in runs:
        data = json.loads(path.read_text())
        if not data["ok"]:
            continue  # a timed-out/failed run has no markdown to blind
        blinded = blind("REDACTED_MODEL", data["markdown"])
        out_path = PHASE_B / "blinded" / (path.stem + ".md")
        out_path.write_text(
            f"<!-- redactions: {blinded.redaction_count} -->\n\n{blinded.markdown}",
            encoding="utf-8",
        )
        rewritten += 1
    print(f"Rewrote {rewritten} blinded files from {len(runs)} run records.")


if __name__ == "__main__":
    main()
