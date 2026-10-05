"""Documentation gates: run in CI so a documentation regression fails
the build instead of waiting for the next scrutiny pass to find it.

Four checks, each independently actionable:

1. **Internal links resolve.** Every `[text](path)` markdown link whose
   target is a repo-relative path (not `http(s)://`, not a bare
   `#anchor`) must point at a file that exists.
2. **Required index files exist.** A handful of directories are large
   enough, or deliberately public enough, that a visitor needs an entry
   point rather than a raw file listing.
3. **README.md stays short.** A hard ceiling, not a style preference --
   this is specifically to catch the README re-accumulating release
   chronology the way it did before `docs/VALIDATION-HISTORY.md` existed
   to hold that content instead.
4. **No present-tense deployed-SHA claim outside history.** The exact
   shape of a real incident: `docs/history/vnext-baseline/paid-run-
   acceptance.md` correctly says, in the past tense and dated, which SHA
   was deployed on 2026-09-29 -- that is a historical record and stays
   exactly as written. A *new* document asserting "the deployed service
   runs `<sha>`" as a present-tense fact would go stale the moment the
   next deploy happens, silently, the same way the original incident did.
   This check does not flag commit-SHA citations in general (those are
   normal and frequent in this project's evidence-based docs) -- only
   this specific "currently deployed/running" phrasing, outside the
   directories whose whole point is to preserve a snapshot.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

README_MAX_LINES = 400

REQUIRED_INDEXES = [
    "docs/README.md",
    "docs/history/README.md",
    "evaluations/README.md",
    "evaluations/phase_b/README.md",
    "examples/README.md",
    "examples/live-validation/README.md",
]

# Directories where a dated, past-tense deployment snapshot is exactly
# the point -- exempt from check 4.
HISTORY_EXEMPT_PREFIXES = (
    "docs/history/",
    "examples/live-validation/",
    "CHANGELOG.md",
)

_CURRENT_STATE_SHA = re.compile(
    r"\bthe deployed (?:service|demo|instance) (?:is|runs|currently runs)\s+\*{0,2}`[0-9a-f]{7,40}`"
    r"|\bthis branch is\s+\*{0,2}`[0-9a-f]{7,40}`",
    re.IGNORECASE,
)


def _iter_markdown() -> list[Path]:
    return [
        p
        for p in ROOT.rglob("*.md")
        if ".venv" not in p.parts and "node_modules" not in p.parts and ".git" not in p.parts
    ]


def check_links() -> list[str]:
    problems = []
    for md in _iter_markdown():
        text = md.read_text(errors="ignore")
        for m in re.finditer(r"\]\(([^)]+)\)", text):
            link = m.group(1)
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            path_part = link.split("#")[0]
            if not path_part:
                continue
            target = (md.parent / path_part).resolve()
            if not target.exists():
                problems.append(f"{md.relative_to(ROOT)}: broken link -> {link}")
    return problems


def check_required_indexes() -> list[str]:
    return [
        f"missing required index: {rel}" for rel in REQUIRED_INDEXES if not (ROOT / rel).is_file()
    ]


def check_readme_length() -> list[str]:
    readme = ROOT / "README.md"
    n = len(readme.read_text().splitlines())
    if n > README_MAX_LINES:
        return [f"README.md is {n} lines, over the {README_MAX_LINES}-line ceiling"]
    return []


def check_no_unscoped_deployed_sha() -> list[str]:
    problems = []
    for md in _iter_markdown():
        rel = md.relative_to(ROOT).as_posix()
        if any(rel.startswith(p) or rel == p for p in HISTORY_EXEMPT_PREFIXES):
            continue
        text = md.read_text(errors="ignore")
        if _CURRENT_STATE_SHA.search(text):
            problems.append(
                f"{rel}: states a deployed/branch SHA as present-tense fact outside "
                "docs/history/ or examples/live-validation/ -- date it and move it, "
                "or rephrase as a dated historical observation"
            )
    return problems


def main() -> int:
    checks = {
        "internal links resolve": check_links,
        "required index files exist": check_required_indexes,
        "README.md length": check_readme_length,
        "no unscoped deployed-SHA claim": check_no_unscoped_deployed_sha,
    }
    all_problems: list[str] = []
    for name, fn in checks.items():
        problems = fn()
        status = "pass" if not problems else "FAIL"
        print(f"[{status}] {name} ({len(problems)} problem(s))")
        all_problems.extend(problems)

    if all_problems:
        print("\nProblems:")
        for p in all_problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
