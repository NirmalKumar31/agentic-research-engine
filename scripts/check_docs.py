"""Documentation gates: run in CI so a documentation regression fails
the build instead of waiting for the next scrutiny pass to find it.

Four checks, each independently actionable:

1. **Internal links resolve.** Every `[text](path)` markdown link whose
   target is a repo-relative path (not `http(s)://`, not a bare
   `#anchor`) must point at a file or directory that is actually
   tracked by git -- not merely present on the machine running the
   check. A link validated against local disk (`Path.exists()`) can
   pass on a checkout that has extra untracked files -- an adjudication
   packet, a local scratch artifact, anything gitignored -- and still
   break on CI's checkout, which has only what git tracks. That exact
   gap broke CI once: `evaluations/phase_b/README.md` linked
   `adjudication/README.md`, which existed locally (intentionally never
   committed) and resolved on this machine, then failed on a fresh
   checkout. Every check in this module that touches file existence
   now goes through `git ls-files`, not the filesystem, for that
   reason.
2. **Required index files exist.** A handful of directories are large
   enough, or deliberately public enough, that a visitor needs an entry
   point rather than a raw file listing. "Exist" means tracked, for the
   same reason as above -- the two `evaluations/` indexes were once
   silently dropped by a blanket `.gitignore` rule and only `git status`
   caught it before commit.
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

import posixpath
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

README_MAX_LINES = 300

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


def tracked_files(root: Path) -> set[str]:
    """Every path git actually tracks, repo-relative, posix-separated.

    Not `rglob` + `.exists()`: those see whatever happens to be on the
    machine running the check, which is a superset of what a fresh
    clone gets. CI's checkout, and anyone else's, only ever has this
    set.
    """
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    return set(out.splitlines())


def _tracked_markdown(tracked: set[str]) -> list[str]:
    return sorted(f for f in tracked if f.endswith(".md"))


def _tracked_dirs(tracked: set[str]) -> set[str]:
    """Every directory implied by a tracked file's path.

    Git has no directory objects, so a link to `history/` (a directory)
    has nothing in `tracked` to match exactly -- it is real only
    because files live under it.
    """
    dirs: set[str] = set()
    for f in tracked:
        parts = f.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    return dirs


def check_links(root: Path, tracked: set[str]) -> list[str]:
    problems = []
    tracked_dirs = _tracked_dirs(tracked)
    for rel in _tracked_markdown(tracked):
        text = (root / rel).read_text(errors="ignore")
        for m in re.finditer(r"\]\(([^)]+)\)", text):
            link = m.group(1)
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            path_part = link.split("#")[0].rstrip("/")
            if not path_part:
                continue
            target_rel = posixpath.normpath(posixpath.join(posixpath.dirname(rel), path_part))
            if target_rel.startswith(".."):
                continue  # escapes the repo root; not ours to validate
            if target_rel in tracked or target_rel in tracked_dirs:
                continue
            problems.append(f"{rel}: broken link -> {link} (not a file or directory git tracks)")
    return problems


def check_required_indexes(tracked: set[str]) -> list[str]:
    return [f"missing required index: {rel}" for rel in REQUIRED_INDEXES if rel not in tracked]


def check_readme_length(root: Path) -> list[str]:
    readme = root / "README.md"
    n = len(readme.read_text().splitlines())
    if n > README_MAX_LINES:
        return [f"README.md is {n} lines, over the {README_MAX_LINES}-line ceiling"]
    return []


def check_no_unscoped_deployed_sha(root: Path, tracked: set[str]) -> list[str]:
    problems = []
    for rel in _tracked_markdown(tracked):
        if any(rel.startswith(p) or rel == p for p in HISTORY_EXEMPT_PREFIXES):
            continue
        text = (root / rel).read_text(errors="ignore")
        if _CURRENT_STATE_SHA.search(text):
            problems.append(
                f"{rel}: states a deployed/branch SHA as present-tense fact outside "
                "docs/history/ or examples/live-validation/ -- date it and move it, "
                "or rephrase as a dated historical observation"
            )
    return problems


def main() -> int:
    tracked = tracked_files(ROOT)
    checks = {
        "internal links resolve": lambda: check_links(ROOT, tracked),
        "required index files exist": lambda: check_required_indexes(tracked),
        "README.md length": lambda: check_readme_length(ROOT),
        "no unscoped deployed-SHA claim": lambda: check_no_unscoped_deployed_sha(ROOT, tracked),
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
