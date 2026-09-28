"""Prove the evaluation set covers what the release requires.

A manifest that lists categories is a claim. This derives coverage
from the cases themselves and fails when something is missing, so the
set cannot quietly stop covering a requirement while still looking
complete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent

REQUIRED_CATEGORIES = {
    "comparison",
    "definition",
    "causal",
    "temporal",
    "numeric",
    "procedural",
    "synthesis",
    "insufficient_evidence",
    "terminology_mismatch",
}

REQUIRED_REGRESSIONS = {
    "llm_vs_neural_network",
    "arxiv_vs_tweet_or_wikipedia",
    "supported_but_irrelevant",
    "authoritative_but_irrelevant",
    "definition_only_comparison_answer",
    "wrong_entity",
    "wrong_period_or_scope",
    "repairable_wording",
    "unsupported_wording_repair_must_not_rescue",
    "responsible_no_answer",
    "bundled_partial_support",
}

# Beyond what the release requires. Listed so a genuine typo still
# shows up as unknown rather than being waved through.
ADDITIONAL_REGRESSIONS = {
    "multi_part_partial_coverage",
    "terminology_differs_from_source",
}


def load() -> dict:
    return json.loads((HERE / "cases.json").read_text())


class Coverage(NamedTuple):
    categories: dict[str, list[str]]
    regressions: dict[str, list[str]]
    missing_categories: list[str]
    missing_regressions: list[str]
    unknown_regressions: list[str]
    additional_regressions: list[str]
    cases: int
    claims: int


def coverage() -> Coverage:
    spec = load()
    categories: dict[str, list[str]] = {}
    regressions: dict[str, list[str]] = {}
    for case in spec["cases"]:
        categories.setdefault(case["contract"]["question_type"], []).append(case["id"])
        for tag in case.get("regressions", []):
            regressions.setdefault(tag, []).append(case["id"])
    return Coverage(
        categories=categories,
        regressions=regressions,
        missing_categories=sorted(REQUIRED_CATEGORIES - set(categories)),
        missing_regressions=sorted(REQUIRED_REGRESSIONS - set(regressions)),
        unknown_regressions=sorted(
            set(regressions) - REQUIRED_REGRESSIONS - ADDITIONAL_REGRESSIONS
        ),
        additional_regressions=sorted(set(regressions) & ADDITIONAL_REGRESSIONS),
        cases=len(spec["cases"]),
        claims=sum(len(c["claims"]) for c in spec["cases"]),
    )


def main() -> int:
    c = coverage()
    print(f"cases {c.cases}, claims {c.claims}\n")
    print("question categories")
    for name in sorted(REQUIRED_CATEGORIES):
        cases = c.categories.get(name, [])
        mark = "OK " if cases else "MISSING"
        print(f"  {mark}  {name:24} {', '.join(cases) or '-'}")
    print("\nrequired regressions")
    for name in sorted(REQUIRED_REGRESSIONS):
        cases = c.regressions.get(name, [])
        mark = "OK " if cases else "MISSING"
        print(f"  {mark}  {name:44} {', '.join(cases) or '-'}")
    if c.additional_regressions:
        print(f"\nadditional coverage: {', '.join(c.additional_regressions)}")
    missing = c.missing_categories + c.missing_regressions
    if c.unknown_regressions:
        print(f"\nunknown regression tags: {c.unknown_regressions}")
    print(f"\n{'COMPLETE' if not missing else 'INCOMPLETE: ' + ', '.join(missing)}")
    return 0 if not missing and not c.unknown_regressions else 1


if __name__ == "__main__":
    raise SystemExit(main())
