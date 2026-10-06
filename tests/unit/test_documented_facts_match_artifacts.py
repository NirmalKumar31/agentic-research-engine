"""Reader-facing documentation must not silently drift from the
machine-readable artifacts it describes -- the way it did four times
before this test module existed. An independent, context-free audit
found all four; no test caught them, because no test checked for them.
This module exists so the next drift of this exact shape is caught
here, not by the next audit:

1. `docs/LIMITATIONS.md` said "the twelve-question benchmark has never
   been run" after Phase B had already executed (PR #26).
2. The same file said "no clean local-versus-cloud comparison exists"
   after Phase B produced one.
3. The same file stated the release audit as "11 of 11 supported, 0
   unsupported" with no disclosure that an independent reviewer later
   found one of those eleven unsupported, and that a second adjudicator
   disagreed (issue #32) -- stating the self-review's number as if it
   were the whole, current answer.
4. The same file named three verifier-experiment branches
   (`verifier-v1`, `verifier-v2`, `verifier-v3`) that do not exist
   under those names, locally or on the remote.

All four historical texts below are pulled verbatim from
`git show 71802cb:docs/LIMITATIONS.md` -- the commit immediately before
the fix -- not reconstructed from memory. `TestChecksRejectTheRealHistoricalBugs`
proves each check actually rejects the real bug, not just a
hypothetical one a looser check would also catch.

Facts are derived from the committed Phase B and release-audit
artifacts directly, not duplicated as hand-kept constants, so a future
artifact regeneration that changes a number is what breaks this test --
not a stale copy of today's numbers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PHASE_B_DIR = ROOT / "evaluations" / "phase_b"
RELEASE_AUDIT_DIR = ROOT / "examples" / "release-audit"

README = ROOT / "README.md"
LIMITATIONS = ROOT / "docs" / "LIMITATIONS.md"
VALIDATION_HISTORY = ROOT / "docs" / "VALIDATION-HISTORY.md"
PHASE_B_RESULTS = PHASE_B_DIR / "RESULTS.md"
RELEASE_AUDIT_README = RELEASE_AUDIT_DIR / "README.md"

READER_FACING_DOCS = {
    "README.md": README,
    "docs/LIMITATIONS.md": LIMITATIONS,
    "docs/VALIDATION-HISTORY.md": VALIDATION_HISTORY,
    "evaluations/phase_b/RESULTS.md": PHASE_B_RESULTS,
    "examples/release-audit/README.md": RELEASE_AUDIT_README,
}


# ---------------------------------------------------------------------
# Facts derived from committed, machine-readable artifacts -- not from
# hand-kept constants.
# ---------------------------------------------------------------------


def _phase_b_facts() -> dict[str, object]:
    raw = json.loads((PHASE_B_DIR / "raw_results.json").read_text())
    ledger = json.loads((PHASE_B_DIR / "spend_ledger.json").read_text())
    return {
        "total_arm_runs": len(raw),
        "completed": sum(1 for r in raw if not r["timed_out"]),
        "timed_out": sum(1 for r in raw if r["timed_out"]),
        "timed_out_all_local": all(r["arm"] == "local" for r in raw if r["timed_out"]),
        "spend_usd": ledger["spent_usd"],
    }


def _release_audit_facts() -> dict[str, object]:
    audit = json.loads((RELEASE_AUDIT_DIR / "candidate-audit.json").read_text())
    reviewer_audit = json.loads((RELEASE_AUDIT_DIR / "reviewer-audit.json").read_text())
    evidence_result = json.loads((RELEASE_AUDIT_DIR / "evidence-review-result.json").read_text())
    return {
        "candidates": audit["totals"]["candidates"],
        "published": audit["totals"]["published"],
        "unsupported_published_count": len(reviewer_audit["unsupported_published"]),
        "uncertain_published_count": len(reviewer_audit["uncertain_published"]),
        "second_opinion_label": evidence_result["second_adjudicator_label"],
        "first_reviewer_label_on_disputed_case": evidence_result["first_reviewer_label"],
    }


# ---------------------------------------------------------------------
# Checks. Each returns a list of problems (empty == clean), so the same
# function is asserted "no problems" against real docs and "has
# problems" against reconstructed historical bug text below -- proving
# the check actually discriminates, rather than always passing.
# ---------------------------------------------------------------------

_UNRUN_CLAIM = re.compile(
    r"benchmark has never been run|no run has been (?:authorized|executed)", re.I
)
_NO_COMPARISON_CLAIM = re.compile(r"no clean local-versus-cloud comparison exists", re.I)
_ELEVEN_OF_ELEVEN = re.compile(r"11 of 11 supported,?\s+0\s+unsupported", re.I)
_FAKE_VERIFIER_BRANCHES = re.compile(
    r"`verifier-v1`,?\s*`verifier-v2`\s*and\s*`verifier-v3`\s*branches", re.I
)
_DISSENT_DISCLOSED = re.compile(
    r"issue #32|second (?:opinion|adjudicator)|independent review\w*.{0,40}(?:found|judged)",
    re.I,
)


def check_phase_b_not_described_as_unrun(text: str) -> list[str]:
    return [f"describes Phase B as unrun: {m.group(0)!r}" for m in _UNRUN_CLAIM.finditer(text)]


def check_no_stale_comparison_claim(text: str) -> list[str]:
    return [
        f"claims no clean comparison exists: {m.group(0)!r}"
        for m in _NO_COMPARISON_CLAIM.finditer(text)
    ]


def check_eleven_of_eleven_is_scoped_and_disclosed(text: str) -> list[str]:
    problems = []
    for m in _ELEVEN_OF_ELEVEN.finditer(text):
        window = text[max(0, m.start() - 300) : m.start()]
        if not re.search(r"self.review", window, re.I):
            problems.append(
                f"'11 of 11 supported, 0 unsupported' at char {m.start()} is not scoped "
                "to a nearby mention of the self-review"
            )
        if not _DISSENT_DISCLOSED.search(text):
            problems.append(
                "states the self-review's '0 unsupported' without disclosing the "
                "independent reviewer's later dissent anywhere in the same document"
            )
    return problems


def check_no_fake_verifier_branch_names(text: str) -> list[str]:
    return [
        f"names nonexistent branches: {m.group(0)!r}"
        for m in _FAKE_VERIFIER_BRANCHES.finditer(text)
    ]


ALL_CHECKS = {
    "phase_b_not_unrun": check_phase_b_not_described_as_unrun,
    "no_stale_comparison_claim": check_no_stale_comparison_claim,
    "eleven_of_eleven_scoped_and_disclosed": check_eleven_of_eleven_is_scoped_and_disclosed,
    "no_fake_verifier_branches": check_no_fake_verifier_branch_names,
}


# ---------------------------------------------------------------------
# Verbatim historical bug text: `git show 71802cb:docs/LIMITATIONS.md`,
# the commit immediately before PR #37's fix. Not reconstructed from
# memory.
# ---------------------------------------------------------------------

HISTORICAL_UNRUN_TEXT = (
    "**The twelve-question benchmark has never been run.** Construction and\n"
    "validation for it exist — the question set, a frozen-configuration\n"
    "manifest, and a blinded-review mechanism — at\n"
    "[`docs/BENCHMARK-PROTOCOL.md`](BENCHMARK-PROTOCOL.md). No run has been\n"
    "authorized or executed; that document states the exact cost and stop\n"
    "rule a run would need approved first.\n"
)

HISTORICAL_NO_COMPARISON_TEXT = (
    "**No clean local-versus-cloud comparison exists.** The earlier one used a\n"
    "corpus whose source text had been stripped, which drove citation integrity\n"
    "to 0% in both arms. It is archived under `examples/archive/` and excluded\n"
    "from current results; its 55.6%/81.2% figures are not valid measurements.\n"
)

HISTORICAL_ELEVEN_OF_ELEVEN_TEXT = (
    "**The release audit passes, and it took four attempts.** Every\n"
    "published claim in the three canonical recordings was read against the\n"
    "exact quote the gate chose for it: 11 of 11 supported, 0 unsupported.\n"
    "The first three audits each published exactly one claim that survived\n"
    "every automated check and failed a human read — a deleted hedge\n"
    '("might lack" as "lack"), a deleted research voice ("We demonstrate\n'
    'that X" as "X"), and a first-person scope deletion hidden inside a\n'
    "two-sentence claim. Each produced a general rule. None of that is\n"
    "evidence the next audit would be clean: the runs use live search and\n"
    "produce different claims every time, and the only thing that caught\n"
    "these was reading every published claim by hand.\n"
)

HISTORICAL_FAKE_BRANCHES_TEXT = (
    "Three designs using `qwen3:4b` as an entailment classifier were measured\n"
    "against human labels and all three failed: the first returned `supported`\n"
    "for none of thirty claims, the second for twenty-five of thirty including\n"
    "sixteen the reviewer had marked otherwise, and the third produced\n"
    "malformed audits on eleven of thirty. Those experiments are preserved on\n"
    "the `verifier-v1`, `verifier-v2` and `verifier-v3` branches. The\n"
    "conclusion was that a 4B instruction model is not a stable semantic\n"
    "classifier, not that the prompt needed more work.\n"
)


class TestPhaseBFactsMatchArtifacts:
    def test_facts_derive_to_the_known_values(self) -> None:
        facts = _phase_b_facts()
        assert facts["total_arm_runs"] == 48
        assert facts["completed"] == 43
        assert facts["timed_out"] == 5
        assert facts["timed_out_all_local"] is True
        assert facts["spend_usd"] == 0.027093

    @pytest.mark.parametrize(
        "doc_name,path",
        [
            ("evaluations/phase_b/RESULTS.md", PHASE_B_RESULTS),
            ("docs/LIMITATIONS.md", LIMITATIONS),
        ],
    )
    def test_doc_states_the_derived_run_counts(self, doc_name: str, path: Path) -> None:
        facts = _phase_b_facts()
        text = path.read_text()
        assert str(facts["total_arm_runs"]) in text, f"{doc_name} missing total arm-run count"
        assert str(facts["completed"]) in text, f"{doc_name} missing completed-run count"
        assert str(facts["timed_out"]) in text, f"{doc_name} missing timed-out count"
        assert f"{facts['spend_usd']:.6f}" in text, f"{doc_name} missing the exact spend figure"


class TestReleaseAuditFactsMatchArtifacts:
    def test_facts_derive_to_the_known_values(self) -> None:
        facts = _release_audit_facts()
        assert facts["candidates"] == 21
        assert facts["published"] == 11
        assert facts["unsupported_published_count"] == 1
        assert facts["uncertain_published_count"] == 0
        assert facts["second_opinion_label"] == "supported"
        assert facts["first_reviewer_label_on_disputed_case"] == "unsupported"

    def test_readme_states_the_derived_candidate_and_published_counts(self) -> None:
        facts = _release_audit_facts()
        text = README.read_text()
        assert f"{facts['candidates']} unique candidates" in text
        match = re.search(r"\*\*(\d+)\s+published,\s+(\d+)\s+withheld\*\*", text)
        assert match, "README's published/withheld aggregate is missing or reworded"
        assert int(match.group(1)) == facts["published"]


class TestNoStaleClaimsInCurrentDocs:
    """Each check runs against every reader-facing document. Most will
    vacuously pass (the pattern simply does not appear there), which is
    correct: the point is that where a pattern *could* appear, it is
    either absent (unrun claim, stale comparison claim, fake branches)
    or scoped and disclosed (the self-review's clean count)."""

    @pytest.mark.parametrize("doc_name,path", list(READER_FACING_DOCS.items()))
    def test_current_doc_passes_every_check(self, doc_name: str, path: Path) -> None:
        text = path.read_text()
        problems = []
        for check in ALL_CHECKS.values():
            problems.extend(check(text))
        assert not problems, f"{doc_name}: {problems}"


class TestChecksRejectTheRealHistoricalBugs:
    """Non-vacuity: a simple text-presence assertion is not sufficient
    unless it actually rejects the real, verbatim historical bug. If
    any test here passed (found no problems), the corresponding check
    above would not have caught the real mistake it exists to catch."""

    def test_rejects_the_real_unrun_claim(self) -> None:
        assert check_phase_b_not_described_as_unrun(HISTORICAL_UNRUN_TEXT)

    def test_rejects_the_real_no_comparison_claim(self) -> None:
        assert check_no_stale_comparison_claim(HISTORICAL_NO_COMPARISON_TEXT)

    def test_rejects_the_real_unscoped_eleven_of_eleven_claim(self) -> None:
        assert check_eleven_of_eleven_is_scoped_and_disclosed(HISTORICAL_ELEVEN_OF_ELEVEN_TEXT)

    def test_rejects_the_real_fake_verifier_branch_names(self) -> None:
        assert check_no_fake_verifier_branch_names(HISTORICAL_FAKE_BRANCHES_TEXT)


class TestPhaseBNeverClaimsSignificance:
    """Phase B must be describable as a completed, descriptive study --
    never as unrun (covered above) and never with an unqualified
    statistical-significance claim. "statistically significant" is only
    ever allowed negated ("not statistically significant", "no
    significance claimed")."""

    _SIGNIFICANCE = re.compile(r"statistical(?:ly)? significan\w*", re.I)

    @pytest.mark.parametrize("doc_name,path", list(READER_FACING_DOCS.items()))
    def test_no_unqualified_significance_claim(self, doc_name: str, path: Path) -> None:
        text = path.read_text()
        for m in self._SIGNIFICANCE.finditer(text):
            window = text[max(0, m.start() - 40) : m.start()]
            assert re.search(r"\bno\b|\bnot\b", window, re.I), (
                f"{doc_name}: unqualified significance claim near "
                f"{text[max(0, m.start() - 60) : m.start() + 60]!r}"
            )

    def test_rejects_an_unqualified_significance_claim(self) -> None:
        """Non-vacuity for the check above."""
        bad = "The results are statistically significant across both arms."
        m = self._SIGNIFICANCE.search(bad)
        assert m
        window = bad[max(0, m.start() - 40) : m.start()]
        assert not re.search(r"\bno\b|\bnot\b", window, re.I)
