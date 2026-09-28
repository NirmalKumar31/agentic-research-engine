"""The frozen baseline stays frozen, and the harness keeps working.

The v1.2.0 work is measured against what the pipeline did before it,
so the comparison point has to be immutable and the harness that
produced it has to keep running. Without this, "clear improvement over
the baseline" becomes a claim nobody can check.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "examples" / "quality-eval"
BASELINE = EVAL / "baseline-v1.1.1.json"


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE.read_text())


class TestTheBaselineRecordsTheDefect:
    """If these ever pass trivially, the baseline stopped describing
    the problem and the comparison is worthless."""

    def test_it_publishes_irrelevant_claims(self, baseline: dict) -> None:
        assert baseline["summary"]["irrelevant_published"] == 6

    def test_it_selects_evidence_by_entailment_alone(self, baseline: dict) -> None:
        """A tweet at 0.994 beats an arXiv paper at 0.985."""
        assert baseline["summary"]["evidence_selection_wrong"] == 1

    def test_it_over_publishes(self, baseline: dict) -> None:
        s = baseline["summary"]
        assert s["published"] > s["expected_published"]

    def test_the_lowest_cited_source_is_a_tweet(self, baseline: dict) -> None:
        assert baseline["summary"]["min_selected_source_quality"] == 0.55


class TestTheCasesAreWellFormed:
    @pytest.fixture(scope="class")
    @classmethod
    def spec(cls) -> dict:
        return json.loads((EVAL / "cases.json").read_text())

    def test_every_case_declares_a_contract(self, spec: dict) -> None:
        for case in spec["cases"]:
            contract = case["contract"]
            assert contract["question_type"]
            assert contract["required_slots"]

    def test_every_claim_cites_evidence_that_exists(self, spec: dict) -> None:
        for case in spec["cases"]:
            ids = {e["id"] for e in case["evidence"]}
            for claim in case["claims"]:
                assert set(claim["cites"]) <= ids, f"{case['id']}/{claim['id']}"

    def test_every_claim_pins_entailment_for_what_it_cites(self, spec: dict) -> None:
        """An unpinned pair would fall to the default and make the run
        depend on a number nobody chose."""
        for case in spec["cases"]:
            for claim in case["claims"]:
                assert set(claim["entailment"]) == set(claim["cites"]), (
                    f"{case['id']}/{claim['id']}"
                )

    def test_every_evidence_item_names_a_real_source(self, spec: dict) -> None:
        for case in spec["cases"]:
            ids = {s["id"] for s in case["sources"]}
            for item in case["evidence"]:
                assert item["source_id"] in ids

    def test_irrelevant_claims_are_never_expected_to_publish(self, spec: dict) -> None:
        """The property the whole release is for."""
        for case in spec["cases"]:
            for claim in case["claims"]:
                if not claim["expected_relevant"]:
                    assert claim["expected_publish"] is False

    def test_the_set_covers_every_required_category_and_regression(self) -> None:
        """Derived from the cases, not asserted in prose. A set that
        quietly stops covering a requirement must fail here rather
        than keep looking complete."""
        import sys

        sys.path.insert(0, str(EVAL))
        from coverage import coverage

        c = coverage()
        assert not c.missing_categories, c.missing_categories
        assert not c.missing_regressions, c.missing_regressions
        assert not c.unknown_regressions, c.unknown_regressions

    def test_every_case_declares_its_regressions(self, spec: dict) -> None:
        for case in spec["cases"]:
            assert "regressions" in case, case["id"]


class TestTheHarnessStillRuns:
    def test_it_reproduces_the_current_recorded_result(self) -> None:
        """Run live rather than trusting the file.

        This used to compare against the frozen baseline, which was
        right until the pipeline started improving and then asserted
        that nothing had. The baseline stays immutable as the
        comparison point; what must be reproducible is where the
        pipeline is *now*.
        """
        proc = subprocess.run(
            [sys.executable, str(EVAL / "run_eval.py"), "--json", "/dev/stdout"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        payload = proc.stdout[proc.stdout.index("{") :]
        fresh = json.loads(payload[: payload.rindex("}") + 1])
        recorded = json.loads((EVAL / "current.json").read_text())
        assert fresh["summary"] == recorded["summary"], (
            "current.json is stale; regenerate it with run_eval.py --json"
        )


class TestNothingRegressedAgainstTheBaseline:
    """Improvement is measured, not asserted.

    Every metric here may only move in one direction. A change that
    publishes more by publishing worse fails these.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def pair(cls) -> tuple[dict, dict]:
        base = json.loads(BASELINE.read_text())["summary"]
        now = json.loads((EVAL / "current.json").read_text())["summary"]
        return base, now

    def test_irrelevant_publications_never_increase(self, pair: tuple[dict, dict]) -> None:
        base, now = pair
        assert now["irrelevant_published"] <= base["irrelevant_published"]

    def test_wrong_evidence_selections_never_increase(self, pair: tuple[dict, dict]) -> None:
        base, now = pair
        assert now["evidence_selection_wrong"] <= base["evidence_selection_wrong"]

    def test_correct_claims_are_not_newly_withheld(self, pair: tuple[dict, dict]) -> None:
        """A gate that fixes relevance by withholding everything is not
        a fix."""
        base, now = pair
        assert now["correct_claims_wrongly_withheld"] <= base["correct_claims_wrongly_withheld"]

    def test_cited_sources_never_get_weaker(self, pair: tuple[dict, dict]) -> None:
        base, now = pair
        assert now["min_selected_source_quality"] >= base["min_selected_source_quality"]
        assert now["mean_selected_source_quality"] >= base["mean_selected_source_quality"]

    def test_the_baseline_itself_is_unchanged(self) -> None:
        """cc4e12c froze these numbers. They are the comparison point
        and may not be edited to make a later result look better."""
        base = json.loads(BASELINE.read_text())["summary"]
        assert base["published"] == 14
        assert base["irrelevant_published"] == 6
        assert base["evidence_selection_wrong"] == 1
        assert base["min_selected_source_quality"] == 0.55


class TestRepairCannotRescueAnUnsupportedProposition:
    """A specification property, asserted in the fixture itself.

    A claim bundling a supported number with an unsupported assertion
    must be withheld permanently. Repair exists for wording, after
    every proposition is already supported; deleting the unsupported
    half and publishing the remainder would be laundering, not repair.
    The supported half reaches print only if generation emits it
    upstream as its own draft and it passes every gate independently.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def bundled(cls) -> dict:
        spec = json.loads((EVAL / "cases.json").read_text())
        case = next(c for c in spec["cases"] if c["id"] == "zero-publication-productivity")
        return {c["id"]: c for c in case["claims"]}

    def test_the_bundled_claim_is_never_publishable(self, bundled: dict) -> None:
        c1 = bundled["c1"]
        assert c1["expected_publish"] is False
        assert c1["expected_after_repair"] is False

    def test_the_supported_half_stands_on_its_own(self, bundled: dict) -> None:
        c2 = bundled["c2"]
        assert c2["expected_publish"] is True
        assert c2["cites"] == c1_cites(bundled), (
            "the independent draft must rest on the same evidence, not new evidence"
        )

    def test_no_case_expects_repair_to_rescue_an_unsupported_claim(self) -> None:
        spec = json.loads((EVAL / "cases.json").read_text())
        for case in spec["cases"]:
            for claim in case["claims"]:
                if claim.get("expected_after_repair") is True:
                    assert claim["expected_relevant"] is True, (
                        f"{case['id']}/{claim['id']}: repair may not rescue irrelevance"
                    )


def c1_cites(bundled: dict) -> list[str]:
    return bundled["c1"]["cites"]
