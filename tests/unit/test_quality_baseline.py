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
        assert baseline["summary"]["irrelevant_published"] == 3

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

    def test_the_set_covers_several_question_types(self, spec: dict) -> None:
        types = {c["contract"]["question_type"] for c in spec["cases"]}
        assert len(types) >= 3, f"only {types}"


class TestTheHarnessStillRuns:
    def test_it_reproduces_the_recorded_baseline(self, baseline: dict) -> None:
        """Run live rather than trusting the file: a harness that has
        drifted from its own recorded output is measuring nothing."""
        proc = subprocess.run(
            [sys.executable, str(EVAL / "run_eval.py"), "--json", "/dev/stdout"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        payload = proc.stdout[proc.stdout.index("{") :]
        fresh = json.loads(payload[: payload.rindex("}") + 1])
        assert fresh["summary"] == baseline["summary"], (
            "the harness no longer reproduces the frozen baseline"
        )
