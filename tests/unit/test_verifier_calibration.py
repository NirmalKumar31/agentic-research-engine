"""The calibration fixture stays honest and the scorer stays correct.

The fixture exists because "the verifier caught 18 overreaches" and "the
verifier is too strict" are equally consistent with the current
recordings. It only settles that if the labels are genuinely independent
of the verdicts -- a label copied from the verifier measures agreement
with itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parents[2] / "examples/verifier-calibration/cases.json"


@pytest.fixture
def doc() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class TestTheFixtureIsUsable:
    def test_every_generated_claim_is_present(self, doc: dict) -> None:
        """One case per generated substantive claim across the three
        recordings. All of them were removed, so every case is a
        rejection and the fixture is the only way to tell a correct
        rejection from an over-strict one."""
        assert len(doc["cases"]) >= 30
        assert all(c["verifier_verdict"] != "supported" for c in doc["cases"])

    def test_each_case_carries_its_evidence_in_full(self, doc: dict) -> None:
        """A labeller cannot judge a claim against evidence it cannot
        read, which is the same defect the verifier had."""
        for case in doc["cases"]:
            assert case["claim"]
            for item in case["evidence"]:
                assert item["quote"]
                assert item["domain"]
                assert "site_category" in item
                assert "page" in item

    def test_evidence_is_not_truncated(self, doc: dict) -> None:
        """The 300-character cut is what made the original verdicts
        unsound; the fixture must not reproduce it."""
        for case in doc["cases"]:
            for item in case["evidence"]:
                assert not item["quote"].endswith("...")

    def test_it_records_the_commit_it_came_from(self, doc: dict) -> None:
        assert doc["source_commit"]

    def test_labels_start_empty(self, doc: dict) -> None:
        """Pre-filling them from the verifier would make the exercise
        circular. This will fail once a human labels them, and that is
        the point at which it should be replaced by an agreement
        assertion."""
        assert all(c["human_label"] is None for c in doc["cases"])

    def test_the_allowed_labels_match_the_verifier(self, doc: dict) -> None:
        assert doc["labels"] == ["supported", "partially_supported", "unsupported"]

    def test_verifier_verdicts_are_recorded_for_comparison(self, doc: dict) -> None:
        assert all(c["verifier_verdict"] in doc["labels"] for c in doc["cases"])


class TestTheScorer:
    @staticmethod
    def _score(cases: list[dict]) -> dict:
        import importlib.util

        spec = importlib.util.spec_from_file_location("score", FIXTURE.parent / "score.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(module)
        return module.score(cases)

    def test_perfect_agreement(self) -> None:
        cases = [
            {"verifier_verdict": "supported", "human_label": "supported"},
            {"verifier_verdict": "unsupported", "human_label": "unsupported"},
        ]
        result = self._score(cases)
        assert result["agreement"] == 1.0
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 0

    def test_a_false_positive_is_an_unsupported_claim_published(self) -> None:
        """The expensive direction."""
        cases = [{"verifier_verdict": "supported", "human_label": "unsupported"}]
        result = self._score(cases)
        assert result["false_positives"] == 1
        assert result["supported_precision"] == 0.0

    def test_a_false_negative_is_a_true_claim_removed(self) -> None:
        cases = [{"verifier_verdict": "partially_supported", "human_label": "supported"}]
        result = self._score(cases)
        assert result["false_negatives"] == 1
        assert result["supported_recall"] == 0.0

    def test_partial_and_unsupported_are_distinguished(self) -> None:
        """Both remove a claim, but confusing one for the other is still
        a disagreement worth seeing in the matrix."""
        cases = [{"verifier_verdict": "partially_supported", "human_label": "unsupported"}]
        result = self._score(cases)
        assert result["agreement"] == 0.0
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 0

    def test_unlabelled_cases_are_ignored_not_counted_as_agreement(self) -> None:
        cases = [
            {"verifier_verdict": "supported", "human_label": "supported"},
            {"verifier_verdict": "supported", "human_label": None},
        ]
        result = self._score(cases)
        assert result["labelled"] == 1
        assert result["unlabelled"] == 1

    def test_an_unlabelled_fixture_scores_nothing(self) -> None:
        assert (
            self._score([{"verifier_verdict": "supported", "human_label": None}])["labelled"] == 0
        )
