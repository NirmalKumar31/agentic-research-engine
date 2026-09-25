"""The calibration workflow is blind by construction, not by discipline.

The first fixture put the verifier's verdict beside the claim and told
the reviewer not to look at it. That is not blind: a label produced after
seeing the model's answer measures agreement, and the whole point of the
exercise is to find out whether the model is right.

The verdicts live in a separate file and are joined by case_id
afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "examples/verifier-calibration"
BLIND = ROOT / "blind_cases.json"
VERDICTS = ROOT / "cases.json"


@pytest.fixture
def blind() -> dict:
    return json.loads(BLIND.read_text(encoding="utf-8"))


@pytest.fixture
def verdicts() -> dict:
    return json.loads(VERDICTS.read_text(encoding="utf-8"))


class TestTheBlindSetIsBlind:
    def test_it_carries_no_verdict_anywhere(self, blind: dict) -> None:
        raw = json.dumps(blind)
        assert "verifier_verdict" not in raw
        assert "verifier_reason" not in raw

    def test_no_case_leaks_a_verdict_through_another_field(self, blind: dict) -> None:
        for case in blind["cases"]:
            assert set(case) == {
                "case_id",
                "recording",
                "claim",
                "claim_kind",
                "evidence_ids",
                "evidence",
                "human_label",
                "human_rationale",
            }

    def test_order_is_shuffled_with_a_recorded_seed(self, blind: dict) -> None:
        """Recording order would group each run's rejections together and
        hint at the pattern. The seed keeps it reproducible."""
        assert blind["shuffle_seed"]
        ids = [c["case_id"] for c in blind["cases"]]
        assert ids != sorted(ids)

    def test_every_case_is_labelled_with_a_valid_label(self, blind: dict) -> None:
        """Was "labels start empty" until the reviewer labelled them.

        The blindness guarantee is structural -- no verdict is in this
        file -- so it survives labelling. What matters now is that every
        case carries one of the three defined labels.
        """
        valid = set(blind["label_definitions"])
        for case in blind["cases"]:
            assert case["human_label"] in valid, case["case_id"]

    def test_definitions_are_stated_not_implied(self, blind: dict) -> None:
        definitions = blind["label_definitions"]
        assert set(definitions) == {"supported", "partially_supported", "unsupported"}
        # The compound-claim rule is the one the current verifier may be
        # getting wrong, so the human definition has to be explicit.
        assert "COMPLETE cited evidence set" in definitions["supported"]
        assert "Do not require one quote" in definitions["supported"]


class TestNoClaimIsTruncatedOrInvented:
    def test_no_usable_claim_sits_on_the_truncation_boundary(self, blind: dict) -> None:
        """200 characters exactly is the CitationIssue cut."""
        assert all(len(c["claim"]) != 200 for c in blind["cases"])

    def test_the_lost_claims_are_excluded_not_reconstructed(self, blind: dict) -> None:
        excluded = blind["excluded_cases"]
        assert len(excluded) == 4
        for case in excluded:
            assert case["unusable_reason"] == "original claim text was not preserved"
            assert len(case["truncated_claim"]) == 200

    def test_excluded_cases_are_not_in_the_labelling_set(self, blind: dict) -> None:
        excluded_ids = {c["case_id"] for c in blind["excluded_cases"]}
        assert not excluded_ids & {c["case_id"] for c in blind["cases"]}

    def test_every_case_carries_its_evidence_in_full(self, blind: dict) -> None:
        for case in blind["cases"]:
            assert case["claim"]
            assert case["evidence"], case["case_id"]
            for item in case["evidence"]:
                assert item["quote"]
                assert not item["quote"].endswith("...")


class TestTheJoinIsSound:
    def test_every_blind_case_has_a_verdict_to_join_to(self, blind: dict, verdicts: dict) -> None:
        known = {c["case_id"] for c in verdicts["cases"]}
        missing = [c["case_id"] for c in blind["cases"] if c["case_id"] not in known]
        assert missing == []

    def test_merge_pairs_labels_with_verdicts(self, blind: dict, verdicts: dict) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("score", ROOT / "score.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(module)

        merged = module.merge(blind, verdicts)
        assert len(merged) == len(blind["cases"])
        assert all(m["verifier_verdict"] for m in merged)
        assert all(m["human_label"] for m in merged)
        # The join is by case_id, so a shuffled blind file must still
        # pair each label with the verdict for that same claim.
        by_id = {c["case_id"]: c for c in verdicts["cases"]}
        for m in merged:
            assert m["verifier_verdict"] == by_id[m["case_id"]]["verifier_verdict"]

    def test_the_usable_count_is_recorded(self, blind: dict) -> None:
        assert blind["usable_cases"] == len(blind["cases"])

    def test_it_records_the_commit_it_came_from(self, blind: dict) -> None:
        assert blind["source_commit"]
