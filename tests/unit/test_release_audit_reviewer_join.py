"""The independent-reviewer join (`examples/release-audit/reviewer_packet.py
join`) is the tool that decides whether a real external reviewer found a
published claim their own quote does not support. Getting the validation
or the join logic wrong here would either silently drop a real finding or
fabricate one, so both are tested directly -- not just smoke-checked by
running the script once and eyeballing the output.

Two kinds of test: synthetic ids/labels for every validation-failure path
(a real 21-case packet can't exercise "wrong count" or "stray id" without
first being broken), and the real committed packet, labels and audit for
the join itself -- so a future edit to any of those three real files that
silently changes the known result (one unsupported-published claim,
`nist-ai-risk-framework-5`) fails here instead of going unnoticed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "release-audit"))

import reviewer_packet as m  # noqa: E402

REAL_IDS = {f"case-{i}" for i in range(21)}


class TestValidateLabels:
    def test_accepts_exactly_matching_valid_labels(self) -> None:
        labels = dict.fromkeys(REAL_IDS, "supported")
        m.validate_labels(REAL_IDS, labels)  # must not raise

    def test_rejects_a_packet_that_is_not_21_cases(self) -> None:
        short_ids = {f"case-{i}" for i in range(20)}
        labels = dict.fromkeys(short_ids, "supported")
        with pytest.raises(m.ValidationError, match="21"):
            m.validate_labels(short_ids, labels)

    def test_rejects_fewer_labels_than_packet_cases(self) -> None:
        labels = {cid: "supported" for cid in REAL_IDS if cid != "case-0"}
        with pytest.raises(m.ValidationError, match="missing"):
            m.validate_labels(REAL_IDS, labels)

    def test_rejects_an_unexpected_extra_label_id(self) -> None:
        labels = dict.fromkeys(REAL_IDS, "supported")
        labels["not-a-real-case"] = "supported"
        with pytest.raises(m.ValidationError, match="unexpected"):
            m.validate_labels(REAL_IDS, labels)

    def test_rejects_an_invalid_label_value(self) -> None:
        labels = dict.fromkeys(REAL_IDS, "supported")
        labels["case-0"] = "yes"
        with pytest.raises(m.ValidationError, match="invalid or unfilled"):
            m.validate_labels(REAL_IDS, labels)

    def test_rejects_an_unfilled_null_label(self) -> None:
        """The template `reviewer-labels.json` ships with every value
        `null`; a label left unfilled must fail the same way a wrong
        value does, not be silently treated as a valid answer."""
        labels = dict.fromkeys(REAL_IDS, "supported")
        labels["case-0"] = None
        with pytest.raises(m.ValidationError, match="invalid or unfilled"):
            m.validate_labels(REAL_IDS, labels)

    def test_rejects_case_sensitive_variants(self) -> None:
        """Only the three exact lowercase values are accepted -- a
        reviewer writing "Supported" or "SUPPORTED" must be caught, not
        silently coerced."""
        labels = dict.fromkeys(REAL_IDS, "supported")
        labels["case-0"] = "Supported"
        with pytest.raises(m.ValidationError, match="invalid or unfilled"):
            m.validate_labels(REAL_IDS, labels)


class TestJoinOnRealData:
    """Exercises the real, committed reviewer-packet.json, reviewer-labels.json
    and candidate-audit.json. Writes its output to tmp_path, never to the
    real reviewer-audit.json, so the test has no side effect on the repo."""

    def test_the_real_packet_has_exactly_21_cases(self) -> None:
        packet = json.loads(m.PACKET.read_text())
        assert len(packet["cases"]) == 21
        assert len({c["case_id"] for c in packet["cases"]}) == 21

    def test_the_real_returned_labels_pass_validation(self) -> None:
        packet = json.loads(m.PACKET.read_text())
        labels = json.loads(m.LABELS.read_text())
        m.validate_labels({c["case_id"] for c in packet["cases"]}, labels)  # must not raise

    def test_join_finds_exactly_one_published_unsupported_claim(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(m, "JOINED", tmp_path / "reviewer-audit.json")

        m.join()

        result = json.loads(m.JOINED.read_text())
        assert result["confusion"] == {"tp": 10, "fp": 1, "fn": 6, "tn": 4}
        assert result["uncertain_published"] == []
        assert [c["case_id"] for c in result["unsupported_published"]] == [
            "nist-ai-risk-framework-5"
        ]

    def test_the_one_finding_is_a_claim_the_engine_actually_published(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The headline question this tool exists to answer: not the
        16/4/1 label split by itself, but whether any negative label
        lands on a claim the engine actually published."""
        monkeypatch.setattr(m, "JOINED", tmp_path / "reviewer-audit.json")

        m.join()

        result = json.loads(m.JOINED.read_text())
        finding = result["unsupported_published"][0]
        matching_case = next(c for c in result["cases"] if c["case_id"] == finding["case_id"])
        assert matching_case["published"] is True
        assert matching_case["human_label"] == "unsupported"

    def test_output_states_it_is_validation_not_a_benchmark(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(m, "JOINED", tmp_path / "reviewer-audit.json")

        m.join()

        result = json.loads(m.JOINED.read_text())
        assert "not a statistical benchmark" in result["validation_note"]
        assert "n=1 reviewer" in result["validation_note"]

    def test_output_carries_a_process_attestation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(m, "JOINED", tmp_path / "reviewer-audit.json")

        m.join()

        result = json.loads(m.JOINED.read_text())
        attestation = result["process_attestation"]
        assert (
            "independent of the person and process that built the verifier"
            in (attestation["reviewer_independence"])
        )
        assert "before any automated outcome existed to see" in attestation["blinding"]
        assert "own shown quote alone" in attestation["basis"]

    def test_join_does_not_mutate_the_labels_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(m, "JOINED", tmp_path / "reviewer-audit.json")
        before = m.LABELS.read_text()

        m.join()

        assert m.LABELS.read_text() == before
