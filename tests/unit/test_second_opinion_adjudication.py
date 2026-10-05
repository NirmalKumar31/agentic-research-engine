"""The second-opinion packet (`examples/release-audit/second_opinion.py`)
exists to get an adjudicator's judgement on one disputed claim without
anchoring them to the first reviewer's label, the system's score, its
guard results or its publication decision. If any of those leaked in,
the second opinion would not be independent of the first -- so the
leak-freedom and the join's branching logic are tested directly, not
just exercised once by hand.

A real leak reached the committed packet once: its hand-written
`description` field said outright "A first independent reviewer
labelled this claim unsupported; the system published it." -- caught by
inspection before any human received it, not by this test suite,
because this test suite did not exist yet. `TestLeakRegression` below
reproduces that exact sentence and asserts the current check rejects
it, so the next version of this mistake is caught here first.

No real second adjudicator has looked at this yet (that is a human step
this repo cannot perform), so `join()` is tested against synthetic
first/second label pairs covering all three second-opinion values, not
against a real returned answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "release-audit"))

import second_opinion as m  # noqa: E402


class TestBuild:
    def test_the_disputed_case_is_found_in_the_real_audit(self) -> None:
        candidate = m._find_candidate()
        assert candidate["claim"].startswith("The NIST AI Risk Management Framework 1.0")

    def test_build_writes_exactly_one_case_under_the_external_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(m, "PACKET", tmp_path / "packet.json")
        monkeypatch.setattr(m, "LABEL", tmp_path / "label.json")

        m.build()

        packet = json.loads(m.PACKET.read_text())
        assert len(packet["cases"]) == 1
        case = packet["cases"][0]
        assert case["case_id"] == m.EXTERNAL_CASE_ID
        assert case["case_id"] != m.CASE_ID
        allowed_keys = {
            "case_id",
            "claim",
            "quote",
            "source_title",
            "source_domain",
            "source_id",
            "page",
        }
        assert set(case) == allowed_keys

    def test_build_writes_a_blank_label_template_under_the_external_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(m, "PACKET", tmp_path / "packet.json")
        monkeypatch.setattr(m, "LABEL", tmp_path / "label.json")

        m.build()

        assert json.loads(m.LABEL.read_text()) == {m.EXTERNAL_CASE_ID: None}

    def test_build_does_not_overwrite_an_existing_label_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second adjudicator may have already started filling this in;
        re-running `build` must not wipe their work."""
        monkeypatch.setattr(m, "PACKET", tmp_path / "packet.json")
        label_path = tmp_path / "label.json"
        monkeypatch.setattr(m, "LABEL", label_path)
        label_path.write_text(json.dumps({m.EXTERNAL_CASE_ID: "uncertain"}))

        m.build()

        assert json.loads(label_path.read_text()) == {m.EXTERNAL_CASE_ID: "uncertain"}

    def test_build_raises_rather_than_write_a_packet_whose_claim_text_leaks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: if the underlying claim or quote text itself
        happened to contain a forbidden term, `build` must fail loudly
        instead of silently writing the compromised packet to disk."""
        packet_path = tmp_path / "packet.json"
        monkeypatch.setattr(m, "PACKET", packet_path)
        monkeypatch.setattr(m, "LABEL", tmp_path / "label.json")
        monkeypatch.setattr(
            m,
            "_find_candidate",
            lambda: {
                "claim": "This claim mentions a guard result in passing.",
                "best_evidence_id": "e1",
                "evidence": [{"evidence_id": "e1", "quote": "A quote."}],
            },
        )

        with pytest.raises(AssertionError, match="leaks"):
            m.build()
        assert not packet_path.exists()


class TestLeakRegression:
    """Direct reproduction of the real incident: a hand-written
    description field that narrates the first reviewer's verdict and
    the system's outcome instead of staying neutral."""

    LEAKED_DESCRIPTION = (
        "Second-opinion adjudication for one disputed case from the Agentic Research "
        "Engine release audit (issue #32). A first independent reviewer labelled this "
        "claim unsupported; the system published it. Judge whether the quote supports "
        "the claim -- nothing about the first reviewer's label, the system's score, "
        "its guard results or its publication decision is included here."
    )

    def _packet_with_description(self, description: str) -> dict[str, Any]:
        return {
            "description": description,
            "instructions": "irrelevant to this test",
            "label_definitions": m.DEFINITIONS,
            "cases": [
                {
                    "case_id": m.EXTERNAL_CASE_ID,
                    "claim": "A claim.",
                    "quote": "A quote.",
                }
            ],
        }

    def test_the_actual_leaked_sentence_is_rejected(self) -> None:
        packet = self._packet_with_description(self.LEAKED_DESCRIPTION)
        with pytest.raises(AssertionError, match="leaks"):
            m._assert_not_leaking(packet)

    @pytest.mark.parametrize(
        "term",
        [
            "first reviewer",
            "independent reviewer",
            "issue #32",
            "unsupported",
            "published",
            "guard",
            "score",
            "NLI",
            "threshold",
            "OpenAI",
            "Qwen",
            "nist-ai-risk-framework",
        ],
    )
    def test_each_forbidden_term_is_individually_rejected_in_the_description(
        self, term: str
    ) -> None:
        packet = self._packet_with_description(f"A description mentioning {term} in passing.")
        with pytest.raises(AssertionError, match="leaks"):
            m._assert_not_leaking(packet)

    def test_each_forbidden_term_is_individually_rejected_in_case_data(self) -> None:
        for term in ("unsupported", "issue #32", "nist-ai-risk-framework"):
            packet = {
                "description": "A neutral description.",
                "instructions": "irrelevant",
                "label_definitions": m.DEFINITIONS,
                "cases": [
                    {"case_id": m.EXTERNAL_CASE_ID, "claim": f"mentions {term}", "quote": "q"}
                ],
            }
            with pytest.raises(AssertionError, match="leaks"):
                m._assert_not_leaking(packet)

    def test_the_neutral_real_description_passes(self) -> None:
        """Non-vacuity: confirms a clean packet does not trip the check
        at all, so the test above is catching real leaks, not everything."""
        packet = json.loads(m.PACKET.read_text())
        m._assert_not_leaking(packet)  # must not raise

    def test_generic_label_menu_wording_does_not_false_positive(self) -> None:
        """`instructions` and `label_definitions` legitimately name all
        three label words as a menu of options -- that is not a leak
        about this case's verdict, and must not fail the check."""
        packet = {
            "description": "A neutral description with no verdict in it.",
            "instructions": "Label the case supported / unsupported / uncertain.",
            "label_definitions": m.DEFINITIONS,
            "cases": [{"case_id": m.EXTERNAL_CASE_ID, "claim": "x", "quote": "q"}],
        }
        m._assert_not_leaking(packet)  # must not raise


class TestValidateLabel:
    def test_accepts_a_valid_label(self) -> None:
        m.validate_label({m.EXTERNAL_CASE_ID: "supported"})  # must not raise

    def test_rejects_the_wrong_key(self) -> None:
        with pytest.raises(m.ValidationError, match="exactly one key"):
            m.validate_label({"some-other-case": "supported"})

    def test_rejects_the_internal_case_id_instead_of_the_external_one(self) -> None:
        """A label keyed by the real, internal case_id would mean the
        internal identifier leaked to whoever filled this in."""
        with pytest.raises(m.ValidationError, match="exactly one key"):
            m.validate_label({m.CASE_ID: "supported"})

    def test_rejects_an_extra_key(self) -> None:
        with pytest.raises(m.ValidationError, match="exactly one key"):
            m.validate_label({m.EXTERNAL_CASE_ID: "supported", "extra": "supported"})

    def test_rejects_an_invalid_value(self) -> None:
        with pytest.raises(m.ValidationError, match="invalid or unfilled"):
            m.validate_label({m.EXTERNAL_CASE_ID: "yes"})

    def test_rejects_an_unfilled_null(self) -> None:
        with pytest.raises(m.ValidationError, match="invalid or unfilled"):
            m.validate_label({m.EXTERNAL_CASE_ID: None})


class TestJoin:
    def _run_join(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: str, second: str
    ) -> dict[str, Any]:
        first_labels_path = tmp_path / "reviewer-labels.json"
        first_labels_path.write_text(json.dumps({m.CASE_ID: first}))
        label_path = tmp_path / "evidence-review-label.json"
        label_path.write_text(json.dumps({m.EXTERNAL_CASE_ID: second}))
        result_path = tmp_path / "result.json"
        monkeypatch.setattr(m, "FIRST_LABELS", first_labels_path)
        monkeypatch.setattr(m, "LABEL", label_path)
        monkeypatch.setattr(m, "RESULT", result_path)

        m.join()
        result: dict[str, Any] = json.loads(result_path.read_text())
        return result

    def test_rejects_an_invalid_returned_label_before_touching_anything_else(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        label_path = tmp_path / "evidence-review-label.json"
        label_path.write_text(json.dumps({m.EXTERNAL_CASE_ID: None}))
        monkeypatch.setattr(m, "LABEL", label_path)

        with pytest.raises(m.ValidationError):
            m.join()

    def test_second_opinion_supported_records_disagreement_not_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._run_join(tmp_path, monkeypatch, first="unsupported", second="supported")

        assert result["first_reviewer_label"] == "unsupported"
        assert result["second_adjudicator_label"] == "supported"
        assert result["exact_label_agreement"] is False
        assert "unresolved reviewer disagreement" in result["next_step"]
        assert "Do not erase" in result["next_step"]

    def test_second_opinion_unsupported_calls_for_investigation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._run_join(tmp_path, monkeypatch, first="unsupported", second="unsupported")

        assert result["exact_label_agreement"] is True
        assert "inspect the claim-generation and verification path" in result["next_step"]
        assert "regression test and fix" in result["next_step"]

    def test_second_opinion_uncertain_also_calls_for_investigation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Instruction #4 groups "unsupported" and "uncertain" together on
        the second adjudicator's side -- a plain string-equality check
        against the first label ("unsupported" != "uncertain") would
        wrongly route this to the disagreement branch instead."""
        result = self._run_join(tmp_path, monkeypatch, first="unsupported", second="uncertain")

        assert result["exact_label_agreement"] is False
        assert "inspect the claim-generation and verification path" in result["next_step"]

    def test_join_does_not_mutate_either_label_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first_labels_path = tmp_path / "reviewer-labels.json"
        first_labels_path.write_text(json.dumps({m.CASE_ID: "unsupported"}))
        label_path = tmp_path / "evidence-review-label.json"
        label_path.write_text(json.dumps({m.EXTERNAL_CASE_ID: "uncertain"}))
        monkeypatch.setattr(m, "FIRST_LABELS", first_labels_path)
        monkeypatch.setattr(m, "LABEL", label_path)
        monkeypatch.setattr(m, "RESULT", tmp_path / "result.json")
        before_first = first_labels_path.read_text()
        before_second = label_path.read_text()

        m.join()

        assert first_labels_path.read_text() == before_first
        assert label_path.read_text() == before_second


class TestRealPacketOnDisk:
    """The real, committed packet and blank label, as they ship in the repo."""

    def test_the_committed_packet_does_not_leak(self) -> None:
        packet = json.loads(m.PACKET.read_text())
        m._assert_not_leaking(packet)  # must not raise

    def test_the_committed_packet_uses_the_external_id_not_the_internal_one(self) -> None:
        packet = json.loads(m.PACKET.read_text())
        case_ids = {c["case_id"] for c in packet["cases"]}
        assert case_ids == {m.EXTERNAL_CASE_ID}

    def test_the_real_second_adjudicator_label_is_recorded(self) -> None:
        """A real, independent second adjudicator returned this label --
        locks in the known state so a future silent edit is caught."""
        assert json.loads(m.LABEL.read_text()) == {m.EXTERNAL_CASE_ID: "supported"}

    def test_the_real_join_result_records_disagreement_not_resolution(self) -> None:
        result = json.loads(m.RESULT.read_text())
        assert result["first_reviewer_label"] == "unsupported"
        assert result["second_adjudicator_label"] == "supported"
        assert result["exact_label_agreement"] is False
        assert "unresolved reviewer disagreement" in result["next_step"]

    def test_the_real_first_label_for_this_case_is_unsupported(self) -> None:
        """Locks in the known state this whole investigation is about --
        if a future edit to reviewer-labels.json changes this case's
        label, that is exactly the kind of silent change this test
        exists to catch."""
        first_labels = json.loads(m.FIRST_LABELS.read_text())
        assert first_labels[m.CASE_ID] == "unsupported"
