"""The 6-case blinded adjudication packet: the leak scanner, the fixed
dispute list staying in sync with the real reconciliation data, and the
built packet's actual blindness (no arm/model/provider/rep anywhere).

No adjudication happens here, and none of these tests write a decision
into anything -- `scripts/prepare_adjudication_packet.py` only builds
the blank packet and key; a human fills in the decisions separately.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_adjudication_packet import (  # noqa: E402
    DISPUTES,
    LeakError,
    build,
    scan_for_leaks,
)


class TestScanForLeaks:
    def test_passes_clean_text(self) -> None:
        scan_for_leaks("A clean candidate report with no identifiers.", context="test")

    def test_catches_a_known_provider_identifier(self) -> None:
        with pytest.raises(LeakError):
            scan_for_leaks("Served via the openai endpoint.", context="test")

    def test_catches_the_bracketed_arm_leak(self) -> None:
        with pytest.raises(LeakError):
            scan_for_leaks("Routed to [CLOUD].", context="test")

    def test_catches_the_redaction_count_side_channel(self) -> None:
        with pytest.raises(LeakError):
            scan_for_leaks("<!-- redactions: 2 -->\n\nSome text.", context="test")


class TestDisputesMatchRealReconciliation:
    """The 6 hardcoded disputes must never silently drift from what the
    actual, committed reconciliation data disagrees on -- this is the
    one regression that would make the packet adjudicate the wrong
    thing, or an outdated thing, without anyone noticing."""

    def test_exactly_six_disputes(self) -> None:
        assert len(DISPUTES) == 6

    def test_disputes_match_reconciliation_csv_exactly(self) -> None:
        rows = list(csv.DictReader((ROOT / "evaluations/phase_b/review/reconciliation.csv").open()))
        score_disputes = {
            (r["question_id"], int(r["repetition"]), r["arm"])
            for r in rows
            if r.get("disagreement_claim_support_1to5") == "True"
        }
        harm_disputes = {
            (r["question_id"], int(r["repetition"]), r["arm"])
            for r in rows
            if r.get("harmful_disagreement") == "True"
        }

        coded_score = {
            (qid, rep, arm) for qid, rep, arm, kind, _ in DISPUTES if kind == "score_dimension"
        }
        coded_harm = {
            (qid, rep, arm) for qid, rep, arm, kind, _ in DISPUTES if kind == "harmful_claim_flag"
        }

        assert coded_score == score_disputes
        assert coded_harm == harm_disputes


class TestBuildProducesABlindPacket:
    def test_build_writes_six_cases_with_unique_ids_and_clean_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import prepare_adjudication_packet as m

        monkeypatch.setattr(m, "ADJ", tmp_path)
        build()

        key = json.loads((tmp_path / "adjudication_key.json").read_text())
        assert len(key) == 6
        assert len(set(key)) == 6  # unique random IDs

        with (tmp_path / "adjudication_template.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6
        assert {r["blinded_id"] for r in rows} == set(key)
        for row in rows:
            assert row["final_decision"] == ""
            assert row["rationale"] == ""

        packet_text = (tmp_path / "packet.md").read_text()
        for leak in ("openai", "ollama", "gpt-6", "qwen3", "[local]", "[cloud]", "redactions:"):
            assert leak not in packet_text.lower()
        # Repetition number is explicitly excluded from the packet per the
        # adjudication brief, even though it's recorded in the private key.
        for mapping in key.values():
            assert f"rep{mapping['repetition']}" not in packet_text

    def test_build_refuses_a_leaking_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import prepare_adjudication_packet as m

        monkeypatch.setattr(m, "ADJ", tmp_path)
        monkeypatch.setattr(
            m, "read_blinded_markdown", lambda qid, rep, arm: "mentions openai directly"
        )
        with pytest.raises(LeakError):
            build()
