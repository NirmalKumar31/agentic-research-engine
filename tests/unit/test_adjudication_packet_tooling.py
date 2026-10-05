"""The 6-case blinded adjudication packet: the leak scanner, the fixed
dispute list staying in sync with the real reconciliation data, the
built packet's actual blindness, the automatic handoff bundle, the
key's permissions, and the refusal to overwrite an existing (possibly
in-progress or completed) packet.

No adjudication happens here, and none of these tests write a decision
into anything -- `scripts/prepare_adjudication_packet.py` only builds
the blank packet and key; a human fills in the decisions separately.
"""

from __future__ import annotations

import csv
import json
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_adjudication_packet import (  # noqa: E402
    DISPUTES,
    HANDOFF_FILES,
    DestinationExistsError,
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
        path = ROOT / "evaluations/phase_b/review/reconciliation.csv"
        rows = list(csv.DictReader(path.open()))
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

        dest = tmp_path / "adjudication"
        monkeypatch.setattr(m, "ADJ", dest)
        build()

        key = json.loads((dest / "adjudication_key.json").read_text())
        assert len(key) == 6
        assert len(set(key)) == 6  # unique random IDs

        with (dest / "adjudication_template.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6
        assert {r["blinded_id"] for r in rows} == set(key)
        for row in rows:
            assert row["final_decision"] == ""
            assert row["rationale"] == ""

        packet_text = (dest / "packet.md").read_text()
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

        dest = tmp_path / "adjudication"
        monkeypatch.setattr(m, "ADJ", dest)
        monkeypatch.setattr(
            m, "read_blinded_markdown", lambda qid, rep, arm: "mentions openai directly"
        )
        with pytest.raises(LeakError):
            build()
        # A leak found while assembling must not leave a partial result
        # sitting at the real destination.
        assert not dest.exists()


class TestHandoffBundle:
    def test_handoff_contains_exactly_the_three_expected_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import prepare_adjudication_packet as m

        dest = tmp_path / "adjudication"
        monkeypatch.setattr(m, "ADJ", dest)
        build()

        handoff_contents = sorted(p.name for p in (dest / "handoff").iterdir())
        assert handoff_contents == sorted(HANDOFF_FILES)

    def test_handoff_excludes_the_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import prepare_adjudication_packet as m

        dest = tmp_path / "adjudication"
        monkeypatch.setattr(m, "ADJ", dest)
        build()

        assert not (dest / "handoff" / "adjudication_key.json").exists()

    def test_handoff_copies_match_the_real_packet_and_template_byte_for_byte(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import prepare_adjudication_packet as m

        dest = tmp_path / "adjudication"
        monkeypatch.setattr(m, "ADJ", dest)
        build()

        for name in HANDOFF_FILES:
            assert (dest / "handoff" / name).read_text() == (dest / name).read_text()

    def test_handoff_readme_instructs_against_inspecting_the_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import prepare_adjudication_packet as m

        dest = tmp_path / "adjudication"
        monkeypatch.setattr(m, "ADJ", dest)
        build()

        readme = (dest / "handoff" / "README.md").read_text().lower()
        assert "do not" in readme
        assert "search" in readme or "browse" in readme
        assert "repository" in readme


class TestKeyPermissions:
    def test_key_file_is_mode_0600(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import prepare_adjudication_packet as m

        dest = tmp_path / "adjudication"
        monkeypatch.setattr(m, "ADJ", dest)
        build()

        mode = stat.S_IMODE((dest / "adjudication_key.json").stat().st_mode)
        assert mode == 0o600


class TestRefusesToOverwrite:
    def test_refuses_when_destination_already_has_a_packet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import prepare_adjudication_packet as m

        dest = tmp_path / "adjudication"
        monkeypatch.setattr(m, "ADJ", dest)
        build()

        with pytest.raises(DestinationExistsError):
            build()

    def test_an_existing_completed_template_is_preserved_on_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import prepare_adjudication_packet as m

        dest = tmp_path / "adjudication"
        monkeypatch.setattr(m, "ADJ", dest)
        build()

        completed = (dest / "adjudication_template.csv").read_text()
        rows = list(csv.DictReader(completed.splitlines()))
        fieldnames = list(rows[0].keys())
        for row in rows:
            row["final_decision"] = "3"
            row["rationale"] = "a human's actual reasoning"
            row["confidence"] = "High"
            row["abstain_insufficient_evidence"] = "No"
        import io

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        (dest / "adjudication_template.csv").write_text(buf.getvalue())

        with pytest.raises(DestinationExistsError):
            build()

        # Compare content, not raw bytes: csv.writer emits "\r\n", while
        # Path.read_text() applies universal-newline translation on the
        # way back in -- a difference in how the two sides of this test
        # read the file, not something the script under test produced.
        after = (dest / "adjudication_template.csv").read_text().replace("\r\n", "\n")
        before = buf.getvalue().replace("\r\n", "\n")
        assert after == before

    def test_an_empty_preexisting_destination_is_not_a_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import prepare_adjudication_packet as m

        dest = tmp_path / "adjudication"
        dest.mkdir()
        monkeypatch.setattr(m, "ADJ", dest)
        build()  # must not raise: an empty pre-existing directory is fine

        assert (dest / "packet.md").exists()
