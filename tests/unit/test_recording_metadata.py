"""Recording metadata must agree with the recording.

The site advertised "PDF page provenance" on a run containing zero
page-numbered citations, for one reason: the badge was an assertion
keyed by recording id, and nothing recomputed it. A reader cannot tell
a stale badge from a true one, so the disagreement has to fail here.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agentic_research.web.recordings import (
    RecordingSummary,
    available,
    load,
    page_citation_count,
    recompute_summary_fields,
)

RECORDINGS = (
    Path(__file__).resolve().parents[2] / "src" / "agentic_research" / "web" / "recorded_runs"
)


def payloads() -> list[tuple[str, dict]]:
    return [(p.stem, json.loads(p.read_text())) for p in sorted(RECORDINGS.glob("*.json"))]


class TestPageProvenanceIsCounted:
    @pytest.mark.parametrize(
        ("name", "payload"), payloads(), ids=lambda v: v if isinstance(v, str) else ""
    )
    def test_count_matches_the_evidence(self, name: str, payload: dict) -> None:
        evidence = payload["result"]["evidence"]
        expected = sum(1 for e in evidence if e.get("citable") and e.get("page") is not None)
        assert page_citation_count(payload) == expected

    def test_a_page_on_uncitable_evidence_does_not_count(self) -> None:
        """A page number on a quote that could not be matched to its
        source grounds nothing, so it is not page provenance."""
        payload = {"result": {"evidence": [{"citable": False, "page": 7}]}}
        assert page_citation_count(payload) == 0

    def test_a_citable_quote_without_a_page_does_not_count(self) -> None:
        payload = {"result": {"evidence": [{"citable": True, "page": None}]}}
        assert page_citation_count(payload) == 0

    def test_a_citable_quote_with_a_page_counts(self) -> None:
        payload = {"result": {"evidence": [{"citable": True, "page": 7}]}}
        assert page_citation_count(payload) == 1


class TestSummariesAgreeWithPayloads:
    def test_every_recording_summary_is_recomputable(self) -> None:
        by_id = {s.id: s for s in available()}
        for name, payload in payloads():
            summary: RecordingSummary = by_id[name]
            expected = recompute_summary_fields(payload)
            actual = {
                "sources": summary.sources,
                "evidence_items": summary.evidence_items,
                "citable_evidence": summary.citable_evidence,
                "page_citation_count": summary.page_citation_count,
                "published_claims": summary.published_claims,
            }
            assert actual == expected, f"{name}: stored summary disagrees with its payload"

    def test_has_page_provenance_tracks_the_count(self) -> None:
        for summary in available():
            assert summary.has_page_provenance == (summary.page_citation_count > 0)


class TestTamperedMetadataIsDetected:
    """Deliberate disagreement must be caught, or the check proves nothing."""

    def base(self) -> dict:
        _, payload = payloads()[0]
        return copy.deepcopy(payload)

    def test_dropping_a_page_changes_the_recomputed_count(self) -> None:
        payload = self.base()
        before = recompute_summary_fields(payload)
        for item in payload["result"]["evidence"]:
            item["page"] = None
        after = recompute_summary_fields(payload)
        assert after["page_citation_count"] == 0
        if before["page_citation_count"] > 0:
            assert after != before

    def test_adding_a_page_changes_the_recomputed_count(self) -> None:
        payload = self.base()
        before = recompute_summary_fields(payload)["page_citation_count"]
        citable = [e for e in payload["result"]["evidence"] if e.get("citable")]
        assert citable, "fixture needs at least one citable item"
        citable[0]["page"] = 99
        after = recompute_summary_fields(payload)["page_citation_count"]
        assert after >= before
        assert after > 0

    def test_removing_evidence_changes_the_counts(self) -> None:
        payload = self.base()
        before = recompute_summary_fields(payload)
        payload["result"]["evidence"] = payload["result"]["evidence"][:1]
        assert recompute_summary_fields(payload) != before

    def test_a_forged_published_count_disagrees_with_the_payload(self) -> None:
        payload = self.base()
        real = recompute_summary_fields(payload)["published_claims"]
        payload["result"]["verification"]["final_published_claims"] = real + 100
        assert recompute_summary_fields(payload)["published_claims"] != real


class TestDescriptionsDoNotAssertRetrievalOutcomes:
    """A description is written before the run; retrieval is not.

    "Includes page-aware evidence from the official PDF" was true of one
    recording and false of its replacement, because whether a PDF is
    reached depends on what search returns that day. Stable descriptions
    say what is researched; derived metadata says what was found.
    """

    FORBIDDEN = ("page-aware", "page-level", "carries the page", "with a page number")

    @pytest.mark.parametrize(
        ("name", "payload"), payloads(), ids=lambda v: v if isinstance(v, str) else ""
    )
    def test_description_makes_no_page_promise(self, name: str, payload: dict) -> None:
        description = (payload["meta"].get("description") or "").lower()
        offending = [p for p in self.FORBIDDEN if p in description]
        assert not offending, (
            f"{name}: description asserts {offending}, which retrieval does not guarantee; "
            "let the derived page_citation_count say what was actually found"
        )

    @pytest.mark.parametrize(
        ("name", "payload"), payloads(), ids=lambda v: v if isinstance(v, str) else ""
    )
    def test_recording_loads(self, name: str, payload: dict) -> None:
        assert load(name) is not None


class TestRecordingsCarryTheirProvenance:
    """§Y: a public artifact has to say what produced it.

    Live search is nondeterministic, so a recording cannot be
    reproduced by re-running it. What it can do is state exactly what
    it was: which commit, which models, which threshold, when.
    """

    @pytest.mark.parametrize(
        ("name", "payload"), payloads(), ids=lambda v: v if isinstance(v, str) else ""
    )
    def test_commit_and_timestamp_are_recorded(self, name: str, payload: dict) -> None:
        meta = payload["meta"]
        assert meta.get("recorded_at"), f"{name}: no timestamp"
        provenance = meta.get("provenance") or {}
        assert provenance.get("commit"), f"{name}: no commit"
        assert provenance.get("dirty") is False, f"{name}: recorded from a dirty tree"

    @pytest.mark.parametrize(
        ("name", "payload"), payloads(), ids=lambda v: v if isinstance(v, str) else ""
    )
    def test_configuration_fingerprints_are_recorded(self, name: str, payload: dict) -> None:
        provenance = payload["meta"]["provenance"]
        for field in ("config_fingerprint", "prompt_version", "schema_version", "engine_version"):
            assert provenance.get(field), f"{name}: missing {field}"

    @pytest.mark.parametrize(
        ("name", "payload"), payloads(), ids=lambda v: v if isinstance(v, str) else ""
    )
    def test_the_verifier_identity_is_on_every_judgment(self, name: str, payload: dict) -> None:
        """Which classifier, which revision, which threshold. A verdict
        without these cannot be re-derived."""
        judgments = (payload["result"].get("verification") or {}).get("judgments") or []
        assert judgments, f"{name}: no judgments recorded"
        for judgment in judgments:
            assert judgment.get("model_id"), f"{name}: judgment without a model id"
            assert judgment.get("model_revision"), f"{name}: judgment without a revision"
            assert judgment.get("support_threshold") is not None, f"{name}: no threshold"

    def test_limitations_states_that_search_is_nondeterministic(self) -> None:
        """The artifact must not imply another run recovers these
        sources, because five passes of the same questions did not."""
        text = Path("docs/LIMITATIONS.md").read_text().lower()
        assert "live search returns different results" in text
        assert "snapshot" in text
