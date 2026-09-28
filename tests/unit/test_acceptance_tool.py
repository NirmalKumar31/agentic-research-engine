"""The capture tool, checked against a capture that already exists.

The committed v1.1.0 hosted artifact carries both a raw SSE stream and
a metrics file whose reconciliation a person worked out by hand. That
makes it the one thing a derivation tool can be tested against without
a network or a paid run: the tool recomputes those numbers from the
bytes, and they have to agree with what the person wrote.

They agreeing is the interesting part. The summary of an earlier run
got the withheld total wrong by counting only 'unsupported' and
forgetting that partially supported claims are withheld too, so
"a human wrote it" is not evidence of anything on its own.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "examples" / "live-validation" / "tools" / "acceptance.py"
CAPTURE = ROOT / "examples" / "live-validation" / "hosted-20260928-045059"


def load_tool() -> Any:
    """Imported by path, the way the eval harness loads its own.

    A module under examples/ is not an installed package, and giving it
    one would put a deployment tool on the import path of the library
    it is meant to test from the outside.
    """
    spec = importlib.util.spec_from_file_location("acceptance_tool", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> Any:
    return load_tool()


@pytest.fixture(scope="module")
def result(tool: Any) -> dict[str, Any]:
    raw = (CAPTURE / "stream.raw.sse").read_text(encoding="utf-8")
    events = tool.parse_events(raw)
    return next(payload for name, payload in events if name == "result")


class TestItReadsTheWholeStream:
    def test_the_result_payload_survives_parsing(self, result: dict[str, Any]) -> None:
        """The failure this whole file descends from: a capture that
        read the stream and kept 150 characters of each line."""
        assert {"report", "metrics", "verification", "sources"} <= set(result)

    def test_every_event_in_the_capture_is_recovered(self, tool: Any) -> None:
        raw = (CAPTURE / "stream.raw.sse").read_text(encoding="utf-8")
        events = tool.parse_events(raw)
        names = [name for name, _ in events]
        assert "result" in names
        assert "done" in names
        # Not an exact count: what matters is that the parser does not
        # silently drop events it did not recognise.
        assert len(events) == raw.count("event: ")

    def test_a_non_json_data_line_is_kept_as_text(self, tool: Any) -> None:
        events = tool.parse_events("event: ping\ndata: not json\n\n")
        assert events == [("ping", "not json")]


class TestItRecomputesWhatAPersonWorkedOut:
    """Independent arithmetic over the same bytes."""

    @pytest.fixture(scope="class")
    @classmethod
    def committed(cls) -> dict[str, Any]:
        return json.loads((CAPTURE / "metrics.json").read_text())

    def test_the_claim_reconciliation_agrees(
        self, tool: Any, result: dict[str, Any], committed: dict[str, Any]
    ) -> None:
        derived = tool.reconciliation(result)["claims_reconciliation"]
        expected = committed["claims_reconciliation"]
        for key in (
            "generated",
            "checked",
            "not_checked",
            "published",
            "withheld",
            "withheld_unsupported",
            "withheld_partially_supported",
        ):
            assert derived[key] == expected[key], key

    def test_withheld_is_the_sum_of_its_parts_not_a_subtraction(
        self, tool: Any, result: dict[str, Any]
    ) -> None:
        """Otherwise the reconciliation identity holds by construction
        and the test below checks nothing."""
        rec = tool.reconciliation(result)["claims_reconciliation"]
        assert rec["withheld"] == rec["withheld_unsupported"] + rec["withheld_partially_supported"]

    def test_the_identity_holds(self, tool: Any, result: dict[str, Any]) -> None:
        rec = tool.reconciliation(result)["claims_reconciliation"]
        assert rec["published"] + rec["withheld"] + rec["not_checked"] == rec["checked"]

    def test_quote_fidelity_agrees_and_defines_its_denominator(
        self, tool: Any, result: dict[str, Any], committed: dict[str, Any]
    ) -> None:
        derived = tool.reconciliation(result)["quote_fidelity"]
        expected = committed["quote_fidelity"]
        assert derived["denominator_extracted_quotes"] == expected["denominator_extracted_quotes"]
        assert derived["numerator_exact_quotes"] == expected["numerator_exact_quotes"]
        assert derived["rate"] == expected["rate"]
        assert "exact normalised substring" in derived["definition"].lower()

    def test_the_cross_provider_total_is_never_claimed_complete(
        self, tool: Any, result: dict[str, Any]
    ) -> None:
        """Hugging Face uptime and Tavily credits are billed elsewhere
        and this process cannot see them."""
        cost = tool.reconciliation(result)["cost_field_definitions"]
        assert cost["total_cross_provider_cost_complete"] is False
        assert cost["why_not_complete"]


class TestItSeparatesTheDecisions:
    def test_the_four_records_are_produced(self, tool: Any, result: dict[str, Any]) -> None:
        records = tool.verification_records(result)
        assert set(records) == {
            "propositions",
            "relevance-decisions",
            "repairs",
            "withheld-reasons",
        }

    def test_v1_1_0_has_no_contract_era_records(self, tool: Any, result: dict[str, Any]) -> None:
        """Truthful emptiness. This run predates the contract, so the
        three new files are empty and the tool must not invent them."""
        records = tool.verification_records(result)
        assert records["propositions"] == []
        assert records["relevance-decisions"] == []
        assert records["repairs"] == []

    def test_the_withheld_reasons_keep_the_whole_claim(
        self, tool: Any, result: dict[str, Any]
    ) -> None:
        """Read from judgments, not from issues. The issue record
        truncates at 200 characters, which is exactly how four claims
        in the canonical recordings ended mid-sentence."""
        withheld = tool.verification_records(result)["withheld-reasons"]
        assert withheld
        judgments = {j["claim_text"] for j in result["verification"]["judgments"]}
        for entry in withheld:
            assert entry["claim"] in judgments
            assert not entry["claim"].endswith("...")

    def test_a_repaired_claim_is_reported_with_both_wordings(self, tool: Any) -> None:
        """Synthetic, because no committed capture has a repair yet.
        The assertion is about the shape the acceptance artifact needs,
        which has to exist before the run that produces one."""
        synthetic = {
            "verification": {
                "judgments": [
                    {
                        "claim_text": "SMOTE always oversamples the minority class.",
                        "publishable": True,
                        "repair": {
                            "original_text": "SMOTE always oversamples the minority class.",
                            "repaired_text": "SMOTE oversamples the minority class.",
                            "guard": "modality",
                            "accepted": True,
                            "reason": "",
                        },
                    }
                ]
            }
        }
        repairs = tool.verification_records(synthetic)["repairs"]
        assert len(repairs) == 1
        assert repairs[0]["original"] != repairs[0]["rewritten"]
        assert repairs[0]["accepted"] is True


class TestItResolvesCitations:
    def test_a_zero_publication_run_lists_no_claims(
        self, tool: Any, result: dict[str, Any]
    ) -> None:
        """Not a failure of the resolver: nothing published."""
        assert tool.citations(result) == []

    def test_each_claim_resolves_to_a_quote_and_a_url(self, tool: Any) -> None:
        synthetic = {
            "report": {
                "summary_claims": [
                    {"text": "A claim.", "kind": "factual", "evidence_ids": ["S1-e1"]}
                ],
                "key_findings": [],
                "sections": [],
            },
            "evidence": [{"id": "S1-e1", "source_id": "S1", "quote": "q", "quote_match": "exact"}],
            "sources": [{"id": "S1", "url": "https://example.com", "title": "T"}],
        }
        out = tool.citations(synthetic)
        assert out[0]["evidence"][0]["resolved"] is True
        assert out[0]["evidence"][0]["source_url"] == "https://example.com"

    def test_an_unresolvable_id_is_reported_not_hidden(self, tool: Any) -> None:
        """The whole point of the file is that a reader can follow a
        claim to a page. A dangling id has to be visible."""
        synthetic = {
            "report": {
                "summary_claims": [
                    {"text": "A claim.", "kind": "factual", "evidence_ids": ["S9-e9"]}
                ],
                "key_findings": [],
                "sections": [],
            },
            "evidence": [],
            "sources": [],
        }
        entry = tool.citations(synthetic)[0]["evidence"][0]
        assert entry["resolved"] is False
        assert entry["source_url"] is None


class TestItRefusesToCommitASecret:
    @pytest.mark.parametrize(
        ("label", "text"),
        [
            ("openai key", "OPENAI: sk-" + "A" * 32),
            ("tavily key", "key tvly-" + "B" * 32),
            ("hf token", "token hf_" + "C" * 24),
            ("bearer token", "Authorization: Bearer abcdefghijklmnop"),
            ("redis url", "redis://user:password@quota.internal:6379/0"),
            ("hugging face endpoint url", "https://abcdef0123456789xy.endpoints.huggingface.cloud"),
            ("local path", "/Users/someone/project/run.json"),
        ],
    )
    def test_every_forbidden_pattern_is_caught(self, tool: Any, label: str, text: str) -> None:
        found = tool.scan("f.json", text)
        assert found, f"{label} was not detected"
        assert any(label in entry for entry in found)

    def test_clean_text_passes(self, tool: Any) -> None:
        assert tool.scan("f.json", "a report about risk-management and sk-ills") == []

    def test_build_refuses_rather_than_writing_a_leak(
        self, tool: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refusal at build time is recoverable; a credential in a
        commit is not."""
        leaked = {
            "markdown": "see https://abcdef0123456789xy.endpoints.huggingface.cloud",
            "metrics": {},
            "verification": {},
            "sources": [],
            "evidence": [],
            "report": None,
            "contract": None,
        }
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "stream.raw.sse").write_text(
            "event: result\ndata: " + json.dumps(leaked) + "\n\nevent: done\ndata: {}\n\n",
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc:
            tool.run_build(run_dir)
        assert "hugging face endpoint url" in str(exc.value)
        assert not (run_dir / "report.md").exists()


class TestTheToolItselfCarriesNoCredential:
    def test_it_reads_no_key_from_the_environment(self) -> None:
        """The public demo route needs none, which is what makes an
        acceptance run reproducible by a reader."""
        source = TOOL.read_text(encoding="utf-8")
        assert "os.environ" not in source
        assert "getenv" not in source
        assert not re.search(r"\b(sk-|tvly-|hf_)[A-Za-z0-9]{8,}", source)
