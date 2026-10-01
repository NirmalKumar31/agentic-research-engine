"""The telemetry the page renders must actually reach the page.

This repo's signature defect is a value computed correctly and never
handed to the thing that needed it -- eleven instances, the last found by
a paid run after a test had asserted only the other call site. The run
telemetry panel reads `metrics.environment` and `metrics.stage_seconds`;
if either stopped being serialised the panel would quietly render a
shorter, duller version of itself and no existing test would notice.

So this asserts the payload, not the metrics object.
"""

from __future__ import annotations

import json

from test_web_api import sample_result

from agentic_research.environment import capture as capture_environment
from agentic_research.web.recordings import serialise_result


def payload() -> dict:
    return serialise_result(sample_result())


class TestThePanelsInputsAreSerialised:
    def test_metrics_reach_the_client(self) -> None:
        assert "metrics" in payload()

    def test_stage_timings_are_a_serialised_field(self) -> None:
        """Per-stage timing is the one thing the panel cannot derive."""
        assert "stage_seconds" in payload()["metrics"]

    def test_the_environment_block_is_a_serialised_field(self) -> None:
        """Temperature, budgets and build provenance all live here."""
        assert "environment" in payload()["metrics"]

    def test_the_payload_is_json_serialisable(self) -> None:
        """`model_dump(mode="json")` has to stay json-mode."""
        json.dumps(payload())

    def test_the_fields_the_panel_groups_are_present(self) -> None:
        metrics = payload()["metrics"]
        for field in (
            "duration_s",
            "llm_calls",
            "provider_requests",
            "input_tokens",
            "output_tokens",
            "known_cost_usd",
            "cost_is_complete",
            "calls_by_role",
            "model_assignments",
            "search_queries",
            "evidence_items",
            "support_breakdown",
            "errors",
            "error_kinds",
        ):
            assert field in metrics, field


class TestTheEnvironmentSnapshotCarriesNoCredential:
    """The panel's safety rests on this being an allowlist.

    `environment.capture` reads no environment variables at all, so there
    is no path for a key to enter. Asserted here rather than trusted,
    because this block is now rendered on a public page.
    """

    def test_capture_reads_no_environment_variables(self) -> None:
        from pathlib import Path

        source = Path("src/agentic_research/environment.py").read_text()
        # `_cpu_count` imports os for cpu_count() only; no env access.
        assert "os.environ" not in source
        assert "getenv" not in source

    def test_no_secret_shaped_key_or_value_in_a_real_snapshot(self) -> None:
        blob = json.dumps(capture_environment()).lower()
        for marker in ("api_key", "apikey", "secret", "password", "token", "bearer", "sk-"):
            assert marker not in blob, marker

    def test_host_detail_is_not_what_the_panel_shows(self) -> None:
        """The snapshot records platform and cpu count; the panel omits them.

        Pinned so that "surface everything in environment" is a decision
        someone has to make again rather than drift into.
        """
        from pathlib import Path

        panel = Path("web/src/telemetry.ts").read_text()
        for field in ("platform", "processor", "cpu_count", "machine"):
            assert field not in panel, field
