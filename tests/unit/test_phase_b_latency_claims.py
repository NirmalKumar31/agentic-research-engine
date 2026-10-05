"""Published latency ranges in README.md and RESULTS.md must match
`raw_results.json` exactly, derived fresh each time rather than
hand-copied -- a hand-copied number is exactly what drifted the first
time (rounding "28.51" down to "28", and letting the 5 timed-out runs'
120.02-122.78s bleed into the "local completed" range as "123").
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _derive_ranges() -> dict[str, tuple[float, float]]:
    raw = json.loads((ROOT / "evaluations/phase_b/raw_results.json").read_text())
    cloud_ok = [r["duration_s"] for r in raw if r["arm"] == "cloud" and r["ok"]]
    local_ok = [r["duration_s"] for r in raw if r["arm"] == "local" and r["ok"]]
    local_timeout = [r["duration_s"] for r in raw if r["arm"] == "local" and r["timed_out"]]
    assert len(cloud_ok) == 24
    assert len(local_ok) == 19
    assert len(local_timeout) == 5
    return {
        "cloud_ok": (min(cloud_ok), max(cloud_ok)),
        "local_ok": (min(local_ok), max(local_ok)),
        "local_timeout": (min(local_timeout), max(local_timeout)),
    }


def _fmt(value: float) -> str:
    return f"{value:.2f}"


class TestLatencyRangesMatchRawData:
    def test_derived_ranges_are_the_known_values(self) -> None:
        # A fixed, human-readable pin on the actual measured data, so a
        # silent change in raw_results.json (it shouldn't ever change --
        # this is historical fact) is itself caught, not just prose drift.
        ranges = _derive_ranges()
        assert ranges["cloud_ok"] == (8.84, 28.51)
        assert ranges["local_ok"] == (54.5, 118.21)
        assert ranges["local_timeout"] == (120.02, 122.78)

    def test_results_md_states_the_derived_cloud_and_local_ranges(self) -> None:
        ranges = _derive_ranges()
        text = (ROOT / "evaluations/phase_b/RESULTS.md").read_text()
        cloud_lo, cloud_hi = (_fmt(v) for v in ranges["cloud_ok"])
        local_lo, local_hi = (_fmt(v) for v in ranges["local_ok"])
        timeout_lo, timeout_hi = (_fmt(v) for v in ranges["local_timeout"])
        assert f"{cloud_lo}-{cloud_hi}s" in text
        assert f"{local_lo}-{local_hi}s" in text
        assert f"{timeout_lo}-{timeout_hi}s" in text

    def test_readme_states_the_derived_cloud_and_local_ranges(self) -> None:
        ranges = _derive_ranges()
        text = (ROOT / "README.md").read_text()
        cloud_lo, cloud_hi = (_fmt(v) for v in ranges["cloud_ok"])
        local_lo, local_hi = (_fmt(v) for v in ranges["local_ok"])
        timeout_lo, timeout_hi = (_fmt(v) for v in ranges["local_timeout"])
        assert f"{cloud_lo}-{cloud_hi}s" in text
        assert f"{local_lo}-{local_hi}s" in text
        assert f"{timeout_lo}-{timeout_hi}s" in text

    def test_no_stale_rounded_ranges_remain(self) -> None:
        """The specific drift this test exists to catch: an integer-
        rounded or timeout-inclusive range sitting next to (or instead
        of) the precise one."""
        for path in (ROOT / "README.md", ROOT / "evaluations/phase_b/RESULTS.md"):
            text = path.read_text()
            assert not re.search(r"\b9-28s\b", text), f"{path}: stale rounded cloud range"
            assert not re.search(r"\b55-123s\b", text), (
                f"{path}: stale local range that bleeds timeout duration into 'completed'"
            )
