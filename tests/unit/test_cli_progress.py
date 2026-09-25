"""The CLI progress printer, fed the events the graph really emits.

Every handler here previously went untested: the suite drives
``run_research`` directly and never renders progress, so
``_on_citations_verified`` read a key the node has never emitted and
raised KeyError at the end of every single CLI run. These tests feed
the printer each event the graph produces, so a key that does not exist
fails here instead of in a user's terminal.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from agentic_research.cli.progress import ProgressPrinter


@pytest.fixture
def printer() -> ProgressPrinter:
    return ProgressPrinter(Console(quiet=True))


def test_the_verification_event_the_node_emits_renders(printer: ProgressPrinter) -> None:
    """Exactly the payload built in reporting.verify_citations."""
    printer.handle(
        {
            "event": "citations_verified",
            "total": 4,
            "evidence_integrity": 1.0,
            "support": {"supported": 1, "partially_supported": 0, "unsupported": 3},
            "exhaustive": True,
            "not_checked": 0,
            "removed": 3,
            "published": 1,
        }
    )


def test_a_verification_event_missing_optional_keys_still_renders(
    printer: ProgressPrinter,
) -> None:
    printer.handle({"event": "citations_verified", "total": 0})


def test_an_unknown_event_is_ignored(printer: ProgressPrinter) -> None:
    printer.handle({"event": "something_new", "detail": 1})


@pytest.mark.parametrize(
    "event",
    [
        {"event": "synthesizing", "evidence": 6},
        {"event": "verifying_citations"},
        {"event": "completed", "stop_reason": "stopped after round 1"},
        {"event": "followups_generated", "count": 2, "questions": ["a", "b"]},
    ],
)
def test_reporting_events_render(printer: ProgressPrinter, event: dict) -> None:
    printer.handle(event)
