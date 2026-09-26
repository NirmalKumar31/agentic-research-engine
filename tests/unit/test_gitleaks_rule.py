"""The secret-scanner regex must not have been weakened.

The `sk-` pattern was anchored at a word boundary to stop a source
page slugged "ai-ri|sk-management-framework-..." from reporting a leak
inside a committed recording. That is the kind of change that quietly
trades detection for a green tree, so both directions are pinned here:
the false positive stays gone, and every shape a real key takes still
matches.

This is a local override of the project's own rule, not an upstream
gitleaks rule, so the regex is read from .gitleaks.toml rather than
duplicated.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parents[2] / ".gitleaks.toml"


def openai_rule() -> re.Pattern[str]:
    rules = tomllib.loads(CONFIG.read_text())["rules"]
    rule = next(r for r in rules if r["id"] == "openai-api-key-broad")
    return re.compile(rule["regex"])


KEY_BODY = "A" * 48


@pytest.mark.parametrize(
    "text",
    [
        f"sk-{KEY_BODY}",
        f"OPENAI_API_KEY=sk-{KEY_BODY}",
        f'"api_key": "sk-{KEY_BODY}"',
        f"export OPENAI_API_KEY='sk-{KEY_BODY}'",
        f"  sk-{KEY_BODY}",
        f"Authorization: Bearer sk-{KEY_BODY}",
        f"https://example.test/?key=sk-{KEY_BODY}",
        "sk-proj-" + "aB3_-" * 10,
        "sk-svcacct-" + "x" * 40,
        f"key = sk-{KEY_BODY}  # inline comment",
    ],
)
def test_real_key_shapes_still_match(text: str) -> None:
    assert openai_rule().search(text), f"scanner would miss: {text[:40]}"


@pytest.mark.parametrize(
    "text",
    [
        "ai-risk-management-framework-practical-study-notes-for-implementation",
        "the risk-management-framework-and-its-four-core-functions-explained",
        "https://example.test/ai-risk-management-framework-overview-and-guide",
        "brisk-marketing-automation-platform-comparison-guide-2026",
    ],
)
def test_word_slugs_no_longer_match(text: str) -> None:
    """`sk-` mid-word is not a key. These all tripped the scanner
    before the anchor and none is a secret."""
    assert not openai_rule().search(text)


def test_the_rule_is_still_anchored() -> None:
    """Guards the guard: removing the anchor would restore the false
    positives, and the tests above would be the only thing to notice."""
    rules = tomllib.loads(CONFIG.read_text())["rules"]
    rule = next(r for r in rules if r["id"] == "openai-api-key-broad")
    assert rule["regex"].startswith("\\b"), "the word-boundary anchor was removed"


def test_the_body_length_floor_is_unchanged() -> None:
    """Shortening this would trade precision for noise; lengthening it
    would miss shorter legacy keys."""
    rules = tomllib.loads(CONFIG.read_text())["rules"]
    rule = next(r for r in rules if r["id"] == "openai-api-key-broad")
    assert "{32,}" in rule["regex"]
