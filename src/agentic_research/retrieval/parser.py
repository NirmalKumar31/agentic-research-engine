"""HTML to readable text.

Uses trafilatura, which strips navigation, cookie banners, comment sections and
boilerplate. That matters for cost as much as quality: feeding a model a full
page of chrome costs tokens on every extraction call and measurably dilutes
what the model attends to.
"""

from __future__ import annotations

import re

import trafilatura

from agentic_research.observability import get_logger

log = get_logger(__name__)

_WHITESPACE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def truncate(text: str, max_chars: int) -> str:
    """Trim to a budget, preferring a paragraph then a sentence boundary.

    Cutting mid-sentence produces fragments that models then quote verbatim,
    and a truncated quote fails the verbatim check against the source later.
    """
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    for boundary in ("\n\n", ". "):
        cut = window.rfind(boundary)
        if cut > max_chars * 0.6:
            return window[: cut + len(boundary)].strip()
    return window.strip()


def extract_main_text(html: str, url: str = "") -> str:
    """Pull the main body text out of an HTML document.

    Returns an empty string when there is nothing worth reading, which the
    caller treats as a failed source rather than as empty evidence.
    """
    if not html or not html.strip():
        return ""
    try:
        extracted = trafilatura.extract(
            html,
            url=url or None,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
            no_fallback=False,
        )
    except Exception as exc:
        log.debug("extract_failed", url=url[:120], error=str(exc)[:200])
        return ""
    return clean_text(extracted) if extracted else ""


def markdown_to_text(markdown: str) -> str:
    """Flatten provider-supplied markdown into plain text.

    Search providers can return page content as markdown. Stripping the syntax
    keeps quotes comparable with text we extracted ourselves, so the verbatim
    check behaves the same regardless of where the content came from.
    """
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links -> label
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)  # quotes
    text = re.sub(r"```[^\n]*\n(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(\*\*|__|\*|_|`)", "", text)  # emphasis
    return clean_text(text)


def word_count(text: str) -> int:
    return len(text.split())
