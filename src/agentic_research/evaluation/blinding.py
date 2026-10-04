"""Strip provider identity from an arm's output before a human reviews it.

A reviewer who can see "gpt-6-luna" or "qwen3:4b" is not blind, whatever
the rest of the protocol does -- and a model's own report sometimes names
itself (synthesis prompts have produced "As an AI language model..." and
similar before). This is the one mechanical guarantee this module makes:
every string a known model id or provider name could appear as is
replaced with an opaque label, consistently, so two blinded outputs from
the same arm still look like the same arm to a reviewer without ever
saying which one it is.

What this does not do: catch a model describing its own architecture in
words that are not its id ("I am a large model trained by..."). That is
a content problem, not an identity-string problem, and no regex catches
it reliably. Flagged as a known gap in ``docs/BENCHMARK-PROTOCOL.md``
rather than quietly assumed solved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Provider and model identifiers this project actually configures. A
# closed list, not an attempt at every model name that exists -- the
# benchmark only ever runs the models named in its own manifest, so this
# only has to cover those.
_KNOWN_IDENTIFIERS = (
    "openai",
    "ollama",
    "gpt-6-luna",
    "gpt-6-astra",
    "qwen3:4b",
    "qwen3",
    "claude",
    "anthropic",
)

_IDENTIFIER_PATTERN = re.compile(
    "|".join(re.escape(name) for name in sorted(_KNOWN_IDENTIFIERS, key=len, reverse=True)),
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BlindedArm:
    """One arm's output, with its identity replaced by an opaque label.

    ``label`` is assigned by the caller (e.g. "Arm A", "Arm B"), not
    derived from anything about the arm itself -- deriving it from, say,
    alphabetical order of the real labels would leak information through
    the assignment itself if a reviewer ever saw two benchmarks.
    """

    label: str
    markdown: str
    redaction_count: int


def blind(label: str, markdown: str, extra_identifiers: tuple[str, ...] = ()) -> BlindedArm:
    """Replace every known model/provider identifier in `markdown` with
    the opaque label, and report how many replacements were made.

    The count matters as much as the redaction: a benchmark run where
    zero replacements happened across every question is worth
    investigating, not assuming clean -- either the output never
    mentioned a model, or the identifier list is missing one that
    appeared.
    """
    pattern = _IDENTIFIER_PATTERN
    if extra_identifiers:
        names = sorted({*_KNOWN_IDENTIFIERS, *extra_identifiers}, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(n) for n in names), re.IGNORECASE)
    redacted, count = pattern.subn(f"[{label}]", markdown)
    return BlindedArm(label=label, markdown=redacted, redaction_count=count)


def contains_no_identifier(markdown: str, extra_identifiers: tuple[str, ...] = ()) -> bool:
    """Whether blinded text is actually free of every known identifier --
    the check a test or a reviewing tool runs *after* calling `blind`,
    rather than trusting the substitution happened correctly."""
    names = {*_KNOWN_IDENTIFIERS, *extra_identifiers}
    lowered = markdown.lower()
    return not any(name.lower() in lowered for name in names)
