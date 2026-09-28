"""A pre-dispatch input size the true token count cannot exceed.

The reservation used to estimate input at one token per four
characters. That is a good average for English prose and it is not a
bound: CJK, emoji, dense punctuation, minified code and JSON schemas
all run above a quarter token per character, so the estimate landed
*under* the real count on exactly the content a research engine
handles, and the dollar reservation was smaller than the charge.

Exact counting is not available. ``tiktoken`` 0.14.0 raises ``KeyError``
for this project's configured models -- it resolves ``gpt-4o`` to
``o200k_base`` and ``gpt-4`` to ``cl100k_base``, and knows nothing about
``gpt-6-luna`` or ``gpt-6-sol``. Assuming they share an existing
encoding would be a guess about a vendor's format, which is the class of
assumption that produced the four-characters rule.

So the bound is bytes.

**Why it holds.** Every token in a byte-level BPE vocabulary decodes to
at least one byte, so a sequence of *n* tokens decodes to at least *n*
bytes, so ``tokens <= utf8_bytes``. The property needed is only that no
token decodes to an empty string, which is true of any practical
tokeniser. Special tokens are the exception, and they come from chat
framing rather than from the content, which is why framing is counted
separately below rather than assumed away.

**What it costs.** Measured against ``o200k_base`` on representative
content: 5.47 bytes/token for English prose, 5.84 for an evidence
quote, 3.27 for a JSON schema, 3.55 for CJK, 2.57 for URLs, 2.05 for
minified code, 1.78 for emoji. The bound therefore over-reserves by
roughly 2x in the worst case and 5-6x on ordinary prose. That is the
price of a number that cannot be too small, and the configured
allowances are denominated in it -- see ``max_cloud_input_tokens``.
"""

from __future__ import annotations

import json
from typing import Any

# Per-message chat framing: role, separators and the structural tokens
# the provider adds around content this process never sees. OpenAI has
# documented 3-4 for its chat formats; 8 is deliberately above that,
# because the whole point of this module is a number that cannot be too
# small, and eight tokens per message is negligible beside the content.
PER_MESSAGE_FRAMING_TOKENS = 8

# Tokens the provider adds once per request to prime the reply.
REPLY_PRIMING_TOKENS = 8


def _content_bytes(message: Any) -> int:
    """UTF-8 length of one message's content.

    Structured content (a list of parts, as multimodal messages use) is
    serialised rather than skipped: an unmeasured part would be a hole
    in the bound.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    try:
        return len(json.dumps(content, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return len(str(content).encode("utf-8"))


def schema_bytes(schema: Any) -> int:
    """Size of the structured-output schema sent with the request.

    Counted because it is sent. The previous estimate summed message
    content only, so every request under-reserved by the whole schema --
    which for this engine's extraction and synthesis schemas is not a
    rounding error.
    """
    if schema is None:
        return 0
    if isinstance(schema, str):
        return len(schema.encode("utf-8"))
    for attr in ("model_json_schema", "schema"):
        getter = getattr(schema, attr, None)
        if callable(getter):
            try:
                return len(json.dumps(getter(), default=str).encode("utf-8"))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                break
    try:
        return len(json.dumps(schema, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return len(str(schema).encode("utf-8"))


def conservative_input_tokens(messages: list[Any], schema: Any = None) -> int:
    """An input size the real token count cannot exceed.

    Not an estimate of the count, and deliberately not close to it. It
    is the number the spend reservation is taken against, so being wrong
    low is a billing error and being wrong high is only a smaller run.
    """
    total = sum(_content_bytes(m) for m in messages)
    total += schema_bytes(schema)
    total += PER_MESSAGE_FRAMING_TOKENS * len(messages)
    total += REPLY_PRIMING_TOKENS
    return total
