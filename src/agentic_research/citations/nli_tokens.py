"""Pair token counts without torch.

The hosted endpoint's built-in text-classification handler returns a
label and a score and nothing else. It does not say whether it had to
cut the input to fit, and a premise silently truncated at 512 tokens is
a different premise than the one on the page -- the qualifying clause
that reverses a claim tends to sit at the end.

The local verifier learns this by tokenising twice, once unbounded and
once truncated. The deployed web image cannot: it carries neither torch
nor transformers, deliberately, because the checkpoint does not fit the
plan's memory. So the count is taken here with ``tokenizers`` alone, a
pure-Rust wheel that loads the same ``tokenizer.json`` from the same
pinned commit as the model itself.

Measured against the transformers tokenizer over 263 pairs, including a
sweep straddling the 512-token boundary: 261 exact, 2 higher, 0 lower,
and no disagreement about whether a pair exceeds the limit. The two
high counts are degenerate empty pairs. The direction matters more than
the magnitude: counting high withholds a publishable claim, counting
low would publish a claim checked against half its evidence.
"""

from __future__ import annotations

import threading
from typing import Any

from agentic_research.citations.nli_pin import (
    NLI_DEFAULT_MODEL_ID,
    NLI_DEFAULT_REVISION,
)

# The checkpoint's own limit. Anything longer is cut by the model, here
# or on the remote, whether or not either says so.
MAX_PAIR_TOKENS = 512


class TokenizerUnavailable(RuntimeError):
    """The pinned tokenizer could not be loaded.

    Raised rather than returning a guess. The caller turns this into a
    withhold: not knowing whether a premise was truncated is not the
    same as knowing it was not.
    """


_LOCK = threading.Lock()
_CACHE: dict[tuple[str, str], Any] = {}


def load_pair_tokenizer(
    model_id: str = NLI_DEFAULT_MODEL_ID,
    revision: str = NLI_DEFAULT_REVISION,
) -> Any:
    """Load the pinned tokenizer, once per process.

    Fetched at the pinned commit rather than vendored: ``tokenizer.json``
    is 8.2MB, which is model-sized for a git repository and is the kind
    of file that quietly stops matching the checkpoint it belongs to.
    """
    key = (model_id, revision)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    with _LOCK:
        # Re-checked under the lock: two threads racing the first call
        # would otherwise both pay the download.
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise TokenizerUnavailable(f"tokenizers is not installed: {exc}") from exc
        try:
            tokenizer = Tokenizer.from_pretrained(model_id, revision=revision)
        except Exception as exc:
            raise TokenizerUnavailable(
                f"could not load {model_id}@{revision[:8]}: {type(exc).__name__}"
            ) from exc

        # tokenizer.json carries a truncation rule baked in at 512. Left
        # enabled it caps every count at exactly the limit, so an
        # overlong pair reports 512 and reads as "fits". Measured: 45 of
        # 69 pairs were wrong this way before these two calls.
        tokenizer.no_truncation()
        tokenizer.no_padding()
        _CACHE[key] = tokenizer
        return tokenizer


def count_pair_tokens(tokenizer: Any, premise: str, hypothesis: str) -> int:
    """Length of the encoded pair, special tokens included."""
    try:
        return len(tokenizer.encode(premise, hypothesis).ids)
    except Exception as exc:
        raise TokenizerUnavailable(f"could not encode pair: {type(exc).__name__}") from exc


def pair_is_truncated(
    tokenizer: Any,
    premise: str,
    hypothesis: str,
    max_length: int = MAX_PAIR_TOKENS,
) -> bool:
    """Whether the model had to cut this pair to score it."""
    return count_pair_tokens(tokenizer, premise, hypothesis) > max_length


def _reset_cache_for_tests() -> None:
    with _LOCK:
        _CACHE.clear()
