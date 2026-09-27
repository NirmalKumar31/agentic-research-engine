"""Truncation is measured, never assumed.

The managed endpoint cuts a pair at 512 tokens and reports nothing. A
premise cut in half is a different premise -- the clause that qualifies
or reverses a claim is usually the one that falls off the end -- so the
count is taken here, and anything that stops it being taken withholds.
"""

from __future__ import annotations

import pytest

from agentic_research.citations import nli_tokens
from agentic_research.citations.nli_tokens import (
    MAX_PAIR_TOKENS,
    TokenizerUnavailable,
    count_pair_tokens,
    load_pair_tokenizer,
    pair_is_truncated,
)


class _Encoding:
    def __init__(self, count: int) -> None:
        self.ids = [0] * count


class _Tokenizer:
    def __init__(self, count: int) -> None:
        self.count = count
        self.calls = 0

    def encode(self, premise: str, hypothesis: str) -> _Encoding:
        self.calls += 1
        return _Encoding(self.count)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    nli_tokens._reset_cache_for_tests()


def test_a_pair_at_the_limit_is_not_truncated() -> None:
    assert pair_is_truncated(_Tokenizer(MAX_PAIR_TOKENS), "p", "h") is False


def test_a_pair_one_token_over_is_truncated() -> None:
    """The boundary is where this decision is actually made."""
    assert pair_is_truncated(_Tokenizer(MAX_PAIR_TOKENS + 1), "p", "h") is True


def test_the_count_includes_special_tokens() -> None:
    assert count_pair_tokens(_Tokenizer(7), "p", "h") == 7


def test_an_encode_failure_withholds() -> None:
    class Broken:
        def encode(self, premise: str, hypothesis: str) -> object:
            raise RuntimeError("boom")

    with pytest.raises(TokenizerUnavailable):
        count_pair_tokens(Broken(), "p", "h")


def test_a_missing_tokenizers_package_withholds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deployed image installs tokenizers deliberately. If that ever
    regresses, runs must stop rather than silently report every premise
    as intact."""
    import builtins

    real_import = builtins.__import__

    def no_tokenizers(name: str, *args: object, **kwargs: object) -> object:
        if name == "tokenizers":
            raise ImportError("no module named tokenizers")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", no_tokenizers)
    with pytest.raises(TokenizerUnavailable, match="not installed"):
        load_pair_tokenizer("some/model", "abc")


def test_a_download_failure_withholds(monkeypatch: pytest.MonkeyPatch) -> None:
    import tokenizers

    def boom(*a: object, **k: object) -> object:
        raise OSError("offline")

    monkeypatch.setattr(tokenizers.Tokenizer, "from_pretrained", staticmethod(boom))
    with pytest.raises(TokenizerUnavailable, match="could not load"):
        load_pair_tokenizer("some/model", "abcdef1234")


def test_the_tokenizer_is_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """8MB over the wire per claim would be its own outage."""
    import tokenizers

    calls = {"n": 0}

    class Stub:
        def no_truncation(self) -> None: ...
        def no_padding(self) -> None: ...

    def counting(*a: object, **k: object) -> Stub:
        calls["n"] += 1
        return Stub()

    monkeypatch.setattr(tokenizers.Tokenizer, "from_pretrained", staticmethod(counting))
    first = load_pair_tokenizer("some/model", "abcdef1234")
    second = load_pair_tokenizer("some/model", "abcdef1234")
    assert first is second
    assert calls["n"] == 1


def test_truncation_is_disabled_on_the_loaded_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tokenizer.json carries a 512 truncation rule. Left on, every
    overlong pair measures exactly 512 and reads as fitting -- which is
    the one answer that must never be produced by guessing."""
    import tokenizers

    seen = {"no_truncation": False, "no_padding": False}

    class Stub:
        def no_truncation(self) -> None:
            seen["no_truncation"] = True

        def no_padding(self) -> None:
            seen["no_padding"] = True

    monkeypatch.setattr(
        tokenizers.Tokenizer, "from_pretrained", staticmethod(lambda *a, **k: Stub())
    )
    load_pair_tokenizer("some/model", "abcdef1234")
    assert seen["no_truncation"]
    assert seen["no_padding"]
