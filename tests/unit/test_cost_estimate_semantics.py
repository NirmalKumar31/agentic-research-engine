"""The pre-dispatch reservation must not be smaller than the charge.

It used to be an estimate of one token per four characters, which is a
reasonable average for English prose and is not a bound. It undercounted
CJK, emoji, dense punctuation, minified code and JSON schemas -- most of
what a research engine sends -- so the dollar reservation came out below
the real cost on exactly the inputs that cost the most.

It is now an upper bound in bytes. Every token in a byte-level BPE
vocabulary decodes to at least one byte, so a sequence of n tokens is at
least n bytes and `tokens <= utf8_bytes` holds for any tokeniser whose
tokens are non-empty.

Exact counting is unavailable rather than unchosen: tiktoken 0.14.0
raises KeyError for this project's configured models. It resolves gpt-4o
to o200k_base and gpt-4 to cl100k_base and knows nothing of gpt-6-luna
or gpt-6-sol, and assuming they share an existing encoding would be the
same class of guess that produced the four-character rule.

A bound on what this process serialises is still not a billing cap, so
the wording tests at the bottom stay.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agentic_research.llm.token_bound import (
    PER_MESSAGE_FRAMING_TOKENS,
    conservative_input_tokens,
    schema_bytes,
)

ROOT = Path(__file__).resolve().parents[2]


class Msg:
    """Stands in for a LangChain message."""

    def __init__(self, content: object) -> None:
        self.content = content


ADVERSARIAL: dict[str, str] = {
    "english_prose": "Retrieval-augmented generation conditions an answer on retrieved passages. "
    * 20,
    "cjk": "研究者は生産性を測定した。" * 80,
    "emoji": "🚀🔬📊🧪" * 200,
    "dense_punctuation": "".join("{}[]():;,.<>/\\|!@#$%^&*" for _ in range(80)),
    "minified_code": "const f=(a,b)=>{return a?.b??c[d]||e;};" * 60,
    "urls": "https://arxiv.org/html/2511.04427v2 " * 60,
    "mixed": "Latency fell 12.5% — 研究 🚀 `a?.b??c` https://x.test/p " * 40,
    "long_quote": "Panel GMM models reveal that accumulated technical debt reduces velocity. " * 50,
}


class TestTheBoundIsNeverBelowTheTrueCount:
    """The property the whole module exists for, checked against a real
    tokeniser on content chosen to break a character-based rule."""

    @pytest.fixture(scope="class")
    @classmethod
    def encoder(cls):
        tiktoken = pytest.importorskip("tiktoken")
        # o200k_base stands in for a tokeniser this project cannot
        # obtain for its configured models. The bound is claimed against
        # any byte-level BPE, so demonstrating it on a real one is the
        # available evidence.
        return tiktoken.get_encoding("o200k_base")

    @pytest.mark.parametrize("name", sorted(ADVERSARIAL))
    def test_the_bound_covers_the_real_token_count(self, name: str, encoder) -> None:
        text = ADVERSARIAL[name]
        actual = len(encoder.encode(text))
        bound = conservative_input_tokens([Msg(text)])
        assert bound >= actual, (
            f"{name}: reserved {bound} for {actual} real tokens -- the "
            "reservation would have been smaller than the charge"
        )

    @pytest.mark.parametrize("name", sorted(ADVERSARIAL))
    def test_the_old_rule_would_have_failed_somewhere(self, name: str, encoder) -> None:
        """Not every sample breaks len//4, but the ones that do are the
        reason it was replaced. Recorded so the motivation stays
        visible rather than becoming folklore."""
        text = ADVERSARIAL[name]
        actual = len(encoder.encode(text))
        old = len(text) // 4
        if name in {"cjk", "emoji", "dense_punctuation", "minified_code"}:
            assert old < actual, f"{name} no longer breaks the old rule"

    def test_a_repair_prompt_is_bounded_too(self, encoder) -> None:
        """A repair carries the failed response and the correction back
        into the prompt, so it is strictly larger than the first try."""
        first = [Msg("system"), Msg("user question")]
        repair = [*first, Msg('{"broken": json'), Msg("Fix the JSON.")]
        assert conservative_input_tokens(repair) > conservative_input_tokens(first)


class TestTheSchemaIsCounted:
    """Structured output sends a JSON schema on every request. Summing
    message content alone under-reserved by the whole schema."""

    SCHEMA = {
        "type": "object",
        "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
        "required": ["claims"],
    }

    def test_including_a_schema_raises_the_bound(self) -> None:
        without = conservative_input_tokens([Msg("hello")])
        with_schema = conservative_input_tokens([Msg("hello")], self.SCHEMA)
        assert with_schema > without
        assert with_schema - without >= len(json.dumps(self.SCHEMA).encode()) - 1

    def test_a_pydantic_model_schema_is_measured(self) -> None:
        from agentic_research.schemas import ExtractionOut

        assert schema_bytes(ExtractionOut) > 0

    def test_no_schema_adds_nothing(self) -> None:
        assert schema_bytes(None) == 0

    def test_an_unserialisable_schema_still_counts(self) -> None:
        """A schema that will not serialise must not silently become
        zero, which is the one answer that breaks the bound."""

        class Opaque:
            pass

        assert schema_bytes(Opaque()) > 0


class TestMessageShapes:
    def test_framing_is_added_per_message(self) -> None:
        one = conservative_input_tokens([Msg("x")])
        two = conservative_input_tokens([Msg("x"), Msg("x")])
        assert two - one >= PER_MESSAGE_FRAMING_TOKENS

    def test_structured_content_is_measured_not_skipped(self) -> None:
        """Multimodal content arrives as a list of parts. Skipping it
        would be a hole in the bound."""
        parts = [{"type": "text", "text": "a" * 500}]
        assert conservative_input_tokens([Msg(parts)]) > 500

    def test_empty_messages_still_reserve_framing(self) -> None:
        assert conservative_input_tokens([Msg("")]) >= PER_MESSAGE_FRAMING_TOKENS

    def test_no_messages_is_not_negative(self) -> None:
        assert conservative_input_tokens([]) >= 0


class TestTheRouterUsesIt:
    def test_the_router_reserves_with_the_bound(self) -> None:
        source = (ROOT / "src/agentic_research/llm/router.py").read_text()
        assert "conservative_input_tokens(messages, schema)" in source
        assert "// 4" not in source, "the character estimate is back"


class TestNothingClaimsABillingCeiling:
    """A bound on what this process serialises is still not a billing
    cap. The provider adds content this side never sees, enforcement is
    not instantaneous, and the account limit remains the backstop."""

    FORBIDDEN = re.compile(
        r"(hard|absolute|guaranteed|billing)\s+(cost\s+)?(ceiling|cap|limit)", re.I
    )
    NEGATED = re.compile(r"(not|never|rather than|isn't|is not|are not|no)\s+(a|an|the)?\s*$", re.I)

    def _tracked_text_files(self) -> list[Path]:
        """Every text file git would carry, staged or not.

        `git ls-files` alone lists the index, so a new document passes
        locally until it is added and fails afterwards.
        """
        import subprocess

        def run(*args: str) -> list[str]:
            return subprocess.run(
                ["git", *args, "*.md", "*.yaml", "*.yml", "*.toml"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()

        seen = dict.fromkeys(run("ls-files") + run("ls-files", "--others", "--exclude-standard"))
        return [ROOT / f for f in seen]

    def test_no_document_calls_the_spend_limit_a_hard_ceiling(self) -> None:
        offenders: list[str] = []
        for path in self._tracked_text_files():
            try:
                text = path.read_text()
            except (OSError, UnicodeDecodeError):  # pragma: no cover
                continue
            for line in text.splitlines():
                if "cost" not in line.lower() and "spend" not in line.lower():
                    continue
                match = self.FORBIDDEN.search(line)
                if match and not self.NEGATED.search(line[: match.start()]):
                    offenders.append(f"{path.relative_to(ROOT)}: {line.strip()[:90]}")
        assert not offenders, "spend limit described as a guarantee:\n" + "\n".join(offenders)

    def test_a_real_claim_would_still_be_caught(self) -> None:
        claim = "MAX_CLOUD_COST_USD is a hard billing cap for every run."
        match = self.FORBIDDEN.search(claim)
        assert match is not None
        assert not self.NEGATED.search(claim[: match.start()])

    def test_the_denial_is_allowed(self) -> None:
        denial = "The per-run limit is an estimate, not a billing cap."
        match = self.FORBIDDEN.search(denial)
        assert match is not None
        assert self.NEGATED.search(denial[: match.start()])


class TestTheAllowanceIsDenominatedInTheBound:
    """Changing the reservation from an estimate to an upper bound
    changed what the configured ceilings mean. A number sized against
    real tokens becomes several times tighter when the thing counted
    over-reserves."""

    def test_the_demo_input_allowance_covers_a_public_run(self) -> None:
        import json

        from agentic_research.web.limits import DemoLimits

        recorded = json.loads(
            (ROOT / "examples/live-validation/20260927-224441-b64016/metrics.json").read_text()
        )
        # The public deployment runs half the recorded run's budget.
        public_real_tokens = recorded["input_tokens"] // 2
        # Worst ratio measured on prose, the dominant content here.
        worst_bound = public_real_tokens * 5.5
        assert DemoLimits().max_cloud_input_tokens >= worst_bound * 1.5, (
            "less than 1.5x headroom over the extrapolated worst case"
        )

    def test_the_dollar_reservation_was_not_widened(self) -> None:
        """Re-denominating a token allowance is not licence to raise
        what a run may spend."""
        from agentic_research.web.limits import DemoLimits

        assert DemoLimits().max_cloud_cost_usd == 0.05
