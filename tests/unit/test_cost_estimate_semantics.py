"""What the spend limit is, and what it is not.

The pre-dispatch reservation approximates input tokens from character
length. That is fine as an approximation and wrong as a guarantee: the
comment above it used to say the check "only has to be conservative",
which is the one property four-characters-per-token does not have. It
undershoots on exactly the content a research engine handles -- quoted
code, dense punctuation, non-Latin scripts, JSON schemas.

These tests pin the honest description rather than a false bound. They
do not assert a token count for a model whose tokeniser this project
does not ship; they assert that content exists for which the estimate
is too low, and that nothing in the repository calls the setting a
billing ceiling.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# The estimator's rule, stated once so the tests below describe the
# production formula rather than re-deriving it.
CHARS_PER_TOKEN = 4


def estimate(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


ADVERSARIAL: dict[str, str] = {
    "cjk": "研究者は生産性を測定した。" * 40,
    "emoji": "🚀🔬📊🧪" * 100,
    "dense_punctuation": "".join("{}[]():;,.<>/\\|!@#$%^&*" for _ in range(40)),
    "minified_code": "const f=(a,b)=>{return a?.b??c[d]||e;};" * 30,
    "json_schema": (
        '{"type":"object","properties":{"claim":{"type":"string"},'
        '"evidence":{"type":"array","items":{"type":"string"}}},'
        '"required":["claim","evidence"]}' * 20
    ),
}


class TestTheEstimateIsNotAConservativeBound:
    @pytest.mark.nli
    @pytest.mark.parametrize("name", sorted(ADVERSARIAL))
    def test_real_tokenisation_exceeds_the_estimate(self, name: str) -> None:
        """Measured, not assumed.

        The tokeniser here is the pinned NLI checkpoint's, not the
        OpenAI model's -- this project ships no OpenAI tokeniser and
        will not guess one. That is enough to make the point: content
        whose true ratio exceeds one token per four characters is
        ordinary, so a reservation built on that ratio is sometimes too
        small. It is not a measurement of OpenAI billing.
        """
        from agentic_research.citations.nli_pin import (
            NLI_DEFAULT_MODEL_ID,
            NLI_DEFAULT_REVISION,
        )
        from agentic_research.citations.nli_tokens import (
            count_pair_tokens,
            load_pair_tokenizer,
        )

        text = ADVERSARIAL[name]
        tokenizer = load_pair_tokenizer(NLI_DEFAULT_MODEL_ID, NLI_DEFAULT_REVISION)
        actual = count_pair_tokens(tokenizer, text, "")
        assert actual > estimate(text), (
            f"{name}: estimate {estimate(text)} was not exceeded by {actual}; "
            "if this ever holds for every case the comment may be revisited"
        )

    def test_the_formula_is_the_one_the_router_uses(self) -> None:
        """Guards against the tests drifting from production."""
        source = (ROOT / "src/agentic_research/llm/router.py").read_text()
        assert "// 4" in source
        assert "estimated_input" in source


class TestNothingClaimsABillingCeiling:
    """The setting bounds what the engine thinks it is spending. Saying
    otherwise in a README is how someone ends up without a provider cap
    set, believing this one protects them."""

    FORBIDDEN = re.compile(
        r"(hard|absolute|guaranteed|billing)\s+(cost\s+)?(ceiling|cap|limit)", re.I
    )

    def _tracked_text_files(self) -> list[Path]:
        import subprocess

        out = subprocess.run(
            ["git", "ls-files", "*.md", "*.yaml", "*.yml", "*.toml"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        return [ROOT / f for f in out]

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
                if self.FORBIDDEN.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}: {line.strip()[:90]}")
        assert not offenders, "spend limit described as a guarantee:\n" + "\n".join(offenders)

    def test_the_setting_documents_the_provider_cap_as_the_backstop(self) -> None:
        config = (ROOT / "src/agentic_research/config.py").read_text()
        block = config[config.index("max_cloud_cost_usd: float = Field(default=0.50") :][:1400]
        assert "Not a billing cap" in block
        assert "provider" in block.lower()
