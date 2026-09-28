"""Unit tests must not read the developer's .env.

A suite whose result depends on whose machine it runs on is not a
suite. This project has hit it twice: once when the default LLM_MODE
needed a key that a local .env supplied and CI did not, and again when
NLI_MODE=remote in a developer's .env made "local is the default" fail
for that developer only.

Both were found by a person noticing. These tests plant a hostile .env
in the working directory and check that the helpers the suite actually
uses ignore it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_research.config import LLMMode, Settings


def _fake_credential(prefix: str, body: str) -> str:
    """Assemble a credential-shaped canary at runtime.

    Never written as a whole literal. A complete shape in the source is
    a finding for the history scan, and allowlisting it would widen the
    scan's blind spot over a string that was never a credential. Split
    across arguments, no single line matches a provider rule while the
    value at runtime is exactly the shape under test.
    """
    return prefix + body


# Assembled, for the reason above.
_OPENAI = _fake_credential("sk-", "hostileValueFromADeveloperEnvFile1234567")
_TAVILY = _fake_credential("tvly-", "hostileValueFromADeveloperEnvFile")
_HF = _fake_credential("hf_", "hostileValueFromADeveloperEnvFile12345")
_QUOTA = _fake_credential("redis://", "hostile:hostile@example.invalid:6379/0")

HOSTILE = f"""
LLM_MODE=cloud
OPENAI_API_KEY={_OPENAI}
TAVILY_API_KEY={_TAVILY}
NLI_MODE=remote
NLI_ENDPOINT=https://hostile.endpoints.huggingface.cloud
NLI_API_KEY={_HF}
NLI_DIALECT=hf
DEMO_QUOTA_URL={_QUOTA}
MAX_SOURCES=99
DEMO_PROVIDER_REQUESTS_PER_DAY=99999
"""


@pytest.fixture
def hostile_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".env").write_text(HOSTILE)
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestTheHostileEnvIsRealEnoughToMatter:
    def test_settings_without_isolation_does_read_it(self, hostile_cwd: Path) -> None:
        """The control. If this ever stops being true the other tests
        below prove nothing, because there would be nothing to ignore."""
        leaked = Settings()
        assert leaked.llm_mode is LLMMode.CLOUD
        assert leaked.nli_mode == "remote"
        assert leaked.max_sources == 99


class TestIsolationActuallyIsolates:
    def test_env_file_none_ignores_it_completely(self, hostile_cwd: Path) -> None:
        settings = Settings(llm_mode=LLMMode.LOCAL, _env_file=None)  # type: ignore[arg-type]
        assert settings.llm_mode is LLMMode.LOCAL
        assert settings.nli_mode == "local"
        assert settings.nli_endpoint is None
        assert settings.nli_api_key is None
        assert settings.max_sources != 99

    def test_no_credential_reaches_an_isolated_settings(self, hostile_cwd: Path) -> None:
        settings = Settings(_env_file=None)  # type: ignore[arg-type]
        assert settings.openai_api_key is None
        assert settings.tavily_api_key is None
        assert settings.nli_api_key is None
        assert settings.demo_quota_url is None


class TestTheSuiteHelpersAreIsolated:
    """The helpers the rest of the unit tests build their apps from."""

    def test_the_shared_settings_fixture_is_isolated(
        self, hostile_cwd: Path, settings: Settings
    ) -> None:
        """conftest's `settings` fixture, used across the suite."""
        assert settings.llm_mode is LLMMode.LOCAL
        assert settings.nli_mode == "local"
        assert settings.max_sources != 99

    def test_the_web_helpers_are_isolated(self, hostile_cwd: Path) -> None:
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        from test_durable_quota_wiring import live_settings
        from test_http_hardening import demo_settings

        for built in (live_settings(), demo_settings()):
            assert built.nli_endpoint != "https://hostile.endpoints.huggingface.cloud"
            assert built.demo_quota_url != _QUOTA

    def test_the_preflight_helpers_are_isolated(self, hostile_cwd: Path) -> None:
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        from test_nli_preflight_gate import settings as gate_settings

        built = gate_settings()
        assert built.llm_mode is LLMMode.LOCAL
        assert built.nli_mode == "local", (
            "the preflight probe read the developer's NLI configuration"
        )
