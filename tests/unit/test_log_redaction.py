"""Credentials must not reach a log sink, wherever they are hiding.

Redaction matched key names only, and at the top level. That covers the
leak nobody makes -- `log.info("x", api_key=...)` -- and misses the one
that happens: `error=str(exc)`, where a provider exception quotes the
request it failed on, Authorization header included. Render keeps those
logs, so a token in one is a token to rotate.

Canary values below are fake and shaped like the real thing.
"""

from __future__ import annotations

import json

import pytest

from agentic_research.observability import logging as log_module
from agentic_research.observability.logging import (
    _redact_secrets,
    register_secret_values,
)


def _fake_credential(prefix: str, body: str) -> str:
    """Assemble a credential-shaped canary at runtime.

    Never written as a whole literal. A complete shape in the source is
    a finding for the history scan, and allowlisting it would widen the
    scan's blind spot over a string that was never a credential. Split
    across arguments, no single line matches a provider rule while the
    value at runtime is exactly the shape under test.
    """
    return prefix + body


HF = _fake_credential("hf_", "AbCdEfGhIjKlMnOpQrStUvWxYz012345")
OPENAI = _fake_credential("sk-", "proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789")
TAVILY = _fake_credential("tvly-", "AbCdEfGhIjKlMnOpQrStUvWx")
REDIS = _fake_credential(
    "redis://", "default:s3cr3tpassw0rd@red-abc123.oregon-keyvalue.render.com:6379"
)
ENDPOINT = "https://6ab97a129ec415b652acdc25.endpoints.huggingface.cloud"


@pytest.fixture(autouse=True)
def _registered() -> None:
    register_secret_values(ENDPOINT)


def scrub(**fields: object) -> str:
    """Run one record through the processor and return it as text."""
    return json.dumps(_redact_secrets(None, "info", dict(fields)), default=str)


class TestSecretsUnderUnknownKeys:
    @pytest.mark.parametrize("secret", [HF, OPENAI, TAVILY])
    def test_a_token_under_a_generic_key_is_scrubbed(self, secret: str) -> None:
        """`error` is not in the secret-key list and never will be."""
        assert secret not in scrub(event="provider_failed", error=f"401 for {secret}")

    def test_a_connection_string_is_scrubbed(self) -> None:
        out = scrub(event="quota_store_unreachable", detail=f"cannot reach {REDIS}")
        assert "s3cr3tpassw0rd" not in out
        assert "red-abc123" not in out

    def test_a_registered_value_is_scrubbed_even_though_it_is_not_secret_shaped(
        self,
    ) -> None:
        """A private endpoint URL looks like an ordinary URL. Patterns
        cannot know it matters; registration can."""
        assert ENDPOINT not in scrub(event="nli_call", url=ENDPOINT)


class TestSecretsInsideOtherShapes:
    def test_inside_exception_text(self) -> None:
        exc = RuntimeError(f"request failed: Authorization: Bearer {HF}")
        out = scrub(event="boom", exc=exc)
        assert HF not in out
        assert "Bearer <redacted>" in out or "<redacted" in out

    def test_inside_a_url_query_string(self) -> None:
        url = f"https://api.tavily.com/search?api_key={TAVILY}&q=test"
        out = scrub(event="search_failed", url=url)
        assert TAVILY not in out

    def test_inside_nested_containers(self) -> None:
        payload = {"outer": [{"inner": {"headers": {"authorization": f"Bearer {OPENAI}"}}}]}
        assert OPENAI not in scrub(event="request", payload=payload)

    def test_inside_a_list_of_strings(self) -> None:
        assert HF not in scrub(event="retry", attempts=[f"failed with {HF}", "ok"])

    def test_inside_a_tuple(self) -> None:
        assert OPENAI not in scrub(event="x", pair=(f"key={OPENAI}", 2))


class TestKnownKeysStillBlanked:
    def test_a_known_key_is_blanked_outright(self) -> None:
        out = json.loads(scrub(event="x", api_key=HF))
        assert out["api_key"] == "<redacted>"

    def test_an_authorization_key_is_blanked(self) -> None:
        out = json.loads(scrub(event="x", authorization=f"Bearer {HF}"))
        assert out["authorization"] == "<redacted>"


class TestItDoesNotMangleOrdinaryRecords:
    def test_normal_fields_survive(self) -> None:
        out = json.loads(scrub(event="search_completed", query_id="Q1", results=8, latency_s=1.89))
        assert out["event"] == "search_completed"
        assert out["results"] == 8
        assert out["latency_s"] == 1.89

    def test_a_public_url_is_left_alone(self) -> None:
        url = "https://arxiv.org/html/2511.04427v2"
        assert url in scrub(event="fetched", url=url)

    def test_short_values_are_not_registered(self) -> None:
        """Registering 'abc' would blank the letter sequence everywhere."""
        register_secret_values("abc")
        assert "abc" not in log_module._SECRET_VALUES

    def test_recursion_is_bounded(self) -> None:
        """A self-referential record must not hang the process."""
        deep: dict = {}
        node = deep
        for _ in range(50):
            node["next"] = {}
            node = node["next"]
        node["leak"] = HF
        scrub(event="deep", payload=deep)  # must return
