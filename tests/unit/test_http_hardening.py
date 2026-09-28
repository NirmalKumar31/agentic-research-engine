"""Headers, body size, and the paths that must not answer.

The deployed service returned no security headers at all, and answered
200 for /docs and /redoc because the SPA fallback caught them after
routing failed. Neither is exotic; both were invisible from the test
suite because every test asked the API a question and none looked at
what came back around the answer.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentic_research.config import Settings
from agentic_research.web.api import _MAX_BODY_BYTES, create_app
from fakes import FakeQuotaStore


def demo_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_mode": "local",
        "demo_mode": True,
        "live_research_enabled": True,
        "nli_mode": "remote",
        "nli_endpoint": "https://nli.test.invalid/score",
        "nli_api_key": "hf-test-placeholder",
        "demo_quota_url": "redis://quota.test.invalid:6379/0",
        "tavily_api_key": "tvly-test-key",
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def client(**overrides: Any) -> TestClient:
    app = create_app(demo_settings(**overrides), counter_factory=FakeQuotaStore().factory)
    return TestClient(app)


class TestSecurityHeaders:
    EXPECTED = {
        "content-security-policy",
        "x-content-type-options",
        "referrer-policy",
        "permissions-policy",
        "x-frame-options",
        "strict-transport-security",
    }

    def test_every_header_is_present_on_an_api_response(self) -> None:
        with client() as c:
            headers = {k.lower() for k in c.get("/api/health").headers}
        assert headers >= self.EXPECTED, f"missing: {self.EXPECTED - headers}"

    def test_they_are_present_on_errors_too(self) -> None:
        """An error page is a page. Omitting them there is how a 404
        becomes the one document an attacker can influence."""
        with client() as c:
            headers = {k.lower() for k in c.get("/api/nope").headers}
        assert headers >= self.EXPECTED

    def test_framing_is_denied_two_ways(self) -> None:
        """frame-ancestors for modern browsers, X-Frame-Options for the
        rest. A framed demo can be overlaid and read as it is typed."""
        with client() as c:
            h = c.get("/api/health").headers
        assert h["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in h["content-security-policy"]

    def test_scripts_are_not_allowed_to_be_inline(self) -> None:
        """The style exemption exists because a static Vite build inlines
        one; extending it to scripts would undo the policy."""
        csp = {
            part.strip().split(" ")[0]: part.strip()
            for part in client().get("/api/health").headers["content-security-policy"].split(";")
        }
        assert "'unsafe-inline'" not in csp["script-src"]
        assert "'unsafe-eval'" not in csp["script-src"]

    def test_hsts_does_not_claim_sibling_subdomains(self) -> None:
        """This runs on a shared onrender.com hostname. Asserting HSTS
        for every sibling is not this service's call to make."""
        with client() as c:
            hsts = c.get("/api/health").headers["strict-transport-security"]
        assert "includesubdomains" not in hsts.lower()

    def test_unused_device_capabilities_are_disabled(self) -> None:
        with client() as c:
            policy = c.get("/api/health").headers["permissions-policy"]
        for feature in ("camera", "microphone", "geolocation"):
            assert f"{feature}=()" in policy


class TestDocsAreActuallyAbsent:
    @pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
    def test_demo_mode_returns_404_not_the_app_shell(self, path: str) -> None:
        """Returning the SPA with 200 hides the docs from a human and
        not from anything that reads a status code."""
        with client() as c:
            response = c.get(path)
        assert response.status_code == 404
        assert response.json() == {"error": "Not found."}

    @pytest.mark.parametrize("path", ["/docs", "/openapi.json"])
    def test_a_non_demo_deployment_still_serves_them(self, path: str) -> None:
        with client(demo_mode=False) as c:
            assert c.get(path).status_code == 200


class TestBodySizeLimit:
    def test_an_oversized_body_is_refused_with_413(self) -> None:
        payload = json.dumps({"query": "a" * (_MAX_BODY_BYTES * 2)})
        with client() as c:
            response = c.post(
                "/api/research", content=payload, headers={"Content-Type": "application/json"}
            )
        assert response.status_code == 413

    def test_a_lying_content_length_is_refused(self) -> None:
        """Declared length is checked when present; the stream is
        counted regardless, because a chunked body can omit or lie."""
        with client() as c:
            response = c.post(
                "/api/research",
                content=json.dumps({"query": "x" * 50}),
                headers={"Content-Type": "application/json", "Content-Length": "99999999"},
            )
        assert response.status_code == 413

    def test_a_non_numeric_content_length_is_refused(self) -> None:
        with client() as c:
            response = c.post(
                "/api/research",
                content=json.dumps({"query": "x" * 50}),
                headers={"Content-Type": "application/json", "Content-Length": "banana"},
            )
        assert response.status_code in (400, 413)

    def test_an_ordinary_question_is_unaffected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The limit must not be the thing that breaks normal use.

        The engine is stubbed: this is about the body limit, and letting
        the request reach a real run made the test depend on whether a
        model provider happened to be reachable.
        """
        from agentic_research.web import api as api_module

        async def fake_stream(query: str, settings: Any, run_id: str = "") -> Any:
            yield {"event": "error", "error": "stubbed"}

        monkeypatch.setattr(api_module, "stream_research", fake_stream)
        with client() as c:
            response = c.post(
                "/api/research",
                json={"query": "What did the study measure about developer productivity?"},
            )
        assert response.status_code != 413

    def test_the_limit_is_far_above_a_real_question(self) -> None:
        assert _MAX_BODY_BYTES >= 4096
