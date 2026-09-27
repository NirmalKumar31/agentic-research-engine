"""The cloud model is verified before it is paid for.

Preflight checked Ollama and then assumed OpenAI would work. A typo in
OPENAI_MODEL, or a model the project cannot access, surfaced as a
failed request part-way through a run that had already spent money on
earlier nodes.

The check uses model retrieval rather than a completion: it answers
"does this exist and can this key see it" without generating a token.
A completion would prove the same thing and bill for it.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from agentic_research.config import ModelRole, Provider, Settings
from agentic_research.llm.base import ModelUnavailableError
from agentic_research.llm.router import ModelRouter

MODELS_RE = r"https://api\.openai\.com/v1/models/.*"


def cloud_settings(**over: object) -> Settings:
    return Settings(llm_mode="cloud", openai_api_key="sk-test-placeholder", _env_file=None, **over)


@respx.mock
async def test_a_retrievable_model_passes() -> None:
    respx.get(url__regex=MODELS_RE).mock(
        return_value=httpx.Response(200, json={"id": "m", "object": "model"})
    )
    assert await ModelRouter(cloud_settings()).preflight() == []


@respx.mock
async def test_a_model_the_project_cannot_retrieve_fails() -> None:
    respx.get(url__regex=MODELS_RE).mock(return_value=httpx.Response(404, json={"error": {}}))
    with pytest.raises(ModelUnavailableError) as caught:
        await ModelRouter(cloud_settings()).preflight()
    assert "cannot retrieve" in str(caught.value)


@respx.mock
@pytest.mark.parametrize("status", [401, 403])
async def test_a_rejected_key_fails(status: int) -> None:
    respx.get(url__regex=MODELS_RE).mock(return_value=httpx.Response(status, json={"error": {}}))
    with pytest.raises(ModelUnavailableError) as caught:
        await ModelRouter(cloud_settings()).preflight()
    assert "key was rejected" in str(caught.value)


@respx.mock
async def test_an_unreachable_api_fails_rather_than_proceeding() -> None:
    respx.get(url__regex=MODELS_RE).mock(side_effect=httpx.ConnectError("no route"))
    with pytest.raises(ModelUnavailableError):
        await ModelRouter(cloud_settings()).preflight()


@respx.mock
async def test_no_completion_is_issued_by_the_check() -> None:
    """Verifying access must not itself cost anything."""
    models = respx.get(url__regex=MODELS_RE).mock(
        return_value=httpx.Response(200, json={"id": "m", "object": "model"})
    )
    completions = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={})
    )
    await ModelRouter(cloud_settings()).preflight()
    assert models.call_count >= 1
    assert completions.call_count == 0


@respx.mock
async def test_a_model_without_a_verified_price_is_refused() -> None:
    """A spend ceiling computed from a missing price is not a ceiling.
    The refusal happens before the network call, so an unpriced model
    cannot be reached at all."""
    respx.get(url__regex=MODELS_RE).mock(
        return_value=httpx.Response(200, json={"id": "m", "object": "model"})
    )
    settings = cloud_settings(openai_model="totally-unpriced-model")
    with pytest.raises(ModelUnavailableError) as caught:
        await ModelRouter(settings).preflight()
    assert "no verified price" in str(caught.value)


@respx.mock
async def test_the_key_is_not_in_the_error_message() -> None:
    respx.get(url__regex=MODELS_RE).mock(return_value=httpx.Response(401, json={"error": {}}))
    with pytest.raises(ModelUnavailableError) as caught:
        await ModelRouter(cloud_settings()).preflight()
    assert "sk-test-placeholder" not in str(caught.value)


class TestPricingIsRequiredNotAssumed:
    def test_unknown_model_raises_rather_than_costing_zero(self) -> None:
        from agentic_research.llm.pricing import PriceUnavailable, require_price

        with pytest.raises(PriceUnavailable):
            require_price(Provider.OPENAI, "no-such-model")

    def test_a_known_model_resolves(self) -> None:
        from agentic_research.llm.pricing import require_price

        price = require_price(Provider.OPENAI, cloud_settings().openai_model)
        assert price.input_per_mtok >= 0

    def test_there_is_one_packaged_pricing_table(self) -> None:
        """A second committed copy meant a price could be corrected in
        the file the wheel does not carry."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        assert not (root / "pricing.toml").exists()
        assert (root / "src" / "agentic_research" / "pricing.toml").is_file()


class TestRoleAssignment:
    def test_cloud_mode_assigns_openai_to_every_role(self) -> None:
        router = ModelRouter(cloud_settings())
        assert all(s.provider is Provider.OPENAI for s in router.assignments().values())
        assert ModelRole.PLANNER in router.assignments()
