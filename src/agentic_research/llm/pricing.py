"""Token pricing lookup.

Prices are data, not code: they live in ``pricing.toml`` so they can be
corrected without a release. An unknown model yields ``None`` rather than an
estimate — a made-up cost figure is worse than an absent one, because it looks
authoritative in a metrics table.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from importlib import resources
from pathlib import Path

from pydantic import BaseModel

from agentic_research.config import Provider

_PRICING_FILENAME = "pricing.toml"


class ModelPrice(BaseModel):
    model_config = {"frozen": True}

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


def _read_pricing() -> str | None:
    """Prefer a local override, then fall back to the packaged table.

    The packaged copy is what makes an installed wheel work: resolving
    prices by walking up from the current directory meant cost reporting
    silently returned nothing whenever the process ran outside a checkout.

    A pricing.toml in the working directory still wins, so an operator
    can correct a price without reinstalling. That override is the only
    reason a second copy may exist: the repository root no longer ships
    one, because two identical committed tables meant a price could be
    edited in the copy the wheel does not carry.
    """
    local = Path.cwd() / _PRICING_FILENAME
    if local.is_file():
        try:
            return local.read_text(encoding="utf-8")
        except OSError:
            pass
    try:
        return resources.files("agentic_research").joinpath(_PRICING_FILENAME).read_text("utf-8")
    except (FileNotFoundError, OSError, ModuleNotFoundError):
        return None


@lru_cache(maxsize=1)
def _load_table() -> dict[str, dict[str, ModelPrice]]:
    text = _read_pricing()
    if text is None:
        return {}
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}

    table: dict[str, dict[str, ModelPrice]] = {}
    for provider, models in raw.items():
        if not isinstance(models, dict):
            continue
        entries: dict[str, ModelPrice] = {}
        for model, price in models.items():
            if isinstance(price, dict) and "input" in price and "output" in price:
                entries[model] = ModelPrice(
                    input_per_mtok=float(price["input"]),
                    output_per_mtok=float(price["output"]),
                )
        table[provider] = entries
    return table


def get_price(provider: Provider, model: str) -> ModelPrice | None:
    """Price for a model, or ``None`` if we genuinely do not know it.

    Falls back to a ``"*"`` wildcard entry, which is how locally hosted models
    are priced at zero without enumerating every tag a user might pull.
    """
    models = _load_table().get(provider.value, {})
    if model in models:
        return models[model]
    # Match the longest matching prefix before the wildcard, so a dated
    # snapshot like "gpt-6-sol-2026-09-01" inherits "gpt-6-sol" pricing.
    prefix_matches = [m for m in models if m != "*" and model.startswith(m)]
    if prefix_matches:
        return models[max(prefix_matches, key=len)]
    return models.get("*")


def reset_cache() -> None:
    _load_table.cache_clear()


class PriceUnavailable(RuntimeError):
    """No verified price exists for a model the run is about to call.

    Raised rather than defaulting to zero or to another model's price.
    A spend ceiling computed from an invented price is not a ceiling,
    and the failure it produces is an invoice.
    """


def require_price(provider: Provider, model: str) -> ModelPrice:
    """The price for a model, or a refusal to proceed without one.

    Callers that reserve budget must use this. ``get_price`` returning
    None is appropriate for reporting -- a historical artifact may name
    a model whose price is no longer listed -- but it must never become
    "assume free" on the path that authorises spending.
    """
    price = get_price(provider, model)
    if price is None:
        raise PriceUnavailable(
            f"no verified price for {provider.value}:{model}. "
            "Add it to pricing.toml before running in a paid mode; "
            "a budget ceiling cannot be enforced without one."
        )
    return price
