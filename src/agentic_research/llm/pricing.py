"""Token pricing lookup.

Prices are data, not code: they live in ``pricing.toml`` so they can be
corrected without a release. An unknown model yields ``None`` rather than an
estimate — a made-up cost figure is worse than an absent one, because it looks
authoritative in a metrics table.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
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


def _find_pricing_file() -> Path | None:
    """Look beside the installed package, then walk up from the CWD.

    Covers both an editable checkout (file at the repo root) and a wheel
    install where the user drops a pricing.toml next to their .env.
    """
    candidates = [Path.cwd() / _PRICING_FILENAME]
    here = Path(__file__).resolve()
    candidates.extend(parent / _PRICING_FILENAME for parent in here.parents[:5])
    return next((c for c in candidates if c.is_file()), None)


@lru_cache(maxsize=1)
def _load_table() -> dict[str, dict[str, ModelPrice]]:
    path = _find_pricing_file()
    if path is None:
        return {}
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
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
