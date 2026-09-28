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
    cached_input_per_mtok: float | None = None
    """Rate for input the provider served from its prompt cache.

    ``None`` means this project has not recorded one. It is not zero and
    it is not the full rate: when a response reports cached tokens and
    no rate is known, the run's cost is marked incomplete rather than
    priced at a number nobody checked."""
    cache_write_per_mtok: float | None = None
    """Rate for writing the cache, where a provider bills it separately.
    Same rule as above when unknown."""

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Flat pricing, every input token at the full rate.

        This is what the *reservation* uses. Cache status is not knowable
        before dispatch, and the full rate is the higher of the two, so
        reserving at it cannot under-reserve."""
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000

    def reconcile(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cached_input_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> tuple[float, bool]:
        """Charge for a completed response, and whether it is complete.

        Returns ``(usd, complete)``. ``complete`` is False when the
        response reported a token category this project has no rate for,
        because pricing an unknown category at zero is how a cost figure
        becomes fiction while still looking authoritative.

        Reasoning tokens are deliberately not a separate term: providers
        that emit them count them inside ``output_tokens``, so adding
        them again would double-charge. They are recorded separately for
        reporting.
        """
        complete = True

        cached = max(0, min(cached_input_tokens, input_tokens))
        uncached = input_tokens - cached
        total = uncached * self.input_per_mtok

        if cached:
            if self.cached_input_per_mtok is None:
                # Charge at the full rate, which cannot under-state, and
                # say the number is incomplete.
                total += cached * self.input_per_mtok
                complete = False
            else:
                total += cached * self.cached_input_per_mtok

        if cache_write_tokens:
            if self.cache_write_per_mtok is None:
                complete = False
            else:
                total += cache_write_tokens * self.cache_write_per_mtok

        total += output_tokens * self.output_per_mtok
        return total / 1_000_000, complete


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
                    cached_input_per_mtok=(
                        float(price["cached_input"]) if "cached_input" in price else None
                    ),
                    cache_write_per_mtok=(
                        float(price["cache_write"]) if "cache_write" in price else None
                    ),
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
    _load_aliases.cache_clear()


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
    models = _load_table().get(provider.value, {})

    if model in models:
        return models[model]

    resolved = _load_aliases().get(provider.value, {}).get(model)
    if resolved and resolved in models:
        return models[resolved]

    # Deliberately no prefix fallback here. get_price() keeps it, which
    # is right for reporting on a historical artifact, and it is wrong
    # on the path that authorises spending: it let any dated snapshot
    # inherit its family's price, so "gpt-6-sol-2027-01-01" would have
    # been billed at today's gpt-6-sol rate without anyone reviewing
    # whether that is what the provider charges. A new snapshot now
    # stops the run until somebody looks.
    raise PriceUnavailable(
        f"no reviewed price for {provider.value}:{model}. "
        "Add it to pricing.toml, or map it to a reviewed snapshot under "
        f"[aliases.{provider.value}], before running in a paid mode. A "
        "budget ceiling cannot be enforced on a price nobody checked."
    )


@lru_cache(maxsize=1)
def _load_aliases() -> dict[str, dict[str, str]]:
    """Explicit alias-to-snapshot mappings, reviewed by a person.

    Exists so a dated snapshot can be priced without the blanket prefix
    rule that made every future model silently inherit an old rate.
    """
    text = _read_pricing()
    if text is None:
        return {}
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}
    aliases = raw.get("aliases", {})
    if not isinstance(aliases, dict):
        return {}
    return {
        provider: {str(k): str(v) for k, v in mapping.items()}
        for provider, mapping in aliases.items()
        if isinstance(mapping, dict)
    }
