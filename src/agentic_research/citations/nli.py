"""Semantic entailment via a dedicated NLI model.

Three attempts at using the 4B instruction model as a verifier failed in
three different ways: it never returned "supported" for any of thirty
claims, then never returned "partially_supported", then approved "is
typically required" against the span "you may need". The constant was a
generative model being asked for a semantic judgment it cannot make
stably, so the judgment moves to a classifier trained for exactly it.

This module returns probabilities only. It does not decide whether a
claim publishes -- that is :mod:`agentic_research.citations.publication`,
working from these scores plus the deterministic guards. Nothing here
takes an instruction prompt; it is sequence-pair classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)

# Pinned. A model id alone is a moving target, and a verifier whose
# behaviour changes under us silently invalidates every calibration
# number we publish.
DEFAULT_MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_REVISION = "main"

# Compared against the primary during calibration; smaller and faster.
ALTERNATIVE_MODEL_ID = "cross-encoder/nli-deberta-v3-base"


@dataclass(frozen=True)
class NLIScores:
    entailment: float
    neutral: float
    contradiction: float


@dataclass(frozen=True)
class NLIPrediction:
    premise: str
    hypothesis: str
    scores: NLIScores
    model_id: str
    model_revision: str


class NLIUnavailable(RuntimeError):
    """The model could not be loaded or inference failed.

    Raised rather than returning a neutral score, because a neutral
    score is indistinguishable from a real "no support" finding and
    would let an unverified claim through as a normal withhold.
    """


def _label_index(config: Any) -> dict[str, int]:
    """Map entailment/neutral/contradiction onto this model's own indices.

    Read from the checkpoint rather than hardcoded. The two candidate
    models order their classes differently, and a wrong assumption here
    silently swaps entailment for contradiction -- which would publish
    exactly the claims that should be withheld.
    """
    id2label = getattr(config, "id2label", None) or {}
    mapping: dict[str, int] = {}
    for index, label in id2label.items():
        name = str(label).strip().lower()
        for wanted in ("entailment", "neutral", "contradiction"):
            if name.startswith(wanted[:5]):
                mapping[wanted] = int(index)
    missing = {"entailment", "neutral", "contradiction"} - set(mapping)
    if missing:
        raise NLIUnavailable(
            f"model config does not expose {sorted(missing)}; got labels {id2label}"
        )
    return mapping


@lru_cache(maxsize=2)
def _load(model_id: str, revision: str) -> tuple[Any, Any, dict[str, int]]:
    """Load once per process. Cached because a verifier that reloads a
    1.4GB checkpoint per claim is not a verifier anyone will run."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise NLIUnavailable(f"transformers/torch not installed: {exc}") from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, use_safetensors=True
        )
        model.eval()
        torch.set_grad_enabled(False)
    except Exception as exc:
        raise NLIUnavailable(f"could not load {model_id}@{revision}: {exc}") from exc

    return tokenizer, model, _label_index(model.config)


class NLIVerifier:
    """Scores (premise, hypothesis) pairs in batches."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION,
        *,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.batch_size = batch_size
        self.max_length = max_length

    def score(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]:
        """Score every (premise, hypothesis) pair.

        Batched rather than one call per pair: a claim citing eight
        items would otherwise pay eight forward passes.
        """
        if not pairs:
            return []

        import torch

        tokenizer, model, labels = _load(self.model_id, self.revision)
        out: list[NLIPrediction] = []

        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            try:
                encoded = tokenizer(
                    [p for p, _ in batch],
                    [h for _, h in batch],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                logits = model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)
            except Exception as exc:
                raise NLIUnavailable(f"inference failed: {exc}") from exc

            for (premise, hypothesis), row in zip(batch, probs, strict=True):
                values = row.tolist()
                out.append(
                    NLIPrediction(
                        premise=premise,
                        hypothesis=hypothesis,
                        scores=NLIScores(
                            entailment=float(values[labels["entailment"]]),
                            neutral=float(values[labels["neutral"]]),
                            contradiction=float(values[labels["contradiction"]]),
                        ),
                        model_id=self.model_id,
                        model_revision=self.revision,
                    )
                )
        return out


class RemoteNLIVerifier:
    """Scores pairs on a hosted inference endpoint.

    For hosts that cannot hold a 1.4GB checkpoint -- the free tier this
    project deploys to has 512MB of RAM, and measured peak RSS for the
    local verifier is 1384MB.

    Every failure mode raises :class:`NLIUnavailable`, which the
    publication gate turns into a withhold. A timeout, a 429, a 5xx, a
    truncated body and a label set that does not parse all mean the same
    thing here: this claim was not verified. None of them may publish,
    and none falls back to a generative model.
    """

    def __init__(
        self,
        endpoint: str,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        batch_size: int = 8,
    ) -> None:
        self.endpoint = endpoint
        self.model_id = model_id
        self.revision = revision
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size

    def score(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]:
        if not pairs:
            return []

        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        out: list[NLIPrediction] = []

        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            payload = {
                "model": self.model_id,
                "revision": self.revision,
                "pairs": [{"premise": p, "hypothesis": h} for p, h in batch],
            }
            try:
                response = httpx.post(
                    self.endpoint, json=payload, headers=headers, timeout=self.timeout
                )
                response.raise_for_status()
                body = response.json()
            except Exception as exc:
                raise NLIUnavailable(f"remote scoring failed: {type(exc).__name__}") from exc

            results = body.get("results") if isinstance(body, dict) else None
            if not isinstance(results, list) or len(results) != len(batch):
                raise NLIUnavailable(
                    f"remote returned {type(results).__name__} for {len(batch)} pairs"
                )

            for (premise, hypothesis), item in zip(batch, results, strict=True):
                try:
                    scores = NLIScores(
                        entailment=float(item["entailment"]),
                        neutral=float(item["neutral"]),
                        contradiction=float(item["contradiction"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise NLIUnavailable(f"malformed remote score: {exc}") from exc
                out.append(
                    NLIPrediction(
                        premise=premise,
                        hypothesis=hypothesis,
                        scores=scores,
                        model_id=self.model_id,
                        model_revision=self.revision,
                    )
                )
        return out


def build_verifier(settings: object) -> NLIVerifier | RemoteNLIVerifier:
    """Pick local or remote from settings, failing loudly on a bad combination."""
    mode = getattr(settings, "nli_mode", "local")
    model_id = getattr(settings, "nli_model_id", DEFAULT_MODEL_ID)
    revision = getattr(settings, "nli_model_revision", DEFAULT_REVISION)

    if mode == "remote":
        endpoint = getattr(settings, "nli_endpoint", None)
        if not endpoint:
            raise NLIUnavailable("nli_mode is 'remote' but no nli_endpoint is configured")
        key = getattr(settings, "nli_api_key", None)
        return RemoteNLIVerifier(
            endpoint,
            model_id,
            revision,
            api_key=key.get_secret_value() if key is not None else None,
            timeout=getattr(settings, "nli_timeout_seconds", 30.0),
        )
    return NLIVerifier(model_id, revision)
