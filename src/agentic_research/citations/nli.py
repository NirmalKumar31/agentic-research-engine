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
