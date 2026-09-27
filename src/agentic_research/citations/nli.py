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
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from agentic_research.citations.nli_pin import (
    NLI_DEFAULT_MODEL_ID,
    NLI_DEFAULT_REVISION,
)

log = logging.getLogger(__name__)

# Pinned. A model id alone is a moving target, and a verifier whose
# behaviour changes under us silently invalidates every calibration
# number we publish.
DEFAULT_MODEL_ID = NLI_DEFAULT_MODEL_ID

# The exact commit the 0.98 threshold was calibrated against. There is no
# unpinned default: "main" moves, and a verifier whose weights change
# under a fixed threshold is no longer the verifier that was measured.
# Settings carries the same pair, and a test asserts they agree.
DEFAULT_REVISION = NLI_DEFAULT_REVISION

# Wire format this client speaks. A remote answering a different version
# is not a compatible endpoint, whatever else it returns.
CONTRACT_VERSION = 1

# Compared against the primary during calibration; smaller and faster.
ALTERNATIVE_MODEL_ID = "cross-encoder/nli-deberta-v3-base"


# Softmax outputs sum to one. A scorer whose three values do not are
# not probabilities, and comparing one of them to a threshold is
# meaningless -- (1, 1, 1) has "entailment 1.0" and says nothing.
_DISTRIBUTION_TOLERANCE = 0.02


@dataclass(frozen=True)
class NLIScores:
    entailment: float
    neutral: float
    contradiction: float

    def is_distribution(self) -> bool:
        """Whether these are three probabilities over one decision.

        The local model's softmax satisfies this by construction. The
        check exists for remote and alternate scorers, where nothing
        guarantees it and a malformed response would otherwise publish
        on a high first number.
        """
        values = (self.entailment, self.neutral, self.contradiction)
        if not all(math.isfinite(v) for v in values):
            return False
        if not all(0.0 <= v <= 1.0 for v in values):
            return False
        return abs(sum(values) - 1.0) <= _DISTRIBUTION_TOLERANCE


@dataclass(frozen=True)
class NLIPrediction:
    premise: str
    hypothesis: str
    scores: NLIScores
    model_id: str
    model_revision: str
    truncated: bool = False
    """The pair did not fit the model's context and was cut.

    Recorded because a truncated premise is not the premise. A quote
    whose first half supports a claim and whose second half qualifies it
    away would score as support with the qualifier cut off, and the
    score would look entirely normal. Callers withhold on this rather
    than trusting it."""


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

        # Guarded: on a host without torch this must withhold, not
        # crash the run. An unhandled ImportError here would propagate
        # out of verification as an ordinary exception rather than as
        # the typed "unverified" signal the gate fails closed on.
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise NLIUnavailable(f"torch is not installed: {exc}") from exc

        tokenizer, model, labels = _load(self.model_id, self.revision)
        out: list[NLIPrediction] = []

        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            try:
                # Tokenised twice on purpose: once unbounded to learn
                # the true length, once truncated for the model. The
                # difference is the only way to know whether anything
                # was cut, and a silently truncated premise is a
                # different claim than the one on the page.
                measured = tokenizer(
                    [p for p, _ in batch],
                    [h for _, h in batch],
                    truncation=False,
                    padding=False,
                )["input_ids"]
                encoded = tokenizer(
                    [p for p, _ in batch],
                    [h for _, h in batch],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                overlong = [len(ids) > self.max_length for ids in measured]
                logits = model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)
            except Exception as exc:
                raise NLIUnavailable(f"inference failed: {exc}") from exc

            for (premise, hypothesis), row, cut in zip(batch, probs, overlong, strict=True):
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
                        truncated=cut,
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
            # Stable ids so results are matched by identity rather than
            # by list position. A remote that reorders, drops or
            # duplicates results would otherwise silently attach one
            # claim's score to another claim.
            pair_ids = [f"{start + i}" for i in range(len(batch))]
            payload = {
                "contract_version": CONTRACT_VERSION,
                "model": self.model_id,
                "revision": self.revision,
                "pairs": [
                    {"pair_id": pid, "premise": p, "hypothesis": h}
                    for pid, (p, h) in zip(pair_ids, batch, strict=True)
                ],
            }
            try:
                response = httpx.post(
                    self.endpoint, json=payload, headers=headers, timeout=self.timeout
                )
                response.raise_for_status()
                body = response.json()
            except Exception as exc:
                raise NLIUnavailable(f"remote scoring failed: {type(exc).__name__}") from exc

            if not isinstance(body, dict):
                raise NLIUnavailable(f"remote returned {type(body).__name__}, not an object")

            # The endpoint has to say what served the request. Without
            # this a silently redeployed remote could answer with a
            # different checkpoint than the calibrated one and every
            # published claim would cite a threshold it was never
            # measured against.
            for field in ("contract_version", "model_id", "model_revision"):
                if not body.get(field):
                    raise NLIUnavailable(f"remote response omits {field}")
            if body["contract_version"] != CONTRACT_VERSION:
                raise NLIUnavailable(
                    f"remote speaks contract {body['contract_version']}, "
                    f"this client speaks {CONTRACT_VERSION}"
                )
            if body["model_id"] != self.model_id or body["model_revision"] != self.revision:
                raise NLIUnavailable(
                    f"remote served {body['model_id']}@{body['model_revision']}, "
                    f"configured for {self.model_id}@{self.revision}"
                )

            results = body.get("results")
            if not isinstance(results, list) or len(results) != len(batch):
                raise NLIUnavailable(
                    f"remote returned {type(results).__name__} for {len(batch)} pairs"
                )

            # Index by echoed id and reject anything that does not line
            # up exactly: a missing, unknown or duplicated id means the
            # mapping from claims to scores is unknown, and an unknown
            # mapping is not a score.
            by_id: dict[str, Any] = {}
            for item in results:
                if not isinstance(item, dict) or "pair_id" not in item:
                    raise NLIUnavailable("remote result omits pair_id")
                pid = str(item["pair_id"])
                if pid in by_id:
                    raise NLIUnavailable(f"remote returned duplicate pair_id {pid}")
                by_id[pid] = item
            if set(by_id) != set(pair_ids):
                raise NLIUnavailable(
                    f"remote pair_ids {sorted(by_id)} do not match requested {sorted(pair_ids)}"
                )

            for pid, (premise, hypothesis) in zip(pair_ids, batch, strict=True):
                item = by_id[pid]
                try:
                    scores = NLIScores(
                        entailment=float(item["entailment"]),
                        neutral=float(item["neutral"]),
                        contradiction=float(item["contradiction"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise NLIUnavailable(f"malformed remote score: {exc}") from exc
                if not scores.is_distribution():
                    raise NLIUnavailable(
                        "remote scores are not a probability distribution: "
                        f"{scores.entailment}, {scores.neutral}, {scores.contradiction}"
                    )
                # Truncation must be stated, not assumed. Defaulting a
                # missing field to False would let an endpoint that
                # silently cut a long premise report support for a claim
                # the qualifying half contradicts.
                if "truncated" not in item:
                    raise NLIUnavailable("remote result omits truncated")
                out.append(
                    NLIPrediction(
                        premise=premise,
                        hypothesis=hypothesis,
                        scores=scores,
                        model_id=body["model_id"],
                        model_revision=body["model_revision"],
                        truncated=bool(item["truncated"]),
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
