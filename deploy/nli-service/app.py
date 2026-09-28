"""Sequence-pair NLI, and nothing else.

This is the server side of the contract in
``agentic_research.citations.nli.RemoteNLIVerifier``. It exists so the
Render web service can stay on a 512 MB plan: the measured peak for this
model is about 1.3 GB, which does not fit, and putting torch into the
web image would not make it fit.

Deliberately incapable of anything else. No search, no generation, no
synthesis, no retrieval, no publication decision. It turns (premise,
hypothesis) pairs into three probabilities. The threshold, the guards
and the publish/withhold decision all stay in the research engine,
where they are tested.

Weights come from the Hugging Face endpoint's mounted repository at
/repository rather than a runtime download, so the served checkpoint is
whatever the endpoint was pinned to and cannot drift between restarts.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("nli-service")

CONTRACT_VERSION = 1
MODEL_DIR = os.environ.get("NLI_MODEL_DIR", "/repository")
EXPECTED_MODEL_ID = os.environ.get(
    "NLI_MODEL_ID", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
)
EXPECTED_REVISION = os.environ.get("NLI_MODEL_REVISION", "")

# Bounded so one request cannot exhaust the box. A claim citing more
# than eight quotes is refused upstream anyway.
MAX_PAIRS = 32
MAX_CHARS = 8000
MAX_LENGTH = 512

app = FastAPI(title="nli-service", docs_url=None, redoc_url=None, openapi_url=None)


class Pair(BaseModel):
    pair_id: str = Field(min_length=1, max_length=64)
    premise: str = Field(max_length=MAX_CHARS)
    hypothesis: str = Field(max_length=MAX_CHARS)


class ScoreRequest(BaseModel):
    model: str
    revision: str
    pairs: list[Pair] = Field(min_length=1, max_length=MAX_PAIRS)
    contract_version: int | None = None


@lru_cache(maxsize=1)
def _load() -> tuple[Any, Any, dict[str, int]]:
    """Load once per process from the mounted repository."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, use_safetensors=True
    )
    model.eval()
    torch.set_grad_enabled(False)

    # Read the class order from the checkpoint. This model puts
    # entailment at index 0 and the cross-encoder family puts
    # contradiction there; hardcoding would invert the gate while still
    # returning plausible numbers.
    id2label = getattr(model.config, "id2label", None) or {}
    labels: dict[str, int] = {}
    for index, label in id2label.items():
        name = str(label).strip().lower()
        for wanted in ("entailment", "neutral", "contradiction"):
            if name.startswith(wanted[:5]):
                labels[wanted] = int(index)
    missing = {"entailment", "neutral", "contradiction"} - set(labels)
    if missing:
        raise RuntimeError(f"checkpoint does not expose {sorted(missing)}; got {id2label}")
    return tokenizer, model, labels


@app.get("/health")
def health() -> dict[str, Any]:
    """Identity and readiness, with nothing secret in it."""
    try:
        _tokenizer, _model, labels = _load()
    except Exception as exc:
        log.exception("model failed to load")
        return {
            "ready": False,
            "detail": type(exc).__name__,
            "contract_version": CONTRACT_VERSION,
        }
    return {
        "ready": True,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_REVISION,
        "contract_version": CONTRACT_VERSION,
        "label_order": labels,
        "max_length": MAX_LENGTH,
    }


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    """Score every pair, refusing anything but the configured checkpoint.

    The client may not select a model. Accepting a client-supplied
    identifier would let a caller silently swap the verifier for one the
    0.98 threshold was never calibrated against.
    """
    if request.contract_version is not None and request.contract_version != CONTRACT_VERSION:
        raise HTTPException(400, f"unsupported contract_version {request.contract_version}")
    if request.model != EXPECTED_MODEL_ID:
        raise HTTPException(400, "model does not match this endpoint")
    if EXPECTED_REVISION and request.revision != EXPECTED_REVISION:
        raise HTTPException(400, "revision does not match this endpoint")

    seen = {p.pair_id for p in request.pairs}
    if len(seen) != len(request.pairs):
        raise HTTPException(400, "duplicate pair_id")

    import torch

    tokenizer, model, labels = _load()

    # Measured unbounded first, so truncation is reported rather than
    # assumed. A premise whose qualifying half was cut is not the
    # premise, and the caller withholds on it.
    measured = tokenizer(
        [p.premise for p in request.pairs],
        [p.hypothesis for p in request.pairs],
        truncation=False,
        padding=False,
    )["input_ids"]
    encoded = tokenizer(
        [p.premise for p in request.pairs],
        [p.hypothesis for p in request.pairs],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    probabilities = torch.softmax(model(**encoded).logits, dim=-1)

    results = []
    for pair, row, ids in zip(request.pairs, probabilities, measured, strict=True):
        values = row.tolist()
        triple = (
            float(values[labels["entailment"]]),
            float(values[labels["neutral"]]),
            float(values[labels["contradiction"]]),
        )
        if not all(v == v and 0.0 <= v <= 1.0 for v in triple):  # NaN-safe
            raise HTTPException(500, "classifier produced an invalid distribution")
        results.append(
            {
                "pair_id": pair.pair_id,
                "entailment": triple[0],
                "neutral": triple[1],
                "contradiction": triple[2],
                "truncated": len(ids) > MAX_LENGTH,
            }
        )

    # Claim and evidence text is the caller's research content; it is
    # not logged here at any level that a hosted provider retains.
    log.info("scored pairs=%d truncated=%d", len(results), sum(r["truncated"] for r in results))
    return {
        "contract_version": CONTRACT_VERSION,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_REVISION,
        "results": results,
    }
