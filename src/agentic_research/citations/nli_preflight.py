"""Prove the verifier works before any paid work begins.

Verification used to be the last thing a run did. Planning, searching,
fetching, extraction and synthesis all completed -- spending OpenAI
tokens and Tavily credits -- and only then was the NLI verifier
constructed. A misconfigured endpoint therefore cost real money to
discover, and the run that discovered it published nothing, because a
claim that cannot be checked is withheld.

So the verifier is exercised first, with a fixed probe that costs
nothing and touches no provider. If it cannot answer correctly, the run
stops before its first paid request.

The probe is deliberately trivial. It is not a calibration: it asks
whether the thing on the other end is a working NLI classifier serving
the revision we pinned, not whether it is a good one. Whether it is a
good one was settled by the calibration and the adversarial suite.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_research.citations.nli import NLIUnavailable, build_verifier
from agentic_research.citations.nli_endpoint_meta import verify_endpoint_pin
from agentic_research.observability import get_logger

log = get_logger(__name__)

# One entailed pair and one contradicted pair. Short enough that no
# tokenizer truncates them, unambiguous enough that any working MNLI
# model separates them by a wide margin.
_ENTAILED = ("The build completed successfully.", "The build succeeded.")
_CONTRADICTED = ("The build completed successfully.", "The build failed.")

# Deliberately loose. A wrong-but-working classifier should pass this;
# only a broken, misconfigured or non-NLI endpoint should fail. Tight
# thresholds here would turn an unrelated model update into an outage.
_MIN_ENTAILMENT = 0.50
_MIN_CONTRADICTION = 0.50


@dataclass(frozen=True)
class NLIReadiness:
    ready: bool
    detail: str
    model_id: str = ""
    model_revision: str = ""


def check_nli_ready(settings: object) -> NLIReadiness:
    """Whether semantic verification can actually run.

    Never raises. A failure here is a reason to decline the run, not an
    exception for the caller to interpret, and the message is written
    to be shown to a person.
    """
    pin = _check_managed_pin(settings)
    if pin is not None:
        return pin

    try:
        scorer = build_verifier(settings)
    except NLIUnavailable as exc:
        return NLIReadiness(False, f"verifier unavailable: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        return NLIReadiness(False, f"verifier could not be built: {type(exc).__name__}")

    try:
        predictions = scorer.score([_ENTAILED, _CONTRADICTED])
    except NLIUnavailable as exc:
        return NLIReadiness(False, f"verifier did not respond: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        return NLIReadiness(False, f"verifier probe failed: {type(exc).__name__}")

    if len(predictions) != 2:
        return NLIReadiness(False, f"probe returned {len(predictions)} of 2 results")

    entailed, contradicted = predictions
    for prediction in predictions:
        if not prediction.scores.is_distribution():
            return NLIReadiness(False, "probe scores are not a probability distribution")

    # Identity, not quality. A remote serving a different checkpoint is
    # already refused inside the client; this catches a local verifier
    # pointed somewhere unexpected.
    model_id = getattr(scorer, "model_id", "")
    revision = getattr(scorer, "revision", "")
    expected_id = getattr(settings, "nli_model_id", model_id)
    expected_rev = getattr(settings, "nli_model_revision", revision)
    if model_id != expected_id or revision != expected_rev:
        return NLIReadiness(
            False, f"verifier serves {model_id}@{revision}, expected {expected_id}@{expected_rev}"
        )

    if entailed.scores.entailment < _MIN_ENTAILMENT:
        return NLIReadiness(
            False,
            f"probe entailment {entailed.scores.entailment:.2f} below {_MIN_ENTAILMENT}; "
            "the endpoint may not be an NLI classifier",
        )
    if contradicted.scores.contradiction < _MIN_CONTRADICTION:
        return NLIReadiness(
            False,
            f"probe contradiction {contradicted.scores.contradiction:.2f} "
            f"below {_MIN_CONTRADICTION}; label order may be wrong",
        )

    log.info(
        "nli_preflight_ok",
        model=model_id,
        revision=revision[:8],
        entailment=round(entailed.scores.entailment, 3),
        contradiction=round(contradicted.scores.contradiction, 3),
    )
    return NLIReadiness(True, "verifier ready", model_id, revision)


def _check_managed_pin(settings: object) -> NLIReadiness | None:
    """Refuse a managed endpoint that is not on the calibrated commit.

    Only the ``hf`` dialect needs this. The project's own service proves
    its identity in every response, so a mismatch is caught per call;
    the stock Hugging Face handler proves nothing, and the only place
    left to ask is the control plane.

    Run before the probe rather than after. The probe wakes a replica,
    and there is no reason to spend a cold start on an endpoint that is
    already known to be serving the wrong weights.

    Returns ``None`` when the check does not apply or passes, so the
    caller continues; returns a failed readiness when it does not.
    """
    if getattr(settings, "nli_mode", "local") != "remote":
        return None
    if getattr(settings, "nli_dialect", "contract") != "hf":
        return None

    endpoint = getattr(settings, "nli_endpoint", None)
    if not endpoint:
        return NLIReadiness(False, "nli_mode is 'remote' but no nli_endpoint is configured")

    key = getattr(settings, "nli_api_key", None)
    api_key = key.get_secret_value() if key is not None else None
    expected_id = getattr(settings, "nli_model_id", "")
    expected_rev = getattr(settings, "nli_model_revision", "")

    try:
        pin = verify_endpoint_pin(
            endpoint,
            api_key,
            expected_id,
            expected_rev,
            timeout=float(getattr(settings, "nli_timeout_seconds", 30.0)),
        )
    except Exception as exc:  # pragma: no cover - defensive
        return NLIReadiness(False, f"endpoint pin could not be read: {type(exc).__name__}")

    if not pin.verified:
        return NLIReadiness(False, f"endpoint pin rejected: {pin.detail}")
    return None
