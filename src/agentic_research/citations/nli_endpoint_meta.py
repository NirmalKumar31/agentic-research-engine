"""Ask Hugging Face which checkpoint an endpoint actually serves.

The project's own scoring service echoes its model id and revision in
every response, so the client can refuse a mismatch per request. A
managed Inference Endpoint running the stock text-classification
handler cannot: it returns labels and scores and says nothing about
what produced them.

That gap matters because the 0.98 support threshold is a property of
one commit. An endpoint edited to a newer revision -- a one-click
change in the Hugging Face console -- would keep answering, keep
returning a clean three-way distribution, and quietly invalidate every
published support decision.

So the pin is checked out of band, against the control plane rather
than the inference path, before a run starts. The token needs only read
access; the call costs nothing and wakes no replica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_research.observability import get_logger

log = get_logger(__name__)

_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
_ENDPOINTS_URL = "https://api.endpoints.huggingface.cloud/v2/endpoint"

# The stock handler for sequence-pair classification. A zero-shot
# endpoint also wraps an NLI model, but it builds the hypothesis from a
# template and normalises over two labels, so its numbers are not the
# three-way distribution the threshold was calibrated on.
_REQUIRED_TASK = "text-classification"


@dataclass(frozen=True)
class EndpointPin:
    verified: bool
    detail: str
    repository: str = ""
    revision: str = ""
    task: str = ""
    state: str = ""
    ready_replicas: int = 0
    names_seen: tuple[str, ...] = field(default_factory=tuple)


def _normalise(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def verify_endpoint_pin(
    endpoint: str,
    api_key: str | None,
    expected_repository: str,
    expected_revision: str,
    *,
    timeout: float = 30.0,
) -> EndpointPin:
    """Confirm the endpoint serves the pinned repository and commit.

    Never raises. An unverifiable pin is a reason to decline the run,
    not an exception for the caller to unpick.
    """
    if not api_key:
        return EndpointPin(False, "no API key, so the endpoint's pin cannot be read")

    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - httpx is a core dependency
        return EndpointPin(False, f"httpx is not installed: {exc}")

    headers = {"Authorization": f"Bearer {api_key}"}
    wanted = _normalise(endpoint)

    try:
        with httpx.Client(timeout=timeout) as client:
            who = client.get(_WHOAMI_URL, headers=headers)
            if who.status_code != 200:
                return EndpointPin(
                    False, f"Hugging Face rejected the token (HTTP {who.status_code})"
                )
            identity = who.json()
            namespaces = [identity.get("name")]
            namespaces += [org.get("name") for org in identity.get("orgs", []) or []]
            namespaces = [n for n in namespaces if n]
            if not namespaces:
                return EndpointPin(False, "token resolves to no account or organisation")

            seen: list[str] = []
            match: dict[str, Any] | None = None
            for namespace in namespaces:
                listing = client.get(f"{_ENDPOINTS_URL}/{namespace}", headers=headers)
                if listing.status_code != 200:
                    # A token scoped to one namespace legitimately cannot
                    # read another. Only a total failure is fatal.
                    log.debug(
                        "nli_endpoint_list_denied",
                        namespace=namespace,
                        status=listing.status_code,
                    )
                    continue
                for item in listing.json().get("items", []) or []:
                    name = str(item.get("name", ""))
                    seen.append(name)
                    if _normalise(item.get("status", {}).get("url", "")) == wanted:
                        match = item
                        break
                if match is not None:
                    break
    except Exception as exc:
        return EndpointPin(False, f"could not reach Hugging Face: {type(exc).__name__}")

    if match is None:
        return EndpointPin(
            False,
            "no endpoint owned by this token serves "
            f"{endpoint}; the token may belong to another account",
            names_seen=tuple(seen),
        )

    model = match.get("model", {}) or {}
    status = match.get("status", {}) or {}
    repository = str(model.get("repository", ""))
    revision = str(model.get("revision", ""))
    task = str(model.get("task", ""))
    state = str(status.get("state", ""))
    try:
        ready = int(status.get("readyReplica") or 0)
    except (TypeError, ValueError):
        ready = 0

    if repository != expected_repository:
        return EndpointPin(
            False,
            f"endpoint serves {repository!r}, expected {expected_repository!r}",
            repository,
            revision,
            task,
            state,
            ready,
            tuple(seen),
        )
    if revision != expected_revision:
        return EndpointPin(
            False,
            f"endpoint is pinned to {revision[:12] or '(none)'}, "
            f"expected {expected_revision[:12]}; the 0.98 threshold was "
            "calibrated against the expected commit only",
            repository,
            revision,
            task,
            state,
            ready,
            tuple(seen),
        )
    if task != _REQUIRED_TASK:
        return EndpointPin(
            False,
            f"endpoint task is {task!r}, expected {_REQUIRED_TASK!r}; "
            "zero-shot normalises over two labels, not three",
            repository,
            revision,
            task,
            state,
            ready,
            tuple(seen),
        )

    log.info(
        "nli_endpoint_pin_ok",
        repository=repository,
        revision=revision[:8],
        task=task,
        state=state,
    )
    return EndpointPin(
        True,
        "endpoint serves the pinned checkpoint",
        repository,
        revision,
        task,
        state,
        ready,
        tuple(seen),
    )
