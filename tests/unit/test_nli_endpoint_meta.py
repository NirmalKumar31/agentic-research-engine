"""The pin is checked against the control plane, not the inference path.

A managed endpoint on the stock handler answers every request the same
way whatever checkpoint it holds: three labels and three scores. The
0.98 threshold is a property of one commit, so an endpoint edited to a
newer revision would keep working, keep looking healthy, and quietly
invalidate every support decision built on it.

The only place that difference is visible is the Hugging Face API, so
that is where it is asked, before the run spends anything.
"""

from __future__ import annotations

import httpx
import respx

from agentic_research.citations.nli_endpoint_meta import verify_endpoint_pin

WHOAMI = "https://huggingface.co/api/whoami-v2"
LIST = "https://api.endpoints.huggingface.cloud/v2/endpoint/tester"
ENDPOINT = "https://abc123.endpoints.huggingface.cloud"
REPO = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"


def identity(orgs: list[str] | None = None) -> dict:
    return {"name": "tester", "orgs": [{"name": o} for o in (orgs or [])]}


def listing(
    *,
    url: str = ENDPOINT,
    repository: str = REPO,
    revision: str = REVISION,
    task: str = "text-classification",
) -> dict:
    return {
        "items": [
            {
                "name": "deberta-nli",
                "model": {"repository": repository, "revision": revision, "task": task},
                "status": {"url": url, "state": "running", "readyReplica": 1},
            }
        ]
    }


def check(**kwargs: object):
    return verify_endpoint_pin(ENDPOINT, "token", REPO, REVISION, **kwargs)  # type: ignore[arg-type]


@respx.mock
def test_a_matching_endpoint_is_accepted() -> None:
    respx.get(WHOAMI).mock(return_value=httpx.Response(200, json=identity()))
    respx.get(LIST).mock(return_value=httpx.Response(200, json=listing()))
    pin = check()
    assert pin.verified, pin.detail
    assert pin.revision == REVISION
    assert pin.state == "running"
    assert pin.ready_replicas == 1


@respx.mock
def test_a_different_revision_is_refused() -> None:
    """The case this exists for: same repository, newer weights."""
    respx.get(WHOAMI).mock(return_value=httpx.Response(200, json=identity()))
    respx.get(LIST).mock(return_value=httpx.Response(200, json=listing(revision="f" * 40)))
    pin = check()
    assert not pin.verified
    assert "calibrated" in pin.detail


@respx.mock
def test_a_different_repository_is_refused() -> None:
    respx.get(WHOAMI).mock(return_value=httpx.Response(200, json=identity()))
    respx.get(LIST).mock(return_value=httpx.Response(200, json=listing(repository="other/model")))
    pin = check()
    assert not pin.verified
    assert "other/model" in pin.detail


@respx.mock
def test_zero_shot_is_refused() -> None:
    """Zero-shot wraps the same NLI model but builds the hypothesis from
    a template and normalises over two labels. Its numbers are not the
    three-way distribution 0.98 was measured against."""
    respx.get(WHOAMI).mock(return_value=httpx.Response(200, json=identity()))
    respx.get(LIST).mock(
        return_value=httpx.Response(200, json=listing(task="zero-shot-classification"))
    )
    pin = check()
    assert not pin.verified
    assert "zero-shot" in pin.detail


@respx.mock
def test_an_endpoint_the_token_does_not_own_is_refused() -> None:
    respx.get(WHOAMI).mock(return_value=httpx.Response(200, json=identity()))
    respx.get(LIST).mock(
        return_value=httpx.Response(
            200, json=listing(url="https://other.endpoints.huggingface.cloud")
        )
    )
    pin = check()
    assert not pin.verified
    assert "another account" in pin.detail


@respx.mock
def test_a_trailing_slash_still_matches() -> None:
    respx.get(WHOAMI).mock(return_value=httpx.Response(200, json=identity()))
    respx.get(LIST).mock(return_value=httpx.Response(200, json=listing(url=ENDPOINT + "/")))
    assert check().verified


@respx.mock
def test_a_rejected_token_is_refused() -> None:
    respx.get(WHOAMI).mock(return_value=httpx.Response(401, json={"error": "unauthorized"}))
    pin = check()
    assert not pin.verified
    assert "401" in pin.detail


@respx.mock
def test_an_unreachable_api_is_refused() -> None:
    """Unverifiable is not the same as verified."""
    respx.get(WHOAMI).mock(side_effect=httpx.ConnectError("no route"))
    pin = check()
    assert not pin.verified
    assert "could not reach" in pin.detail


def test_no_key_is_refused() -> None:
    pin = verify_endpoint_pin(ENDPOINT, None, REPO, REVISION)
    assert not pin.verified


@respx.mock
def test_an_organisation_namespace_is_searched() -> None:
    """A token whose personal namespace holds nothing must still find an
    endpoint owned by an organisation it belongs to."""
    respx.get(WHOAMI).mock(return_value=httpx.Response(200, json=identity(["acme"])))
    respx.get(LIST).mock(return_value=httpx.Response(200, json={"items": []}))
    respx.get("https://api.endpoints.huggingface.cloud/v2/endpoint/acme").mock(
        return_value=httpx.Response(200, json=listing())
    )
    assert check().verified


@respx.mock
def test_a_namespace_the_token_cannot_read_is_skipped() -> None:
    """A fine-grained token scoped to one namespace legitimately gets
    403 on another. That must not mask an endpoint it can see."""
    respx.get(WHOAMI).mock(return_value=httpx.Response(200, json=identity(["acme"])))
    respx.get(LIST).mock(return_value=httpx.Response(403, json={"error": "forbidden"}))
    respx.get("https://api.endpoints.huggingface.cloud/v2/endpoint/acme").mock(
        return_value=httpx.Response(200, json=listing())
    )
    assert check().verified
