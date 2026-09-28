"""Two deployments must not spend one allowance.

A staging copy of this service and the production one are the same
image with the same code, so they compute the same counter key. Point
both at one Key Value store -- by intent, or by pasting the wrong
connection string once -- and a staging run silently consumes a
production run for the day. The failure is invisible: nothing errors,
the day's allowance is just short.

``DEMO_QUOTA_NAMESPACE`` separates them. These tests pin the two
properties that make it worth having:

* it *isolates* -- two namespaces against one store do not see each
  other's count; and
* it *did not change production* -- an unset namespace produces the
  exact key a deployment predating the setting is already counting
  against, so merging this does not reset a live counter mid-day.

Both directions matter. A namespace that isolated by accident, because
each app happened to hold its own counter object, would pass a weaker
test and protect nothing.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agentic_research.config import Settings
from agentic_research.web import api as api_module
from agentic_research.web.api import create_app
from agentic_research.web.durable_quota import DurableRunQuota
from fakes import FakeQuotaStore

STAGING = "v12-staging"
QUESTION = {"query": "What did the study measure about developer productivity?"}


class SharedStore:
    """One counter, handed to every quota that asks. Redis, in a dict."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def incr(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def expire(self, key: str, seconds: int) -> None:
        pass

    def healthy(self) -> bool:
        return True


class TestTheKeyShape:
    def test_an_unset_namespace_leaves_the_production_key_untouched(self) -> None:
        """The compatibility property, written as the literal format.

        Production is counting against this exact string today. If the
        default ever grows a segment, a deploy would reset the day's
        usage to zero on the way in -- the one moment the cap is most
        load-bearing.
        """
        day = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
        assert DurableRunQuota.key(day) == "are:live-runs:2026-09-28"

    def test_a_namespace_is_inserted_before_the_date(self) -> None:
        day = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
        assert DurableRunQuota.key(day, namespace=STAGING) == f"are:live-runs:{STAGING}:2026-09-28"

    def test_the_date_is_always_the_last_segment(self) -> None:
        """Why the ordering is not cosmetic: it is what makes the
        namespace unable to address another namespace's key."""
        for ns in ("", STAGING, "prod", "a"):
            assert DurableRunQuota.key(namespace=ns).endswith(
                datetime.now(UTC).strftime("%Y-%m-%d")
            )

    def test_no_permitted_namespace_can_spell_another_deployments_key(self) -> None:
        """The forgery check, over values the validator actually admits.

        A namespace that could contain a colon could be set to
        ``2026-09-28`` in one deployment and reach the un-namespaced
        production key for that day. The validator forbids the colon;
        this asserts the consequence rather than trusting it.
        """
        days = [datetime(2026, 9, d, tzinfo=UTC) for d in (27, 28, 29)]
        # Includes a namespace that is itself a valid date string --
        # permitted by the character class, and still harmless.
        namespaces = ["", STAGING, "prod", "2026-09-28", "a", "0"]
        keys = [(ns, DurableRunQuota.key(d, namespace=ns)) for ns in namespaces for d in days]
        assert len({k for _, k in keys}) == len(keys), "two configurations collided on one key"


class TestIsolationAgainstOneStore:
    def test_two_namespaces_do_not_spend_each_others_allowance(self) -> None:
        store = SharedStore()
        prod = DurableRunQuota(store, limit=1, required=True, namespace="")
        staging = DurableRunQuota(store, limit=1, required=True, namespace=STAGING)

        assert staging.reserve().allowed
        assert prod.reserve().allowed, "a staging run consumed production's only slot"

    def test_one_namespace_still_shares_one_allowance(self) -> None:
        """Non-vacuity. Without this, the test above would pass just as
        happily against an implementation where nothing is shared at
        all -- which would break the cap instead of scoping it.
        """
        store = SharedStore()
        a = DurableRunQuota(store, limit=1, required=True, namespace=STAGING)
        b = DurableRunQuota(store, limit=1, required=True, namespace=STAGING)

        assert a.reserve().allowed
        assert not b.reserve().allowed, "the shared cap stopped being shared"

    def test_the_namespace_reaches_the_store(self) -> None:
        store = SharedStore()
        DurableRunQuota(store, limit=1, required=True, namespace=STAGING).reserve()
        assert list(store.counts) == [DurableRunQuota.key(namespace=STAGING)], (
            "reserve() ignored the namespace it was constructed with"
        )


class TestTheSettingIsValidated:
    def test_the_default_is_empty(self) -> None:
        """Production behaviour is the behaviour you get by not setting
        anything."""
        assert Settings(_env_file=None).demo_quota_namespace == ""  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "value",
        [
            "prod:2026-09-28",  # the forgery the separator would allow
            "are:live-runs",
            "V12-Staging",  # case folding is locale-dependent; refuse it
            "v12 staging",
            "-leading-hyphen",
            "x" * 33,
            "staging/1",
            # Dotless i, written as an escape: the literal character is
            # ambiguous on sight, which is the point of refusing it.
            "stag\u0131ng",
        ],
    )
    def test_it_refuses_anything_that_is_not_a_plain_label(self, value: str) -> None:
        with pytest.raises(ValidationError):
            Settings(demo_quota_namespace=value, _env_file=None)  # type: ignore[call-arg]

    @pytest.mark.parametrize("value", ["", "   ", "\t"])
    def test_blank_means_no_namespace(self, value: str) -> None:
        assert Settings(demo_quota_namespace=value, _env_file=None).demo_quota_namespace == ""  # type: ignore[call-arg]

    @pytest.mark.parametrize("value", [STAGING, "prod", "a", "0", "x" * 32])
    def test_it_accepts_a_plain_label(self, value: str) -> None:
        assert (
            Settings(demo_quota_namespace=value, _env_file=None).demo_quota_namespace == value  # type: ignore[call-arg]
        )

    def test_the_permitted_shape_cannot_contain_the_key_separator(self) -> None:
        """Stated once, as the invariant the key format depends on."""
        pattern = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
        assert not pattern.fullmatch("a:b")


def live_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_mode": "local",
        "demo_mode": True,
        "live_research_enabled": True,
        "nli_mode": "remote",
        "nli_endpoint": "https://nli.test.invalid/score",
        "nli_api_key": "hf-test-placeholder",
        "demo_quota_url": "redis://quota.test.invalid:6379/0",
        "demo_quota_required": True,
        "tavily_api_key": "tvly-test-key",
        "demo_runs_per_hour": 100,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _never_really_research(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_stream(query: str, settings: Any, run_id: str = "") -> Any:
        yield {"event": "error", "error": "stubbed"}

    monkeypatch.setattr(api_module, "stream_research", fake_stream)


class TestTheDeploymentActuallyUsesIt:
    """The setting has to be on the request path, not merely defined.

    A namespace field that ``create_app`` never forwards is the same
    class of defect as the quota that was implemented and never called.
    """

    def test_the_configured_namespace_appears_in_the_key_the_store_sees(self) -> None:
        store = FakeQuotaStore()
        settings = live_settings(demo_quota_namespace=STAGING)
        with TestClient(create_app(settings, counter_factory=store.factory)) as client:
            client.post("/api/research", json=QUESTION)
        assert list(store.counts) == [DurableRunQuota.key(namespace=STAGING)], (
            "create_app did not forward DEMO_QUOTA_NAMESPACE to the quota"
        )

    def test_without_it_the_key_is_the_one_production_already_uses(self) -> None:
        store = FakeQuotaStore()
        with TestClient(create_app(live_settings(), counter_factory=store.factory)) as client:
            client.post("/api/research", json=QUESTION)
        assert list(store.counts) == [DurableRunQuota.key()]

    def test_staging_and_production_against_one_store_keep_separate_days(self) -> None:
        """The whole point, end to end: the real route, one store, two
        namespaces, and neither eats the other's allowance."""
        store = FakeQuotaStore()
        prod = live_settings()
        staging = live_settings(demo_quota_namespace=STAGING)
        cap = _cap(prod)

        with TestClient(create_app(staging, counter_factory=store.factory)) as cs:
            for _ in range(cap):
                assert cs.post("/api/research", json=QUESTION).status_code == 200
            assert cs.post("/api/research", json=QUESTION).status_code == 429

        with TestClient(create_app(prod, counter_factory=store.factory)) as cp:
            assert cp.post("/api/research", json=QUESTION).status_code == 200, (
                "production was refused because staging had exhausted the shared key"
            )


def _cap(settings: Settings) -> int:
    from agentic_research.web.limits import limits_from_settings

    return limits_from_settings(settings).global_runs_per_day
