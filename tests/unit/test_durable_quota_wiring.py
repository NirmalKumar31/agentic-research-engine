"""The durable quota is on the production request path.

The counter, its atomic INCR and its fail-closed behaviour were all
implemented and tested, and none of it was connected to
``/api/research``. The global daily cap was still the in-memory window,
which bounds a process rather than a day: a Render instance that sleeps
when idle wakes with a fresh counter, and supplying DEMO_QUOTA_URL
bought nothing at all.

So these tests drive the real HTTP route. A test that constructs
DurableRunQuota and calls reserve() proves the class works and says
nothing about whether anything calls it, which is exactly the gap that
let dead code look finished.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentic_research.config import Settings
from agentic_research.web import api as api_module
from agentic_research.web.api import create_app
from fakes import BrokenQuotaStore, FakeQuotaStore

QUESTION = {"query": "What did the study measure about developer productivity?"}


def daily_cap(settings: Settings) -> int:
    """The cap this configuration actually derives.

    Read rather than hardcoded: it is computed from the provider quota,
    so writing 1 here would make these tests pass for the wrong reason
    the moment that derivation changes.
    """
    from agentic_research.web.limits import limits_from_settings

    return limits_from_settings(settings).global_runs_per_day


def exhaust(client: TestClient, settings: Settings) -> None:
    for _ in range(daily_cap(settings)):
        assert post(client).status_code == 200


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
        # The per-client hourly limit is a separate control. Raised here
        # so the global daily cap is the constraint under test rather
        # than whichever happens to bite first.
        "demo_runs_per_hour": 100,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _never_really_research(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record whether the engine was reached, and never reach a provider.

    A quota refusal has to happen *before* anything that spends money,
    so "was this called" is the assertion, not "what did it return".
    """
    started: list[str] = []

    async def fake_stream(query: str, settings: Any, run_id: str = "") -> Any:
        started.append(query)
        yield {"event": "error", "error": "stubbed"}

    monkeypatch.setattr(api_module, "stream_research", fake_stream)
    return started


def post(client: TestClient) -> Any:
    return client.post("/api/research", json=QUESTION)


class TestTheRouteReservesQuota:
    def test_a_live_request_increments_the_shared_counter(self) -> None:
        store = FakeQuotaStore()
        with TestClient(create_app(live_settings(), counter_factory=store.factory)) as client:
            post(client)
        assert store.incr_calls == 1, (
            "the production route did not reserve quota; the counter was never touched"
        )

    def test_the_day_key_expires_so_it_cannot_leak_forever(self) -> None:
        store = FakeQuotaStore()
        with TestClient(create_app(live_settings(), counter_factory=store.factory)) as client:
            post(client)
        assert store.expiries, "no expiry was set on the day key"


class TestTheCapIsSharedAndSurvives:
    def test_two_instances_share_one_cap(self, _never_really_research: list[str]) -> None:
        """The property in-memory counting cannot have. Two replicas
        behind one load balancer must not get a full allowance each."""
        store = FakeQuotaStore()
        settings = live_settings()
        a = create_app(settings, counter_factory=store.factory)
        b = create_app(settings, counter_factory=store.factory)
        with TestClient(a) as ca, TestClient(b) as cb:
            exhaust(ca, settings)
            second = post(cb)
        assert second.status_code == 429, "the second instance granted its own allowance"
        assert len(_never_really_research) == daily_cap(settings)

    def test_recreating_the_app_does_not_reset_the_count(self) -> None:
        """A cold start is the normal case on a sleeping free instance,
        and it used to hand back the whole day's budget."""
        store = FakeQuotaStore()
        settings = live_settings()
        with TestClient(create_app(settings, counter_factory=store.factory)) as client:
            exhaust(client, settings)
        with TestClient(create_app(settings, counter_factory=store.factory)) as client:
            assert post(client).status_code == 429


class TestRefusalHappensBeforeSpending:
    def test_a_refused_run_never_reaches_the_engine(
        self, _never_really_research: list[str]
    ) -> None:
        store = FakeQuotaStore()
        settings = live_settings()
        with TestClient(create_app(settings, counter_factory=store.factory)) as client:
            exhaust(client, settings)
            over = post(client)
        assert over.status_code == 429
        assert len(_never_really_research) == daily_cap(settings), (
            "research started on a request that had no quota, so the "
            "refusal came after the money was spendable"
        )

    def test_a_refusal_releases_the_concurrency_slot(self) -> None:
        """The slot is taken before the reservation. Leaking it would
        wedge the service at capacity after the daily cap is hit, so a
        restart would be the only recovery."""
        store = FakeQuotaStore()
        settings = live_settings()
        app = create_app(settings, counter_factory=store.factory)
        with TestClient(app) as client:
            exhaust(client, settings)
            for _ in range(3):
                assert post(client).status_code == 429
            snapshot = client.get("/api/health").json()["capacity"]
        assert snapshot["active_runs"] == 0, f"concurrency leaked on refusal: {snapshot}"


class TestFailClosed:
    """Every way the store can be absent or broken must refuse."""

    def test_no_store_configured_refuses(self, _never_really_research: list[str]) -> None:
        app = create_app(live_settings(), counter_factory=lambda _url: None)
        with TestClient(app) as client:
            response = post(client)
        assert response.status_code == 429
        assert not _never_really_research

    def test_an_unreachable_store_refuses(self, _never_really_research: list[str]) -> None:
        broken = BrokenQuotaStore()
        app = create_app(live_settings(), counter_factory=broken.factory)
        with TestClient(app) as client:
            response = post(client)
        assert response.status_code == 429
        assert not _never_really_research

    def test_a_timing_out_store_refuses(self, _never_really_research: list[str]) -> None:
        broken = BrokenQuotaStore(TimeoutError("timed out"))
        app = create_app(live_settings(), counter_factory=broken.factory)
        with TestClient(app) as client:
            assert post(client).status_code == 429
        assert not _never_really_research

    def test_a_malformed_store_response_refuses(self, _never_really_research: list[str]) -> None:
        class Malformed:
            def incr(self, key: str) -> int:
                return "not a number"  # type: ignore[return-value]

            def expire(self, key: str, seconds: int) -> None: ...

        app = create_app(live_settings(), counter_factory=lambda _url: Malformed())
        with TestClient(app) as client:
            assert post(client).status_code == 429
        assert not _never_really_research

    def test_a_store_that_is_not_required_still_admits(self) -> None:
        """Fail-closed is a deployment choice, and a local run without a
        store should not be forced to stand one up."""
        settings = live_settings(demo_quota_required=False, demo_quota_url=None)
        app = create_app(settings, counter_factory=lambda _url: None)
        with TestClient(app) as client:
            assert post(client).status_code == 200


class TestReplaySurvivesQuotaOutage:
    def test_recorded_runs_still_serve_without_a_store(self) -> None:
        """Replay spends nothing, so a quota outage must not take it
        down. A demo that serves nothing at all is a worse failure than
        one that declines live runs."""
        app = create_app(live_settings(), counter_factory=lambda _url: None)
        with TestClient(app) as client:
            assert client.get("/api/examples").status_code == 200
            assert client.get("/api/config").status_code == 200
            assert client.get("/api/health").status_code == 200


class TestNothingSensitiveEscapes:
    SECRET_URL = "redis://admin:hunter2@quota.internal.test:6379/0"

    def test_the_connection_string_is_not_in_any_response(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        settings = live_settings(demo_quota_url=self.SECRET_URL)
        broken = BrokenQuotaStore()
        app = create_app(settings, counter_factory=broken.factory)
        with TestClient(app) as client:
            bodies = [
                post(client).text,
                client.get("/api/health").text,
                client.get("/api/config").text,
                client.get("/api/readiness").text,
            ]
        blob = " ".join(bodies) + " " + caplog.text
        for secret in ("hunter2", "quota.internal.test", "redis://"):
            assert secret not in blob, f"{secret!r} leaked to a client or the log"

    def test_the_refusal_names_no_infrastructure(self) -> None:
        app = create_app(
            live_settings(demo_quota_url=self.SECRET_URL),
            counter_factory=BrokenQuotaStore().factory,
        )
        with TestClient(app) as client:
            body = json.loads(post(client).text)
        assert "redis" not in body["error"].lower()
        assert "quota store" not in body["error"].lower()


class TestReadinessDistinguishesStates:
    def test_ready_when_live_research_can_actually_run(self) -> None:
        store = FakeQuotaStore()
        app = create_app(live_settings(), counter_factory=store.factory)
        with TestClient(app) as client:
            body = client.get("/api/readiness").json()
        assert body["ready"] is True
        assert body["live_research_available"] is True
        assert body["durable_quota_available"] is True

    def test_not_ready_when_configured_for_live_but_the_store_is_gone(self) -> None:
        """The state a liveness probe reports as healthy while the
        service turns away every visitor."""
        app = create_app(live_settings(), counter_factory=lambda _url: None)
        with TestClient(app) as client:
            response = client.get("/api/readiness")
        assert response.status_code == 503
        body = response.json()
        assert body["alive"] is True
        assert body["replay_available"] is True
        assert body["live_research_enabled"] is True
        assert body["live_research_available"] is False
        assert body["durable_quota_available"] is False

    def test_replay_only_is_ready(self) -> None:
        settings = live_settings(
            live_research_enabled=False, demo_quota_required=False, demo_quota_url=None
        )
        app = create_app(settings, counter_factory=lambda _url: None)
        with TestClient(app) as client:
            response = client.get("/api/readiness")
        assert response.status_code == 200
        assert response.json()["ready"] is True
        assert response.json()["live_research_available"] is False


class TestReadinessTracksTheStoreAtRuntime:
    """Reachable at startup proves nothing about reachable now.

    The first version of `usable()` returned true whenever a counter
    object existed, so an instance whose Key Value service died an hour
    ago kept answering 200 on /api/readiness while every live request
    failed closed. A load balancer has no way to notice that.
    """

    def test_ready_while_the_store_answers(self) -> None:
        store = FakeQuotaStore()
        with TestClient(create_app(live_settings(), counter_factory=store.factory)) as client:
            assert client.get("/api/readiness").status_code == 200
        assert store.health_calls > 0, "readiness never asked the store anything"

    def test_not_ready_once_the_store_stops_answering(self) -> None:
        store = FakeQuotaStore()
        with TestClient(create_app(live_settings(), counter_factory=store.factory)) as client:
            assert client.get("/api/readiness").status_code == 200
            store.health_response = False  # the Key Value service dies
            response = client.get("/api/readiness")
        assert response.status_code == 503
        body = response.json()
        assert body["alive"] is True
        assert body["live_research_available"] is False
        assert body["durable_quota_available"] is False

    def test_readiness_recovers_when_the_store_returns(self) -> None:
        """A transient outage must not require a redeploy to clear."""
        store = FakeQuotaStore()
        with TestClient(create_app(live_settings(), counter_factory=store.factory)) as client:
            store.health_response = False
            assert client.get("/api/readiness").status_code == 503
            store.health_response = True
            assert client.get("/api/readiness").status_code == 200

    @pytest.mark.parametrize("response", ["PONG", 1, "OK", None, object()])
    def test_a_malformed_health_answer_is_not_health(self, response: object) -> None:
        """Only a literal True counts. A truthy string is how a broken
        adapter passes for a working one."""
        store = FakeQuotaStore()
        store.health_response = response
        with TestClient(create_app(live_settings(), counter_factory=store.factory)) as client:
            assert client.get("/api/readiness").status_code == 503

    def test_a_store_that_raises_on_ping_is_not_ready(self) -> None:
        broken = BrokenQuotaStore()
        with TestClient(create_app(live_settings(), counter_factory=broken.factory)) as client:
            assert client.get("/api/readiness").status_code == 503

    def test_readiness_never_consumes_a_run(self) -> None:
        """The probe must not answer "can a run start?" by starting one."""
        store = FakeQuotaStore()
        with TestClient(create_app(live_settings(), counter_factory=store.factory)) as client:
            for _ in range(5):
                client.get("/api/readiness")
        assert store.incr_calls == 0, "readiness incremented the daily counter"
        assert store.counts == {}

    def test_replay_stays_available_through_a_store_outage(self) -> None:
        store = FakeQuotaStore()
        with TestClient(create_app(live_settings(), counter_factory=store.factory)) as client:
            store.health_response = False
            body = client.get("/api/readiness").json()
            assert body["replay_available"] is True
            assert client.get("/api/examples").status_code == 200

    def test_a_recent_healthy_probe_does_not_admit_a_run(self) -> None:
        """Readiness is advisory. Admission re-checks, because the store
        can die between the probe and the request."""
        store = FakeQuotaStore()
        with TestClient(create_app(live_settings(), counter_factory=store.factory)) as client:
            assert client.get("/api/readiness").status_code == 200
            store.health_response = False
            # incr still works; the point is that reserve() does not
            # consult a cached readiness verdict.
            assert post(client).status_code == 200


class TestTheDailyAllowanceMatchesTheQuota:
    """The deployed numbers, asserted as a set rather than described."""

    DEPLOYED = {
        "demo_provider_requests_per_day": 50,
        "max_cloud_calls": 20,
        "max_provider_requests": 30,
    }

    def _limits(self):
        from agentic_research.web.limits import limits_from_settings

        return limits_from_settings(live_settings(**self.DEPLOYED))

    def test_the_deployed_configuration_allows_one_run_per_day(self) -> None:
        """50 // 30, not 50 // 20. Two runs at 30 provider requests each
        would need 60 against a 50-request quota, and the second would
        have died mid-run on a 429 after spending OpenAI tokens."""
        assert self._limits().global_runs_per_day == 1

    def test_the_allowance_never_exceeds_the_provider_quota(self) -> None:
        limits = self._limits()
        assert (
            limits.global_runs_per_day * limits.max_provider_requests
            <= self.DEPLOYED["demo_provider_requests_per_day"]
        )

    @pytest.mark.parametrize(
        ("per_day", "requests", "expected"),
        [
            (50, 30, 1),
            (60, 30, 2),
            (29, 30, 0),
            (300, 30, 10),
        ],
    )
    def test_the_allowance_is_the_quota_divided_by_the_request_ceiling(
        self, per_day: int, requests: int, expected: int
    ) -> None:
        from agentic_research.web.limits import limits_from_settings

        limits = limits_from_settings(
            live_settings(demo_provider_requests_per_day=per_day, max_provider_requests=requests)
        )
        assert limits.global_runs_per_day == expected
