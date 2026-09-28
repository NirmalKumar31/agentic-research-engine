"""Deployment configuration has to be safe by construction.

Two failures this pins. The live blueprint once enabled paid research
while leaving NLI_MODE=local and building an image with no torch in it,
so a run would have spent money and then withheld every claim because
verification was never possible. And the web image must stay small
enough for a 512MB plan, which means it must never acquire torch.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def blueprint(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text())


def env_of(spec: dict) -> dict[str, dict]:
    return {e["key"]: e for e in spec["services"][0]["envVars"]}


class TestTheWebImageStaysSmall:
    def test_the_web_extra_has_no_machine_learning_stack(self) -> None:
        """The measured verifier peak is about 1.3GB against a 512MB
        plan. torch here would not make it fit; it would only make the
        image too large to deploy."""
        extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "optional-dependencies"
        ]
        joined = " ".join(extras["web"]).lower()
        for banned in ("torch", "transformers", "safetensors", "sentencepiece"):
            assert banned not in joined

    def test_the_web_dockerfile_installs_only_that_extra(self) -> None:
        text = (ROOT / "Dockerfile.web").read_text()
        assert "[web]" in text
        assert "nli-local" not in text

    def test_a_local_verifier_extra_exists(self) -> None:
        """Local research needs a verifier; a bare install has none, and
        the quickstart used to imply otherwise."""
        extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "optional-dependencies"
        ]
        assert "nli-local" in extras
        assert any("torch" in d for d in extras["nli-local"])

    def test_the_cli_image_can_verify_locally(self) -> None:
        assert "nli-local" in (ROOT / "Dockerfile").read_text()

    def test_the_web_extra_can_still_measure_truncation(self) -> None:
        """The web image runs NLI_MODE=remote against a handler that
        reports no truncation flag, so the count is taken in-process.
        Without tokenizers the verifier withholds every claim rather
        than assume a premise survived intact -- correct, and a total
        outage. It is a 3MB pure-Rust wheel, not part of the ML stack
        the test above bans."""
        extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "optional-dependencies"
        ]
        assert any("tokenizers" in d for d in extras["web"])


class TestTheLiveBlueprintIsComplete:
    SPEC = "deploy/render-live.yaml"

    def test_it_configures_remote_verification(self) -> None:
        env = env_of(blueprint(self.SPEC))
        assert env["NLI_MODE"]["value"] == "remote"
        assert "NLI_ENDPOINT" in env and "NLI_API_KEY" in env

    def test_the_revision_is_the_calibrated_one(self) -> None:
        env = env_of(blueprint(self.SPEC))
        revision = env["NLI_MODEL_REVISION"]["value"]
        assert revision == "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
        assert len(revision) == 40

    def test_the_threshold_is_the_calibrated_one(self) -> None:
        env = env_of(blueprint(self.SPEC))
        assert env["NLI_SUPPORT_THRESHOLD"]["value"] == "0.98"

    def test_the_dialect_matches_the_deployed_endpoint(self) -> None:
        """The blueprint points at a managed Hugging Face endpoint. The
        default dialect expects this project's own service, which
        answers in a different shape entirely, so leaving it unset
        withholds every claim on a correctly configured endpoint."""
        env = env_of(blueprint(self.SPEC))
        assert env["NLI_DIALECT"]["value"] == "hf"

    def test_it_builds_the_small_web_image(self) -> None:
        assert blueprint(self.SPEC)["services"][0]["dockerfilePath"] == "./Dockerfile.web"

    def test_a_durable_quota_is_required(self) -> None:
        """Live research spends at three providers per run, so the cap
        must outlive a cold start."""
        env = env_of(blueprint(self.SPEC))
        assert env["DEMO_QUOTA_REQUIRED"]["value"] == "true"
        # Wired from the Key Value service rather than prompted: a value
        # nobody pastes is a value nobody pastes wrongly, and the two
        # cannot drift apart.
        source = env["DEMO_QUOTA_URL"]["fromService"]
        assert source["property"] == "connectionString"
        assert source["type"] == "keyvalue"

    def test_warm_up_waiting_covers_the_measured_cold_start(self) -> None:
        """The endpoint runs at minimum replicas 0, so every run after a
        quiet period pays a cold start. At 0 that run withholds every
        claim. The budget must clear the 49.2s measured on 2026-09-27
        with real margin, and stay well inside the run deadline it is
        spent from."""
        env = env_of(blueprint(self.SPEC))
        budget = float(env["NLI_SCALE_UP_TIMEOUT_SECONDS"]["value"])
        deadline = float(env["DEMO_MAX_RUNTIME_SECONDS"]["value"])
        assert budget >= 49.2 * 1.5
        assert budget < deadline / 2

    @pytest.mark.parametrize(
        "key",
        [
            "OPENAI_API_KEY",
            "TAVILY_API_KEY",
            "NLI_API_KEY",
            "NLI_ENDPOINT",
            "OPENAI_MODEL",
        ],
    )
    def test_secrets_and_deployment_values_are_prompted_never_committed(self, key: str) -> None:
        entry = env_of(blueprint(self.SPEC))[key]
        assert entry.get("sync") is False
        assert "value" not in entry, f"{key} carries a committed value"

    def test_per_run_ceilings_survive(self) -> None:
        env = env_of(blueprint(self.SPEC))
        assert env["MAX_CLOUD_COST_USD"]["value"] == "0.05"
        assert env["ALLOW_CLOUD_FALLBACK"]["value"] == "false"


class TestTheReplayBlueprintNeedsNothing:
    SPEC = "render.yaml"

    def test_live_research_is_off(self) -> None:
        env = env_of(blueprint(self.SPEC))
        assert env["LIVE_RESEARCH_ENABLED"]["value"] == "false"

    def test_it_declares_no_secrets(self) -> None:
        """The accepted replay release must keep deploying with zero
        credentials."""
        env = env_of(blueprint(self.SPEC))
        assert not [k for k, e in env.items() if e.get("sync") is False]


class TestSettingsAndExampleAgree:
    def test_the_documented_default_mode_is_the_actual_default(self) -> None:
        """These disagreed: the code defaulted to hybrid while
        .env.example and the quickstart said local, so a fresh checkout
        failed asking for a key for a mode nobody chose."""
        from agentic_research.config import LLMMode, Settings

        example = (ROOT / ".env.example").read_text()
        documented = next(
            line.split("=", 1)[1].strip()
            for line in example.splitlines()
            if line.startswith("LLM_MODE=")
        )
        assert Settings(_env_file=None).llm_mode is LLMMode(documented)

    def test_the_example_documents_the_verifier(self) -> None:
        example = (ROOT / ".env.example").read_text()
        for key in ("NLI_MODE", "NLI_MODEL_REVISION", "NLI_SUPPORT_THRESHOLD"):
            assert key in example

    def test_the_readme_anchor_the_ui_links_to_exists(self) -> None:
        """The footer linked to README.md#usage and no such heading
        existed."""
        app = (ROOT / "web" / "src" / "App.tsx").read_text()
        if "README.md#usage" in app:
            assert "\n## Usage\n" in (ROOT / "README.md").read_text()


class TestTheQuotaStoreIsDeclared:
    """The counter is only shared if something actually provisions it."""

    SPEC = "deploy/render-live.yaml"

    def _keyvalue(self) -> dict:
        services = blueprint(self.SPEC)["services"]
        stores = [s for s in services if s.get("type") == "keyvalue"]
        assert len(stores) == 1, f"expected one Key Value service, found {len(stores)}"
        return stores[0]

    def test_the_blueprint_provisions_the_store(self) -> None:
        assert self._keyvalue()["name"] == "agentic-research-quota"

    def test_the_web_service_points_at_that_store(self) -> None:
        env = env_of(blueprint(self.SPEC))
        assert env["DEMO_QUOTA_URL"]["fromService"]["name"] == self._keyvalue()["name"]

    def test_the_store_is_not_reachable_from_the_internet(self) -> None:
        assert self._keyvalue().get("ipAllowList") == []

    def test_a_free_store_is_not_described_as_durable(self) -> None:
        """Render: "Data persistence is not available for free Key Value
        instances." A free store shares the cap across replicas and
        loses it when the store itself restarts, so the comment must not
        promise otherwise."""
        import re

        # Comment wrapping splits sentences across lines, so the prose
        # is normalised before being searched rather than the comment
        # being reflowed to suit the test.
        raw = (ROOT / self.SPEC).read_text().lower()
        prose = re.sub(r"\s+", " ", raw.replace("#", " "))
        if self._keyvalue().get("plan") == "free":
            assert "persistence is not available" in prose
            assert "not durable across a restart" in prose


class TestDeployChecksUseReadiness:
    SPEC = "deploy/render-live.yaml"

    def test_the_health_check_path_is_readiness(self) -> None:
        """Liveness answers 200 on an instance configured for live
        research that cannot serve it. A deploy gated on that goes green
        and then refuses every visitor."""
        web = blueprint(self.SPEC)["services"][0]
        assert web["healthCheckPath"] == "/api/readiness"

    def test_auto_deploy_uses_the_current_key(self) -> None:
        web = blueprint(self.SPEC)["services"][0]
        assert web.get("autoDeployTrigger") == "off", (
            "autoDeploy is deprecated and a bare off parses as boolean false"
        )
        assert "autoDeploy" not in web


class TestTheReadmeMatchesTheBlueprint:
    """Counts in prose go stale silently. This one already had.

    The README said the live blueprint prompts for two keys while the
    blueprint prompted for six values, three of which are credentials.
    """

    SPEC = "deploy/render-live.yaml"
    SECRET_KEYS = {"OPENAI_API_KEY", "TAVILY_API_KEY", "NLI_API_KEY"}

    def _prompted(self) -> set[str]:
        env = env_of(blueprint(self.SPEC))
        return {k for k, v in env.items() if v.get("sync") is False}

    def test_the_credentials_are_exactly_the_three_named(self) -> None:
        assert self._prompted() & self.SECRET_KEYS == self.SECRET_KEYS

    def test_the_readme_states_the_right_number_of_credentials(self) -> None:
        readme = (ROOT / "README.md").read_text()
        assert "prompts for three\ncredentials" in readme or (
            "three\ncredentials" in readme or "three credentials" in readme
        ), "README no longer states three credentials"

    def test_the_quota_url_is_not_among_the_prompted_values(self) -> None:
        """Wired from the store, so a person never handles it."""
        assert "DEMO_QUOTA_URL" not in self._prompted()


class TestTheVersionMatchesTheRelease:
    """v0.2.0 was the accepted replay release. A live service reporting
    0.2.0 is claiming to be that build."""

    def test_package_and_module_versions_agree(self) -> None:
        import tomllib

        from agentic_research import __version__

        declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
        assert declared == __version__

    def test_the_live_release_is_not_the_replay_release(self) -> None:
        from agentic_research import __version__

        assert __version__ != "0.2.0", "live research shipped under the replay release's version"


class TestTheDeploymentEvidenceStaysTrue:
    """Prose describing a deployment goes stale the moment the
    deployment changes. These pin the numbers to the blueprint."""

    DOC = "examples/live-validation/DEPLOYMENT-EVIDENCE.md"
    SPEC = "deploy/render-live.yaml"

    def _text(self) -> str:
        import re

        raw = (ROOT / self.DOC).read_text()
        return re.sub(r"\s+", " ", raw)

    def test_it_exists(self) -> None:
        assert (ROOT / self.DOC).is_file()

    def test_the_daily_admission_count_matches_the_blueprint(self) -> None:
        from agentic_research.web.limits import runs_affordable

        env = env_of(blueprint(self.SPEC))
        expected = runs_affordable(
            int(env["DEMO_PROVIDER_REQUESTS_PER_DAY"]["value"]),
            int(env["MAX_PROVIDER_REQUESTS"]["value"]),
        )
        assert f"**{expected}** = " in self._text() or f"Daily admissions | **{expected}**" in (
            (ROOT / self.DOC).read_text()
        )

    def test_the_warm_up_budget_matches_the_blueprint(self) -> None:
        env = env_of(blueprint(self.SPEC))
        budget = env["NLI_SCALE_UP_TIMEOUT_SECONDS"]["value"]
        assert f"{budget}s" in self._text()

    def test_the_pinned_revision_matches(self) -> None:
        env = env_of(blueprint(self.SPEC))
        assert env["NLI_MODEL_REVISION"]["value"] in self._text()

    def test_the_free_store_is_not_called_durable(self) -> None:
        text = self._text().lower()
        assert "does **not** survive a restart" in text or "not survive a restart" in text
        assert "atomic and shared" in text

    def test_the_spend_limit_is_called_an_estimate(self) -> None:
        text = self._text().lower()
        assert "estimate" in text
        assert "not a billing cap" in text

    def test_it_does_not_claim_research_quality(self) -> None:
        """One run is not an evaluation, and the document has to say so
        rather than let a reader infer otherwise from a green table."""
        text = self._text().lower()
        assert "not a research-quality evaluation" in text
