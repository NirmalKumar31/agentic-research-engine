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

    def test_it_builds_the_small_web_image(self) -> None:
        assert blueprint(self.SPEC)["services"][0]["dockerfilePath"] == "./Dockerfile.web"

    def test_a_durable_quota_is_required(self) -> None:
        """Live research spends at three providers per run, so the cap
        must outlive a cold start."""
        env = env_of(blueprint(self.SPEC))
        assert env["DEMO_QUOTA_REQUIRED"]["value"] == "true"
        assert env["DEMO_QUOTA_URL"].get("sync") is False

    def test_warm_up_waiting_is_off_until_acceptance(self) -> None:
        env = env_of(blueprint(self.SPEC))
        assert env["NLI_SCALE_UP_TIMEOUT_SECONDS"]["value"] == "0"

    @pytest.mark.parametrize(
        "key",
        [
            "OPENAI_API_KEY",
            "TAVILY_API_KEY",
            "NLI_API_KEY",
            "DEMO_QUOTA_URL",
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
