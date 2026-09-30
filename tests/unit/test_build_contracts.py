"""Contracts between the repository and the things that build from it.

Both failures these cover reached a pushed branch. A root pricing.toml
was removed once it turned out to duplicate the packaged copy, but two
Dockerfiles still copied it, so every image build failed on a file that
no longer existed. The pinned model id and revision moved into nli_pin
so config and the client could not drift apart, and a CI step still
read them out of config with a regex that then matched nothing --
surfacing as AttributeError on None rather than as the configuration
error it was.

Neither is visible from a test run or a local check. They are only
visible to docker build and to the workflow, which is why they are
asserted here instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILES = ("Dockerfile", "Dockerfile.web")


def copied_paths(dockerfile: str) -> list[str]:
    """Build-context sources a Dockerfile copies in.

    Stage-to-stage copies are excluded: their source is another stage's
    filesystem, not this repository.
    """
    out: list[str] = []
    for raw in (ROOT / dockerfile).read_text().splitlines():
        line = raw.strip()
        if not line.upper().startswith("COPY "):
            continue
        parts = line.split()[1:]
        if any(p.startswith("--from=") for p in parts):
            continue
        parts = [p for p in parts if not p.startswith("--")]
        # Everything but the destination is a source.
        out.extend(parts[:-1])
    return out


@pytest.mark.parametrize("dockerfile", DOCKERFILES)
def test_every_copied_path_exists(dockerfile: str) -> None:
    """A COPY of a deleted file fails the build and nothing else."""
    missing = [
        src
        for src in copied_paths(dockerfile)
        if "*" not in src and src not in (".", "./") and not (ROOT / src).exists()
    ]
    assert not missing, f"{dockerfile} copies paths that do not exist: {missing}"


@pytest.mark.parametrize("dockerfile", DOCKERFILES)
def test_pricing_is_not_copied_from_the_build_context(dockerfile: str) -> None:
    """It ships inside the wheel as agentic_research/pricing.toml. A
    second copy at the image root is the duplicate that went stale."""
    assert not [src for src in copied_paths(dockerfile) if src.endswith("pricing.toml")]


class TestTheWorkflowCanReadThePin:
    """The NLI job caches weights by revision, so it parses the pin out
    of the source rather than repeating it."""

    def test_the_pin_module_holds_literals(self) -> None:
        text = (ROOT / "src" / "agentic_research" / "citations" / "nli_pin.py").read_text()
        for const in ("NLI_DEFAULT_MODEL_ID", "NLI_DEFAULT_REVISION"):
            assert re.search(rf'^{const} = "([^"]+)"', text, re.M), (
                f"{const} is no longer a plain literal; the workflow step "
                "that reads it will match nothing"
            )

    def test_the_workflow_reads_the_module_that_holds_them(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        step = workflow[workflow.index("Read the pinned model revision") :][:1200]
        assert "nli_pin.py" in step
        assert "config.py" not in step

    def test_the_revision_is_a_full_commit(self) -> None:
        from agentic_research.citations.nli_pin import NLI_DEFAULT_REVISION

        assert len(NLI_DEFAULT_REVISION) == 40


def test_the_workflow_builds_every_branch_by_default() -> None:
    """A branch pushed for review with no matching prefix carries no
    remote status, and a green local run gets mistaken for a verified
    one -- which is how the two failures above reached a push.

    It then happened a third time, to `quality/*`. This test passed
    throughout, because it checked that three known prefixes were on
    the allowlist rather than that the branch being pushed was -- and
    the prefix it could not have known about was the one that broke.

    An allowlist cannot be tested for the case it is missing. So the
    trigger is now a denylist, and this asserts that shape: every
    branch builds except throwaway prefixes, which fails in the safe
    direction when a new convention appears.
    """
    spec = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    push = spec[True]["push"]
    assert "branches" not in push, (
        "an allowlist of prefixes cannot cover a convention nobody has "
        "invented yet; three branches have already been pushed without CI"
    )
    ignored = push["branches-ignore"]
    assert ignored, "an empty ignore list is indistinguishable from no trigger at all"
    for throwaway in ("wip/**", "scratch/**", "tmp/**"):
        assert throwaway in ignored
    # Non-vacuity: the prefixes that have historically been used for real
    # work must not be ignored.
    for real in ("main", "feat/**", "fix/**", "quality/**", "release/**"):
        assert real not in ignored
