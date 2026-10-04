"""The committed hash lock stays honest about what it covers and what it
does not, and stays consistent with the ranges pyproject.toml declares.

This cannot prove the lock *installs* -- that needs the exact platform
it targets, which is CI's "Install from the hash-pinned lock" step, not
this machine. What it can prove: the file is well-formed, every pinned
version satisfies pyproject's own declared range, torch is nowhere in
it on purpose, and the generator script that produces it still points
at the platform CI actually runs on.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "requirements-lock.txt"

_PIN_LINE = re.compile(r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.!+-]+) \\$")
_HASH_LINE = re.compile(r"^ {4}--hash=sha256:[0-9a-f]{64}$")


def _parse_lock() -> dict[str, str]:
    lines = [ln for ln in LOCK_PATH.read_text().splitlines() if ln and not ln.startswith("#")]
    pins: dict[str, str] = {}
    i = 0
    while i < len(lines):
        m = _PIN_LINE.match(lines[i])
        assert m, f"not a pin line: {lines[i]!r}"
        name, version = m.group(1), m.group(2)
        assert _HASH_LINE.match(lines[i + 1]), (
            f"missing/malformed hash after {name}: {lines[i + 1]!r}"
        )
        pins[name.lower()] = version
        i += 2
    return pins


class TestTheLockFileIsWellFormed:
    def test_it_exists(self) -> None:
        assert LOCK_PATH.is_file()

    def test_every_pin_has_exactly_one_sha256_hash(self) -> None:
        pins = _parse_lock()
        assert len(pins) > 50, "suspiciously few pins -- did generation truncate?"

    def test_the_project_itself_is_not_pinned_in_it(self) -> None:
        """Installed separately (`-e .` or the built wheel); pinning it
        here would fight with that install rather than support it."""
        pins = _parse_lock()
        assert "agentic-research-engine" not in pins


class TestTorchIsDeliberatelyAbsent:
    """nli-local is excluded by design -- see the lock's own header and
    the generator script's HEADER constant. This is the check that a
    future edit does not quietly pull it back in."""

    def test_no_torch_family_package_is_pinned(self) -> None:
        pins = _parse_lock()
        for banned in (
            "torch",
            "transformers",
            "safetensors",
            "sentencepiece",
            "nvidia-cublas-cu12",
        ):
            assert banned not in pins, f"{banned} should not be in the hash lock"


class TestTheLockAgreesWithPyproject:
    """Catches the drift this kind of file always eventually suffers:
    someone bumps a range in pyproject.toml and forgets to regenerate."""

    def test_every_core_dependency_is_pinned_within_its_declared_range(self) -> None:
        from packaging.requirements import Requirement

        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
        pins = _parse_lock()
        deps = pyproject["project"]["dependencies"]
        unpinned = []
        for raw in deps:
            req = Requirement(raw)
            name = req.name.lower()
            pinned_version = pins.get(name)
            if pinned_version is None:
                unpinned.append(name)
                continue
            assert req.specifier.contains(pinned_version, prereleases=True), (
                f"{name}: lock pins {pinned_version}, outside pyproject's range {req.specifier}"
            )
        assert not unpinned, f"declared in pyproject.toml but missing from the lock: {unpinned}"


class TestTheGeneratorTargetsWhatCIActuallyRuns:
    def test_the_generator_targets_the_clean_install_jobs_platform(self) -> None:
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        generator = (ROOT / "scripts" / "generate-lock.py").read_text()
        assert 'python-version: "3.12"' in ci
        assert 'TARGET_PYTHON = "3.12"' in generator
        assert "manylinux2014_x86_64" in generator


class TestTheLocalScriptsAreAtLeastSyntacticallyValid:
    """`bash -n` catches a real class of bug cheaply: these scripts are
    not otherwise exercised by the normal test suite, since bootstrap
    genuinely installs things and verify genuinely needs a configured
    environment."""

    def test_bootstrap_local_has_valid_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(ROOT / "scripts" / "bootstrap-local.sh")], capture_output=True
        )
        assert result.returncode == 0, result.stderr.decode()

    def test_verify_local_has_valid_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(ROOT / "scripts" / "verify-local.sh")], capture_output=True
        )
        assert result.returncode == 0, result.stderr.decode()

    def test_both_scripts_are_executable(self) -> None:
        import os
        import stat

        for name in ("bootstrap-local.sh", "verify-local.sh"):
            mode = os.stat(ROOT / "scripts" / name).st_mode
            assert mode & stat.S_IXUSR, f"{name} is not executable"

    def test_neither_script_echoes_env_file_contents(self) -> None:
        """A script that ever does `cat .env` or `source .env; echo
        $VAR` would be one edit away from leaking a credential into CI
        logs. Neither script should read .env's contents at all --
        configuration comes from `agentic-research check`, which
        reports "set" or "missing", never the value."""
        for name in ("bootstrap-local.sh", "verify-local.sh"):
            text = (ROOT / "scripts" / name).read_text()
            assert ".env" not in text or "EXPECTED_OLLAMA_DIGEST" in text, (
                f"{name} should not reference .env's contents directly"
            )
            assert "cat .env" not in text
            assert "source .env" not in text
