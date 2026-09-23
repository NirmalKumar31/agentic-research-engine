"""Code provenance for measurements.

A metric is only meaningful alongside the code that produced it. That is
not a theoretical concern here: the first cloud run surfaced three bugs and
the implementation changed immediately afterwards, which silently
invalidated the numbers that run had just produced. Nothing in the artifact
recorded which code it came from, so there was no way to tell from the file
alone.

Everything here is derived rather than hand-maintained. A version constant
someone has to remember to bump is a version constant that will be wrong,
so prompts and schemas are hashed from their own content and the git SHA is
read from the repository.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

# Bumped by hand when the *meaning* of an evaluation metric changes, which
# no hash can detect: renaming citation_validity to citation_integrity and
# tightening quote fidelity to exact-only both changed what the numbers
# mean without changing how they are computed.
EVALUATOR_VERSION = "2"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@lru_cache(maxsize=1)
def prompt_version() -> str:
    """Stable hash of every prompt the engine sends.

    Prompts change behaviour as much as code does, and editing one is
    easier to forget than editing a module. Hashing the source of the
    prompts module means the fingerprint moves whenever a prompt does.
    """
    try:
        from agentic_research.graph import prompts

        source = Path(prompts.__file__).read_text(encoding="utf-8")
    except (OSError, ImportError):  # pragma: no cover - importable in practice
        return "unknown"
    return _digest(source)


@lru_cache(maxsize=1)
def schema_version() -> str:
    """Stable hash of the structured-output contracts.

    Derived from the JSON Schema the models are actually held to, not from
    the Python source, so a comment change does not move it but a field
    change does.
    """
    try:
        from agentic_research import schemas
    except ImportError:  # pragma: no cover
        return "unknown"

    from pydantic import BaseModel

    shapes: dict[str, Any] = {}
    for name in sorted(getattr(schemas, "__all__", [])):
        candidate = getattr(schemas, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            shapes[name] = candidate.model_json_schema()
    if not shapes:
        return "unknown"
    return _digest(json.dumps(shapes, sort_keys=True))


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_state() -> dict[str, Any]:
    """Commit the measurement came from, and whether the tree was dirty.

    A dirty tree means the commit does not describe what actually ran, so
    the flag matters as much as the SHA. Absent git (an installed wheel,
    say) is reported rather than guessed at.
    """
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return {"commit": "unavailable", "dirty": None, "branch": None}
    status = _git("status", "--porcelain")
    return {
        "commit": sha,
        "short_commit": sha[:8],
        "dirty": bool(status) if status is not None else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def config_fingerprint(settings: Any) -> str:
    """Hash of the settings that change results.

    Only behaviour-affecting fields, and never a credential: a fingerprint
    that moved when a key rotated would be useless, and one that embedded a
    key would be dangerous.
    """
    relevant = {
        "llm_mode": getattr(settings.llm_mode, "value", str(settings.llm_mode)),
        "models": {role.value: str(spec) for role, spec in settings.resolve_models().items()},
        "temperature": settings.llm_temperature,
        "max_research_rounds": settings.max_research_rounds,
        "max_search_queries": settings.max_search_queries,
        "max_sources": settings.max_sources,
        "max_sources_per_round": settings.max_sources_per_round,
        "max_llm_calls": settings.max_llm_calls,
        "max_provider_requests": settings.max_provider_requests,
        "max_search_results": settings.max_search_results,
        "search_depth": settings.search_depth,
        "search_provider": settings.search_provider,
        "max_extract_chars": settings.max_extract_chars,
        "max_pdf_pages": settings.max_pdf_pages,
        "local_num_ctx": settings.local_num_ctx,
        "output_caps": settings.cloud_budget.max_output_tokens_per_call,
    }
    return _digest(json.dumps(relevant, sort_keys=True, default=str))


def capture(settings: Any = None) -> dict[str, Any]:
    """The full provenance block attached to every measurement."""
    from agentic_research import __version__

    block: dict[str, Any] = {
        "engine_version": __version__,
        "evaluator_version": EVALUATOR_VERSION,
        "prompt_version": prompt_version(),
        "schema_version": schema_version(),
        "git": git_state(),
    }
    if settings is not None:
        block["config_fingerprint"] = config_fingerprint(settings)
        block["model_assignments"] = {
            role.value: str(spec) for role, spec in settings.resolve_models().items()
        }
    return block


def describe(block: dict[str, Any]) -> str:
    """One-line summary for labelling a figure in prose."""
    git = block.get("git", {})
    dirty = " (dirty)" if git.get("dirty") else ""
    return (
        f"v{block.get('engine_version')} @ {git.get('short_commit', '?')}{dirty}, "
        f"prompts {block.get('prompt_version')}, "
        f"schemas {block.get('schema_version')}, "
        f"evaluator v{block.get('evaluator_version')}"
    )
