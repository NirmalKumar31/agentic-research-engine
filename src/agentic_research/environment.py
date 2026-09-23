"""Execution environment capture.

Latency and throughput numbers are hardware-specific, and a local-model
benchmark run on a laptop says nothing without the laptop. Recording the
environment alongside the measurement is what lets a number be labelled
rather than quietly generalised.

Collected at run time rather than pinned in a file so it describes the
machine that actually produced the numbers.
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx

from agentic_research.config import Settings

_TRACKED_PACKAGES = (
    "agentic-research-engine",
    "langgraph",
    "langchain-core",
    "langchain-openai",
    "langchain-ollama",
    "openai",
    "pydantic",
    "httpx",
    "trafilatura",
    "pypdf",
)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _ollama_info(base_url: str) -> dict[str, Any]:
    """Server version and the digest of each loaded model.

    The digest matters: "qwen3:4b" is a moving tag, so a benchmark that
    records only the tag cannot be reproduced later.
    """
    info: dict[str, Any] = {}
    try:
        with httpx.Client(timeout=3.0) as client:
            info["version"] = (
                client.get(f"{base_url.rstrip('/')}/api/version").json().get("version", "unknown")
            )
            models = client.get(f"{base_url.rstrip('/')}/api/tags").json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return {"version": "unreachable"}

    info["models"] = {
        m.get("name", "?"): {
            "digest": (m.get("digest") or "")[:16],
            "size_bytes": m.get("size"),
            "quantization": (m.get("details") or {}).get("quantization_level"),
            "parameter_size": (m.get("details") or {}).get("parameter_size"),
        }
        for m in models
        if isinstance(m, dict)
    }
    return info


def capture(settings: Settings | None = None) -> dict[str, Any]:
    """Describe the machine and software producing a measurement."""
    snapshot: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": _cpu_count(),
        "packages": _package_versions(),
    }
    if settings is not None:
        snapshot["ollama"] = _ollama_info(settings.ollama_base_url)
        snapshot["settings"] = {
            "llm_mode": settings.llm_mode.value,
            "max_research_rounds": settings.max_research_rounds,
            "max_sources": settings.max_sources,
            "max_sources_per_round": settings.max_sources_per_round,
            "max_llm_calls": settings.max_llm_calls,
            "max_parallel_searches": settings.max_parallel_searches,
            "max_parallel_fetches": settings.max_parallel_fetches,
            "max_parallel_local_llm_calls": settings.max_parallel_local_llm_calls,
            "search_depth": settings.search_depth,
            "llm_temperature": settings.llm_temperature,
        }
    return snapshot


def _cpu_count() -> int | None:
    import os

    try:
        return os.cpu_count()
    except NotImplementedError:  # pragma: no cover - platform dependent
        return None


def describe(snapshot: dict[str, Any]) -> str:
    """One-line human summary, for labelling a figure in prose."""
    packages = snapshot.get("packages", {})
    ollama = snapshot.get("ollama", {})
    parts = [
        f"Python {snapshot.get('python')}",
        snapshot.get("platform", "?"),
        f"langgraph {packages.get('langgraph', '?')}",
    ]
    if ollama.get("version") and ollama["version"] != "unreachable":
        parts.append(f"ollama {ollama['version']}")
    return ", ".join(parts)
