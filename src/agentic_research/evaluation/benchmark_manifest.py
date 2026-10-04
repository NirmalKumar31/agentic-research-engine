"""The frozen configuration a local-vs-cloud benchmark is measured under.

Written before any run, not derived from one afterwards: every field here
is something that must be fixed *before* the first question is asked, or
a difference in results could be attributed to a change in setup rather
than to the models being compared. This is Phase A infrastructure --
construction and validation, no provider call. See
``docs/BENCHMARK-PROTOCOL.md`` for the protocol this manifest belongs to,
and the explicit statement that Phase B (any paid run) needs separate,
specific authorization this module does not grant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkManifest:
    """One benchmark's frozen identity.

    Every field is either read from the repository (versions, hashes) or
    supplied once by whoever runs the benchmark (model ids, ceilings) --
    nothing here is filled in by a run, which is what makes it checkable
    *before* spending anything.
    """

    benchmark_version: str
    frozen_engine_commit: str
    cloud_model_id: str
    nli_model_id: str
    nli_model_revision: str
    prompt_version: str
    schema_version: str
    config_fingerprint: str
    corpus_hashes: dict[str, str] = field(default_factory=dict)
    scoring_version: str = "v1"
    ceilings: dict[str, int] = field(default_factory=dict)
    ollama_model: str = ""
    ollama_version: str = ""
    ollama_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_version": self.benchmark_version,
            "frozen_engine_commit": self.frozen_engine_commit,
            "cloud_model_id": self.cloud_model_id,
            "ollama_model": self.ollama_model,
            "ollama_version": self.ollama_version,
            "ollama_digest": self.ollama_digest,
            "nli_model_id": self.nli_model_id,
            "nli_model_revision": self.nli_model_revision,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "config_fingerprint": self.config_fingerprint,
            "corpus_hashes": dict(self.corpus_hashes),
            "scoring_version": self.scoring_version,
            "ceilings": dict(self.ceilings),
        }

    def completeness_problems(self) -> list[str]:
        """What is missing before this manifest could gate a real run.

        Checked explicitly rather than left to show up as a confusing
        failure three questions into a paid benchmark: a manifest is
        either ready to freeze a run against, or it names exactly what
        is not filled in yet.
        """
        problems = []
        required_str = {
            "frozen_engine_commit": self.frozen_engine_commit,
            "cloud_model_id": self.cloud_model_id,
            "nli_model_id": self.nli_model_id,
            "nli_model_revision": self.nli_model_revision,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "config_fingerprint": self.config_fingerprint,
        }
        for name, value in required_str.items():
            if not value:
                problems.append(f"{name} is empty")
        if len(self.nli_model_revision) not in (0, 40):
            problems.append(
                f"nli_model_revision is {len(self.nli_model_revision)} chars, "
                "not a full 40-character commit -- the same requirement "
                "config.py enforces for every other run"
            )
        if not self.corpus_hashes:
            problems.append("no corpus hashes recorded -- nothing is frozen yet")
        return problems


def build_manifest(
    settings: Any,
    *,
    benchmark_version: str,
    cloud_model_id: str,
    corpus_hashes: dict[str, str],
    ceilings: dict[str, int] | None = None,
) -> BenchmarkManifest:
    """Assemble a manifest from the repository's own provenance and the
    settings a run would use -- the same sources `environment.capture`
    and `provenance.capture` already read for every other measurement in
    this project, not a parallel source of truth."""
    from agentic_research.provenance import (
        config_fingerprint,
        git_state,
        prompt_version,
        schema_version,
    )

    git = git_state()
    return BenchmarkManifest(
        benchmark_version=benchmark_version,
        frozen_engine_commit=git.get("commit", "unavailable"),
        cloud_model_id=cloud_model_id,
        nli_model_id=getattr(settings, "nli_model_id", ""),
        nli_model_revision=getattr(settings, "nli_model_revision", ""),
        prompt_version=prompt_version(),
        schema_version=schema_version(),
        config_fingerprint=config_fingerprint(settings),
        corpus_hashes=dict(corpus_hashes),
        ceilings=dict(
            ceilings
            or {
                "max_research_rounds": getattr(settings, "max_research_rounds", 0),
                "max_sources": getattr(settings, "max_sources", 0),
                "max_llm_calls": getattr(settings, "max_llm_calls", 0),
                "max_provider_requests": getattr(settings, "max_provider_requests", 0),
            }
        ),
    )
