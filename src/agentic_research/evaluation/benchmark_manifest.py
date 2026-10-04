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


class CorpusHashMismatchError(Exception):
    """A frozen corpus does not match the hash recorded for it.

    Raised, not logged: a benchmark run over evidence that silently
    drifted from what was frozen is not measuring what it claims to be
    measuring, and the whole reason a corpus is hashed at freeze time is
    so this is checkable mechanically rather than trusted by convention.
    """


def corpus_hash(corpus: Any) -> str:
    """A deterministic sha256 of exactly what `EvidenceCorpus.save` would
    write to disk.

    Hashes the artifact as it will actually be read back -- including
    `captured_at` -- not a looser "same semantic content" notion. Two
    freezes of the same question on different days are different
    artifacts and get different hashes on purpose: a benchmark run
    should be verified against the one frozen corpus it was actually
    pointed at, not against "a" corpus that happens to answer the same
    question.
    """
    import hashlib
    import json

    payload = corpus.to_dict()
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def freeze_corpus_hashes(corpora: dict[str, Any]) -> dict[str, str]:
    """`{question_id: sha256}` for a batch of frozen corpora -- the form
    `BenchmarkManifest.corpus_hashes` expects, computed rather than typed
    in by hand."""
    return {question_id: corpus_hash(corpus) for question_id, corpus in corpora.items()}


def verify_corpus_hash(question_id: str, corpus: Any, manifest: BenchmarkManifest) -> None:
    """Fail closed if `corpus` does not match the hash the manifest
    recorded for `question_id` at freeze time.

    Called immediately before a corpus is handed to `compare()` -- after
    this, the comparison itself is read-only evidence, and checking here
    rather than trusting the caller is what makes "frozen" an enforced
    property instead of a naming convention.
    """
    expected = manifest.corpus_hashes.get(question_id)
    if expected is None:
        raise CorpusHashMismatchError(
            f"no hash recorded for {question_id!r} -- the manifest does not "
            "know about this corpus at all, which is worse than a mismatch"
        )
    actual = corpus_hash(corpus)
    if actual != expected:
        raise CorpusHashMismatchError(
            f"{question_id}: corpus hash is {actual}, manifest recorded "
            f"{expected} -- the evidence this would run against is not the "
            "evidence that was frozen and verified"
        )


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
