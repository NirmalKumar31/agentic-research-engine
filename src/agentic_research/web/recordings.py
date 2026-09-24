"""Recorded research runs, served without touching a provider.

The public site shows real executions rather than running new ones. That is
a deliberate deployment decision, not a limitation of the engine: the demo's
daily run cap lives in process memory, and a free host that spins down when
idle resets it on every cold start, so it cannot bound an account-level
quota. Replaying a recording bounds it at zero.

What is recorded is the *derived* artifact the API already returns -- report,
claims, evidence with its quotes, sources with their metadata, verification
and metrics. Source page text is never included: it is large, it is somebody
else's copyright, and nothing in the provenance UI needs it. A quote is a
short span the report already cites.

A recording is therefore safe to commit, safe to serve anonymously, and
indistinguishable in the UI from a completed live run -- except that it is
labelled as a recording, which it must be.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from agentic_research.observability import get_logger
from agentic_research.runner import RunResult

log = get_logger(__name__)

# Ships inside the package so the wheel and the container carry it. A
# top-level directory would be excluded by .dockerignore and the site would
# deploy with no examples at all.
#
# Named "recorded_runs" rather than "recordings" so it cannot collide with
# this module. A sibling directory of the same name is a namespace package,
# which import resolution ranks below a regular module -- so runtime still
# finds this file -- but type checkers resolve the directory instead and
# report every symbol here as missing.
RECORDINGS_DIR = Path(__file__).resolve().parent / "recorded_runs"

# Bumped when the recording payload changes shape. A committed example
# outlives the code that produced it, and a silently-incompatible old file
# renders as a subtly broken demo rather than an obvious error.
RECORDING_SCHEMA_VERSION = 1

# Progress events are copied into a committed, publicly served file, so the
# fields that survive are listed rather than filtered. An allowlist cannot
# leak a field added later; a denylist silently can.
_TRACE_FIELDS: dict[str, frozenset[str]] = {
    "started": frozenset({"query"}),
    "analyzing_query": frozenset({"query"}),
    "query_analyzed": frozenset({"intent", "format"}),
    "planning": frozenset(),
    "plan_generated": frozenset({"count", "questions"}),
    "generating_queries": frozenset({"round", "sub_questions"}),
    "queries_generated": frozenset({"count", "queries", "round"}),
    "search_completed": frozenset({"query", "query_id", "results"}),
    "search_failed": frozenset({"query_id"}),
    "sources_deduplicated": frozenset({"results", "unique", "avoided", "selected"}),
    "source_retrieved": frozenset({"source_id", "title"}),
    "source_failed": frozenset({"source_id", "status"}),
    "sources_registered": frozenset({"usable", "duplicates", "extracting"}),
    "evidence_extracted": frozenset({"source_id", "items"}),
    "assessing_coverage": frozenset({"round"}),
    "coverage_evaluated": frozenset({"round", "ratio", "covered", "weak", "missing", "sufficient"}),
    "synthesizing": frozenset({"evidence"}),
    "synthesized": frozenset({"sections", "findings"}),
    "verifying_citations": frozenset(),
    "citations_verified": frozenset({"total", "evidence_integrity", "support", "exhaustive"}),
    "completed": frozenset({"stop_reason"}),
}

# `started` also carries run_id and a models map. Neither belongs in a
# published recording: the run id is meaningless once replayed, and the
# model map is deployment configuration.


def sanitise_trace(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce progress events to the fields a visitor may see.

    Unknown event types are dropped rather than passed through. A new
    event added to the graph should have to be considered here before it
    appears in a public file.
    """
    cleaned: list[dict[str, Any]] = []
    for event in events:
        name = event.get("event")
        if not isinstance(name, str):
            continue
        allowed = _TRACE_FIELDS.get(name)
        if allowed is None:
            # Not `event=`: structlog reserves that key for the message.
            log.debug("trace_event_dropped", event_name=name)
            continue
        cleaned.append({"event": name, **{k: event[k] for k in sorted(allowed) if k in event}})
    return cleaned


def public_provenance(environment: dict[str, Any]) -> dict[str, Any]:
    """The identifiers that make a recording reproducible, and nothing else.

    A full environment capture holds the resolved settings, installed
    package versions and local model configuration. That is the right
    content for a private run artifact and the wrong content for a file
    served to the public: it is deployment detail, and it is exactly the
    kind of blob a credential eventually gets added to.
    """
    provenance = environment.get("provenance", {}) or {}
    git = provenance.get("git", {}) or {}
    return {
        "engine_version": provenance.get("engine_version"),
        "evaluator_version": provenance.get("evaluator_version"),
        "prompt_version": provenance.get("prompt_version"),
        "schema_version": provenance.get("schema_version"),
        "config_fingerprint": provenance.get("config_fingerprint"),
        "commit": git.get("commit"),
        "short_commit": git.get("short_commit"),
        "dirty": git.get("dirty"),
        "python": environment.get("python"),
        "platform": environment.get("platform"),
    }


# Ids appear in a URL path and are used to build a filename, so they are
# restricted rather than sanitised. Rejecting anything unexpected is easier
# to reason about than trying to neutralise it.
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

# Keys that must never appear in a committed recording, checked on load so a
# bad recording fails loudly in CI rather than quietly on the public site.
_FORBIDDEN_KEYS = frozenset(
    {"api_key", "openai_api_key", "tavily_api_key", "brave_api_key", "authorization", "token"}
)

# Key names are not enough: {"note": "sk-proj-..."} passes a name-only
# check. These match the *value* shapes, and mirror the rules in
# .gitleaks.toml so the two cannot disagree about what a key looks like.
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-proj-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}"),
    re.compile(r"\btvly-[A-Za-z0-9]{16,}"),
    re.compile(r"\bBSA[A-Za-z0-9_-]{20,}"),  # Brave
    re.compile(r"\bghp_[A-Za-z0-9]{36}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


class RecordingNotFound(Exception):
    """No recording with that id. Raised rather than returning None so the
    route cannot accidentally serve a 200 with an empty body."""


@dataclass(frozen=True)
class RecordingSummary:
    """What the homepage needs to offer a recording, and nothing more."""

    id: str
    question: str
    label: str
    description: str
    mode: str
    recorded_at: str
    sources: int
    evidence_items: int
    citable_evidence: int
    has_pdf_evidence: bool
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "label": self.label,
            "description": self.description,
            "mode": self.mode,
            "recorded_at": self.recorded_at,
            "sources": self.sources,
            "evidence_items": self.evidence_items,
            "citable_evidence": self.citable_evidence,
            "has_pdf_evidence": self.has_pdf_evidence,
            "duration_s": self.duration_s,
        }


def valid_id(candidate: str) -> bool:
    return bool(_SAFE_ID.match(candidate or ""))


def assert_no_secrets(recording_id: str, payload: Any) -> None:
    """Walk a recording and refuse anything that looks like a credential.

    Both halves matter. A name-only check passes ``{"note": "sk-proj-..."}``
    straight through, and a value-only check misses an empty-but-named
    field that a later edit fills in. Neither the key nor the matched text
    is ever included in the error: this runs in CI logs, and a message that
    quotes the secret it found has published it again.
    """
    stack: list[tuple[str, Any]] = [("$", payload)]
    while stack:
        where, node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() in _FORBIDDEN_KEYS:
                    raise ValueError(
                        f"recording {recording_id!r} contains a forbidden key at {where}"
                    )
                stack.append((f"{where}.{key}", value))
        elif isinstance(node, list):
            stack.extend((f"{where}[{i}]", item) for i, item in enumerate(node))
        elif isinstance(node, str):
            for pattern in _SECRET_VALUE_PATTERNS:
                if pattern.search(node):
                    raise ValueError(
                        f"recording {recording_id!r} contains a credential-shaped value at {where}"
                    )


def _summary_from(recording_id: str, payload: dict[str, Any]) -> RecordingSummary:
    meta = payload.get("meta", {})
    result = payload.get("result", {})
    evidence = result.get("evidence", []) or []
    sources = result.get("sources", []) or []
    metrics = result.get("metrics", {}) or {}
    return RecordingSummary(
        id=recording_id,
        question=meta.get("question", ""),
        label=meta.get("label", recording_id),
        description=meta.get("description", ""),
        mode=meta.get("mode", "unknown"),
        recorded_at=meta.get("recorded_at", ""),
        sources=len(sources),
        evidence_items=len(evidence),
        citable_evidence=sum(1 for e in evidence if e.get("citable")),
        has_pdf_evidence=any(e.get("page") for e in evidence),
        duration_s=float(metrics.get("duration_s") or 0.0),
    )


@lru_cache(maxsize=1)
def _index() -> dict[str, dict[str, Any]]:
    """Load every recording once.

    Cached because the files are committed and cannot change while the
    process runs, and the public site would otherwise re-read and re-validate
    them on every request.
    """
    loaded: dict[str, dict[str, Any]] = {}
    if not RECORDINGS_DIR.is_dir():
        log.info("no_recordings_directory", path=str(RECORDINGS_DIR))
        return loaded

    for path in sorted(RECORDINGS_DIR.glob("*.json")):
        recording_id = path.stem
        if not valid_id(recording_id):
            log.warning("recording_id_rejected", filename=path.name)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("recording_unreadable", id=recording_id, error=str(exc)[:200])
            continue
        if not isinstance(payload, dict) or "result" not in payload:
            log.warning("recording_malformed", id=recording_id)
            continue
        version = payload.get("recording_schema_version")
        if version != RECORDING_SCHEMA_VERSION:
            # Skipped rather than rendered. A recording outlives the code
            # that made it, and a shape change would otherwise surface as a
            # subtly broken demo instead of an obvious absence.
            log.warning(
                "recording_schema_mismatch",
                id=recording_id,
                found=version,
                expected=RECORDING_SCHEMA_VERSION,
            )
            continue
        assert_no_secrets(recording_id, payload)
        loaded[recording_id] = payload

    log.info("recordings_loaded", count=len(loaded), ids=sorted(loaded))
    return loaded


def available() -> list[RecordingSummary]:
    """Every recording, ordered for display."""
    index = _index()
    summaries = [_summary_from(rid, payload) for rid, payload in index.items()]
    # Ordered by the curator's intent rather than by filename.
    summaries.sort(key=lambda s: (_index()[s.id].get("meta", {}).get("order", 99), s.id))
    return summaries


def load(recording_id: str) -> dict[str, Any]:
    """One recording's full payload.

    The id is validated against a strict pattern before it is used, so a
    traversal attempt is rejected as an unknown id rather than resolved and
    then checked.
    """
    if not valid_id(recording_id):
        raise RecordingNotFound(recording_id)
    payload = _index().get(recording_id)
    if payload is None:
        raise RecordingNotFound(recording_id)
    return payload


def trace(recording_id: str) -> list[dict[str, Any]]:
    """The progress events the original run actually emitted.

    Empty when the run was not captured with its trace. Replaying an empty
    trace shows the finished report with no progress animation, which is
    honest; inventing plausible-looking stages would not be.
    """
    return list(load(recording_id).get("trace", []) or [])


# ---------------------------------------------------------------------------
# Shared result serialisation
# ---------------------------------------------------------------------------
#
# Lives here rather than in api.py so the recorder and the live endpoint
# cannot drift: a recording must be byte-identical in shape to what a live
# run returns, or the UI would need two code paths to render them. It also
# keeps the recorder free of a FastAPI import, since fastapi is an optional
# extra and `agentic-research record` has to work without it.


def serialise_result(result: RunResult) -> dict[str, Any]:
    """Everything the UI needs to show provenance, and nothing more.

    Deliberately assembled field by field rather than dumping state: state
    holds full source text and would leak configuration if serialised
    wholesale.
    """
    state = result.state
    report = state.get("report")
    evidence = state.get("evidence", []) or []
    sources = state.get("sources", []) or []
    plan = state.get("plan")

    def claim(c: Any) -> dict[str, Any]:
        return {
            "text": c.text,
            "kind": c.kind.value,
            "evidence_ids": list(c.evidence_ids),
            "citation_ids": list(c.citation_ids),
        }

    return {
        "run_id": result.run_id,
        "report": None
        if report is None
        else {
            "title": report.title,
            "summary_claims": [claim(c) for c in report.summary_claims],
            "key_findings": [claim(c) for c in report.key_findings],
            "sections": [
                {"heading": s.heading, "claims": [claim(c) for c in s.claims]}
                for s in report.sections
            ],
            "contradictions": [
                {
                    "topic": c.topic,
                    "left_summary": c.left_summary,
                    "left_evidence_ids": list(c.left_evidence_ids),
                    "left_citation_ids": list(c.left_citation_ids),
                    "right_summary": c.right_summary,
                    "right_evidence_ids": list(c.right_evidence_ids),
                    "right_citation_ids": list(c.right_citation_ids),
                    "auditable": c.is_auditable,
                }
                for c in report.contradictions
            ],
            "limitations": list(report.limitations),
        },
        "plan": None
        if plan is None
        else {
            "strategy": plan.strategy_note,
            "sub_questions": [
                {
                    "id": q.id,
                    "text": q.text,
                    "rationale": q.rationale,
                    "is_followup": q.is_followup,
                }
                for q in plan.sub_questions
            ],
        },
        "evidence": [
            {
                "id": e.id,
                "source_id": e.source_id,
                "sub_question_id": e.sub_question_id,
                "claim": e.claim,
                "quote": e.quote,
                "quote_match": e.quote_match.value,
                "page": e.page,
                "stance": e.stance.value,
                "confidence": e.confidence,
                "citable": e.is_citable,
                "query_id": e.query_id,
                "cross_attributed": e.cross_attributed,
            }
            for e in evidence
        ],
        "sources": [
            {
                "id": s.id,
                "url": s.url,
                "title": s.title,
                "domain": s.domain,
                "source_type": s.source_type.value,
                "content_origin": s.content_origin.value,
                "quality_score": s.quality_score,
                "page_count": s.page_count,
                "usable": s.is_usable,
                "fetch_status": s.fetch_status.value,
            }
            for s in sources
        ],
        "verification": state.get("verification"),
        "metrics": result.metrics.model_dump(mode="json"),
        "markdown": result.markdown,
    }
