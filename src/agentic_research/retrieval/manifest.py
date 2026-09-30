"""What was searched, what was found, and why each page was or was not read.

The gap this closes. A hosted run on the causes of overfitting read six
pages -- a tweet, a LinkedIn-style blog, a newsletter and two papers on
double descent -- from 44 unique candidates. Afterwards the other 38
were unrecoverable: the service runs with ``PERSIST_RUNS=false`` and the
search stage streamed only counts, so the question "was there a good
source in the pool that selection passed over?" had no answer and could
not be given one. Diagnosing the failure meant reasoning from six
survivors.

So the next run records its own selection. The manifest travels in the
run's result payload, which the acceptance capture writes to disk
byte-for-byte, so it survives without Render log access and without
persisting anything server-side.

**Bounded by construction.** One record per issued query and one per
deduplicated candidate, both already limited by the run's own
``max_search_queries`` and the provider's results-per-query. There is no
path here that grows with anything unbounded, and
:data:`_MAX_CANDIDATES` is a hard stop with an explicit truncation count
rather than a silent cut.

**What it deliberately does not carry.** No credentials, no
``Authorization`` headers, no cookies, no raw provider response
objects, no fetched page bodies, no prompts, no internal service URLs
and no filesystem paths. URLs are reduced to scheme, host and path:
query strings are where API keys, session tokens and tracking
identifiers live, and a fragment can carry the same. See
:func:`sanitize_url`.

This is evaluation evidence. The normal interface stays concise; the
manifest belongs in the technical trace and the artifact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

from agentic_research.citations.guards import authority_of
from agentic_research.evidence.dedup import Candidate
from agentic_research.evidence.quality import classify_source
from agentic_research.retrieval.selection import SelectionResult, score_breakdown

# One record per candidate the dedup barrier produced. Six queries at
# ten results each cannot exceed this, so it is a guard against a
# provider returning more than it advertises rather than a routine cut.
_MAX_CANDIDATES = 120

# Characters that let a title or URL break out of the line or block it
# is rendered in. A manifest is read as JSON, as a log line and in
# Markdown, and a page title is attacker-controlled text.
_UNSAFE = str.maketrans(
    {
        "\n": " ",
        "\r": " ",
        "\t": " ",
        "\x00": "",
        "`": "'",
        "|": "/",
        "<": "(",
        ">": ")",
    }
)

_MAX_TEXT = 300


def safe_text(text: str, limit: int = _MAX_TEXT) -> str:
    """A title or reason, safe to place in a line, a cell or a log.

    Newlines are the injection that matters: a title containing one
    becomes a second log line, and in Markdown a pipe becomes a second
    table cell. Truncation is marked rather than silent.
    """
    cleaned = (text or "").translate(_UNSAFE).strip()
    if len(cleaned) > limit:
        return cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def sanitize_url(url: str) -> str:
    """Scheme, host and path. No userinfo, query, or fragment.

    Everything removed here is somewhere a secret has been found in the
    wild: query strings carry API keys, signed-URL tokens, session ids
    and campaign identifiers; ``user:password@`` carries credentials
    outright; a fragment can carry either because it is not sent to the
    server and so gets treated as scratch space.

    The host keeps its port, because a port is not sensitive and
    dropping it would make two different services look identical.
    """
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return ""
    if not parts.scheme or not parts.netloc:
        return ""
    # `hostname` drops any user:password@; the port is re-attached.
    host = (parts.hostname or "").lower()
    if not host:
        return ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return safe_text(urlunsplit((parts.scheme.lower(), host, parts.path, "", "")))


def host_of(sanitized: str) -> str:
    """The host of an already-sanitised URL.

    Derived rather than taken from ``Candidate.domain``. That field is
    populated upstream and in one path carried ``user:pw@example.com``
    straight into the manifest while the URL beside it was clean --
    caught by the redaction test. Anything in this structure that could
    contain userinfo is computed from the sanitised URL, so there is one
    place to be right rather than two to keep in step.
    """
    try:
        return urlsplit(sanitized).hostname or ""
    except ValueError:
        return ""


@dataclass(frozen=True)
class QueryRecord:
    """One issued search query and what it was for."""

    query_id: str
    sub_question_id: str
    text: str
    rationale: str
    round_number: int
    result_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "sub_question_id": self.sub_question_id,
            "text": safe_text(self.text),
            "rationale": safe_text(self.rationale),
            "round": self.round_number,
            "results": self.result_count,
        }


@dataclass(frozen=True)
class CandidateRecord:
    """One deduplicated candidate and the decision taken about it."""

    url: str
    domain: str
    source_type: str
    authority: str
    provider_score: float
    authority_adjustment: float
    ranking_score: float
    sub_question_ids: tuple[str, ...]
    selected: bool
    reason: str
    fetch_outcome: str = "not fetched"

    def to_dict(self) -> dict[str, object]:
        return {
            "url": self.url,
            "domain": self.domain,
            "source_type": self.source_type,
            "authority": self.authority,
            "provider_score": round(self.provider_score, 4),
            # The two halves of the adjustment separately, so the
            # ordering can be recomputed rather than trusted.
            "authority_adjustment": round(self.authority_adjustment, 4),
            "ranking_score": round(self.ranking_score, 4),
            "sub_question_ids": list(self.sub_question_ids),
            "selected": self.selected,
            "reason": safe_text(self.reason),
            "fetch_outcome": self.fetch_outcome,
        }


@dataclass(frozen=True)
class RetrievalManifest:
    """Every query and candidate for one round, bounded and sanitised."""

    round_number: int
    queries: tuple[QueryRecord, ...] = ()
    candidates: tuple[CandidateRecord, ...] = ()
    truncated: int = 0
    explanatory_policy: bool = True
    by_sub_question: dict[str, list[str]] = field(default_factory=dict)

    @property
    def selected(self) -> tuple[CandidateRecord, ...]:
        return tuple(c for c in self.candidates if c.selected)

    @property
    def dropped(self) -> tuple[CandidateRecord, ...]:
        return tuple(c for c in self.candidates if not c.selected)

    def to_dict(self) -> dict[str, object]:
        return {
            "round": self.round_number,
            "explanatory_policy": self.explanatory_policy,
            "queries": [q.to_dict() for q in self.queries],
            "candidates": [c.to_dict() for c in self.candidates],
            "selected_count": len(self.selected),
            "dropped_count": len(self.dropped),
            # Non-zero means the provider returned more than the bound
            # allows for. Recorded rather than hidden, because a silent
            # cut would make the manifest look complete when it is not.
            "truncated": self.truncated,
            "by_sub_question": {k: list(v) for k, v in self.by_sub_question.items()},
        }


def build_manifest(
    *,
    round_number: int,
    queries: Sequence[object],
    results: Sequence[object],
    candidates: Sequence[Candidate],
    selection: SelectionResult,
    explanatory: bool,
) -> RetrievalManifest:
    """Assemble one round's manifest from what the node already has.

    ``queries`` are this round's :class:`SearchQuery` objects and
    ``results`` the provider hits they produced; the result count per
    query is counted here rather than tracked, so a query that returned
    nothing is visible as ``results: 0`` instead of being absent.

    Selection decisions come from the :class:`SelectionResult` rather
    than being re-derived, so the manifest cannot disagree with what was
    actually fetched -- two derivations of one decision is the defect
    this repository keeps producing.
    """
    per_query: dict[str, int] = {}
    for result in results:
        query_id = str(getattr(result, "query_id", "") or "")
        per_query[query_id] = per_query.get(query_id, 0) + 1

    query_records = tuple(
        QueryRecord(
            query_id=str(getattr(q, "id", "")),
            sub_question_id=str(getattr(q, "sub_question_id", "")),
            text=str(getattr(q, "text", "")),
            rationale=str(getattr(q, "rationale", "")),
            round_number=int(getattr(q, "round_number", round_number) or round_number),
            result_count=per_query.get(str(getattr(q, "id", "")), 0),
        )
        for q in queries
    )

    chosen = {c.canonical_url for c in selection.selected}
    reasons = {d.url: d.reason for d in selection.dropped}

    records: list[CandidateRecord] = []
    truncated = 0
    for candidate in candidates:
        if len(records) >= _MAX_CANDIDATES:
            truncated += 1
            continue
        source_type = classify_source(candidate.url, candidate.domain)
        provider, adjustment, total = score_breakdown(candidate, explanatory=explanatory)
        selected = candidate.canonical_url in chosen
        clean_url = sanitize_url(candidate.url)
        records.append(
            CandidateRecord(
                url=clean_url,
                # From the sanitised URL, not from `candidate.domain`.
                domain=safe_text(host_of(clean_url), 120),
                source_type=source_type.value,
                authority=authority_of(source_type.value).value,
                provider_score=provider,
                authority_adjustment=adjustment,
                ranking_score=total,
                # Every sub-question the candidate serves, not the one
                # that happened to claim it.
                sub_question_ids=tuple(candidate.sub_question_ids),
                selected=selected,
                reason=(
                    "selected for its highest-ranked sub-question"
                    if selected
                    else reasons.get(candidate.url, "not selected")
                ),
            )
        )

    return RetrievalManifest(
        round_number=round_number,
        queries=query_records,
        candidates=tuple(records),
        truncated=truncated,
        explanatory_policy=explanatory,
        # Sanitised here too. `SelectionResult.by_sub_question` holds
        # raw canonical URLs, and copying it verbatim carried a query
        # string -- with an API key in it -- straight into the
        # manifest. Caught by the redaction test, which is the only
        # reason it is not in the artifact: every URL entering this
        # structure has to pass through `sanitize_url`, not just the
        # ones on the candidate records.
        by_sub_question={
            sub_question: [sanitize_url(url) for url in urls]
            for sub_question, urls in selection.by_sub_question.items()
        },
    )


def with_fetch_outcomes(
    manifest: dict[str, object], sources: Sequence[object]
) -> dict[str, object]:
    """Fill in what happened to each selected candidate.

    Fetch outcomes are only known after the fetch stage, so they are
    merged into the already-serialised manifest by sanitised URL rather
    than reopening it. A selected candidate with no matching source
    keeps ``not fetched``, which is itself a finding: it means selection
    chose a page the fetch stage never attempted.
    """
    outcome_by_url: dict[str, str] = {}
    for source in sources:
        url = sanitize_url(str(getattr(source, "url", "")))
        status = getattr(getattr(source, "fetch_status", None), "value", "unknown")
        origin = getattr(getattr(source, "content_origin", None), "value", "")
        if url:
            outcome_by_url[url] = f"{status} ({origin})" if origin else str(status)

    candidates = manifest.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("selected"):
                url = str(candidate.get("url", ""))
                candidate["fetch_outcome"] = outcome_by_url.get(url, "not fetched")
    return manifest
