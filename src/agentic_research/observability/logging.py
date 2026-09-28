"""Structured logging.

Every log line is an event name plus typed key/value pairs, never an
interpolated sentence. That makes a run greppable (``event=search_completed``)
and lets the CLI, tests and any future log shipper read the same records.

``run_id`` is bound once per run via contextvars, so nodes deep in the graph do
not have to thread it through their signatures.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog

_configured = False

# Keys whose values must never reach a log sink.
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "openai_api_key",
        "tavily_api_key",
        "brave_api_key",
        "langsmith_api_key",
        # The remote verifier's credential. Added after it was found
        # missing: NLI arrived late and the redaction sets were written
        # before it existed.
        "nli_api_key",
        "hf_token",
        "huggingface_token",
        "authorization",
        "token",
        "bearer",
    }
)


# Credential shapes, scrubbed wherever they appear rather than only
# under a known key. Matching on the key name alone was the whole gap:
# most call sites log ``error=str(exc)`` or a URL, and a provider error
# can quote the request it failed on -- Authorization header included.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bhf_[A-Za-z0-9]{16,}"), "<redacted:hf>"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), "<redacted:openai>"),
    (re.compile(r"\btvly-[A-Za-z0-9_\-]{16,}"), "<redacted:tavily>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"), "Bearer <redacted>"),
    # Any URL carrying credentials, which is what a Key Value connection
    # string is: redis://user:password@host:port/db
    (re.compile(r"(?i)\b[a-z][a-z0-9+.\-]*://[^\s\"']*:[^\s\"'@/]+@[^\s\"']*"), "<redacted:url>"),
    (re.compile(r"(?i)\bredis(s)?://[^\s\"']*"), "<redacted:url>"),
    # ?api_key=... and friends in a query string.
    (
        re.compile(r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|key|secret)=)[^&\s\"']+"),
        r"\1<redacted>",
    ),
)

# Exact values registered at startup. Patterns cannot know the shape of
# every credential -- a Hugging Face endpoint URL is not secret-shaped
# and is still not something to publish -- so the configured values are
# scrubbed literally as well.
_SECRET_VALUES: set[str] = set()

# Below this, a "secret" is too short to scrub without mangling
# ordinary text.
_MIN_SCRUBBABLE = 8


def register_secret_values(*values: object) -> None:
    """Record literal values that must never appear in a log line.

    Accepts ``SecretStr``, ``str`` or ``None`` so callers can pass
    settings fields straight in without unwrapping them at every site.
    """
    for value in values:
        raw = getattr(value, "get_secret_value", None)
        text = raw() if callable(raw) else value
        if isinstance(text, str) and len(text.strip()) >= _MIN_SCRUBBABLE:
            _SECRET_VALUES.add(text.strip())


def _scrub_text(text: str) -> str:
    for secret in _SECRET_VALUES:
        if secret in text:
            text = text.replace(secret, "<redacted>")
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _scrub(value: Any, depth: int = 0) -> Any:
    """Walk a log record, scrubbing strings wherever they are.

    Bounded depth so a self-referential structure cannot spin here; a
    log processor that hangs takes the process with it.
    """
    if depth > 6:
        return value
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, dict):
        return {k: _scrub(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        rebuilt = [_scrub(v, depth + 1) for v in value]
        return type(value)(rebuilt) if not isinstance(value, set) else set(rebuilt)
    if isinstance(value, BaseException):
        return _scrub_text(f"{type(value).__name__}: {value}")
    return value


def _redact_secrets(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Defence in depth, in two layers.

    Known key names are blanked outright. Everything else is walked and
    scrubbed by value, because the leak that actually happens is not
    ``log.info("x", api_key=...)`` -- it is ``error=str(exc)`` where the
    exception quotes a URL or a header.
    """
    for key in list(event_dict):
        if key.lower() in _SECRET_KEYS:
            event_dict[key] = "<redacted>"
        else:
            event_dict[key] = _scrub(event_dict[key])
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Idempotent logging setup. Safe to call from the CLI and from tests."""
    global _configured

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redact_secrets,
        structlog.processors.StackInfoRenderer(),
    ]
    if fmt == "json":
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    # Third-party libraries log through stdlib; keep them from drowning the run.
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)
    for noisy in ("httpx", "httpcore", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # trafilatura warns on every page it cannot parse, which is routine here.
    logging.getLogger("trafilatura").setLevel(logging.ERROR)
    # pypdf warns per malformed file; those are classified and reported by
    # the fetcher, so the library's own noise adds nothing.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


def bind_run(run_id: str, **extra: Any) -> None:
    """Attach ``run_id`` (and anything else) to every subsequent log line."""
    structlog.contextvars.bind_contextvars(run_id=run_id, **extra)


def clear_run() -> None:
    structlog.contextvars.clear_contextvars()
