"""Structured logging.

Every log line is an event name plus typed key/value pairs, never an
interpolated sentence. That makes a run greppable (``event=search_completed``)
and lets the CLI, tests and any future log shipper read the same records.

``run_id`` is bound once per run via contextvars, so nodes deep in the graph do
not have to thread it through their signatures.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False

# Keys whose values must never reach a log sink.
_SECRET_KEYS = frozenset(
    {"api_key", "openai_api_key", "tavily_api_key", "langsmith_api_key", "authorization", "token"}
)


def _redact_secrets(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Defence in depth: keys are held in ``SecretStr``, but a careless
    ``log.info("x", api_key=...)`` should still not leak."""
    for key in list(event_dict):
        if key.lower() in _SECRET_KEYS:
            event_dict[key] = "<redacted>"
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
