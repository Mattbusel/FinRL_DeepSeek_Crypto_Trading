"""Structured JSON logging factory for the LARSA trading system.

Every module should obtain its logger via :func:`get_logger` rather than
calling :func:`logging.getLogger` directly.  This ensures a consistent JSON
format across the entire application and respects the ``log_level`` setting
defined in :mod:`config`.

Example::

    from logger import get_logger
    log = get_logger(__name__)
    log.info("pipeline.start", rows=1024)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Each record includes ``timestamp``, ``level``, ``logger``, ``message``,
    and any extra key-value pairs attached via the ``extra`` argument.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach any structured fields passed via extra={}
        skip = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs", "message",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "thread", "threadName", "exc_info", "exc_text",
        }
        for key, value in record.__dict__.items():
            if key not in skip:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _build_handler(stream: Any = sys.stdout) -> logging.StreamHandler:
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_JsonFormatter())
    return handler


def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured with JSON output.

    The log level is taken from :data:`config.settings.log_level`.  Repeated
    calls with the same *name* return the same :class:`logging.Logger` instance
    (standard Python behaviour).

    Args:
        name: Dotted module name, typically ``__name__``.

    Returns:
        A :class:`logging.Logger` that emits JSON-formatted records to stdout.
    """
    # Import here to avoid circular imports during package initialisation.
    try:
        from config import settings
        level_name = settings.log_level
    except Exception:
        level_name = "INFO"

    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_build_handler())
    logger.setLevel(level)
    logger.propagate = False
    return logger
