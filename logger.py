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


class _StructuredLogger:
    """Thin wrapper around :class:`logging.Logger` that accepts structlog-style calls.

    Translates ``log.info("event", key=value, ...)`` into the standard
    ``logger.info("event", extra={"key": value, ...})`` calling convention so
    that structured fields are captured by :class:`_JsonFormatter`.

    Args:
        logger: Underlying :class:`logging.Logger` instance to delegate to.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        self._logger.log(level, msg, extra=kwargs if kwargs else None)

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log at DEBUG level with optional structured fields."""
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log at INFO level with optional structured fields."""
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log at WARNING level with optional structured fields."""
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log at ERROR level with optional structured fields."""
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        """Log at CRITICAL level with optional structured fields."""
        self._log(logging.CRITICAL, msg, **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        """Log at ERROR level and include the current exception traceback."""
        self._logger.exception(msg, extra=kwargs if kwargs else None)

    # Delegate attribute access (e.g. .setLevel, .handlers) to the inner logger.
    def __getattr__(self, item: str) -> Any:
        return getattr(self._logger, item)


def get_logger(name: str) -> _StructuredLogger:
    """Return a named structured logger configured with JSON output.

    The log level is taken from :data:`config.settings.log_level`.  Repeated
    calls with the same *name* return a wrapper around the same underlying
    :class:`logging.Logger` instance (standard Python behaviour).

    Callers may use structlog-style syntax::

        log = get_logger(__name__)
        log.info("pipeline.start", rows=1024)

    Args:
        name: Dotted module name, typically ``__name__``.

    Returns:
        A :class:`_StructuredLogger` that emits JSON-formatted records to stdout.
    """
    # Import here to avoid circular imports during package initialisation.
    try:
        from config import settings
        level_name = settings.log_level
    except (ImportError, AttributeError, ValueError):
        level_name = "INFO"

    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_build_handler())
    logger.setLevel(level)
    logger.propagate = False
    return _StructuredLogger(logger)
