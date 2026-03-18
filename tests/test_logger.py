"""Tests for logger.py: _JsonFormatter, _build_handler, get_logger, and _StructuredLogger."""

from __future__ import annotations

import json
import logging
import sys
from io import StringIO
from unittest.mock import patch

from logger import _build_handler, _JsonFormatter, get_logger


class TestJsonFormatter:
    def test_output_is_valid_json(self):
        """Each formatted record must be parseable JSON."""
        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_required_fields_present(self):
        """Formatted record must contain timestamp, level, logger, message."""
        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="my.module",
            level=logging.WARNING,
            pathname=__file__,
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "message" in parsed

    def test_level_name_correct(self):
        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="x",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="err",
            args=(),
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["level"] == "ERROR"

    def test_message_content_correct(self):
        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="x",
            level=logging.DEBUG,
            pathname=__file__,
            lineno=1,
            msg="the quick brown fox",
            args=(),
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["message"] == "the quick brown fox"

    def test_extra_fields_included(self):
        """Extra key-value pairs passed via extra= appear in the JSON output."""
        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="x",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="with extras",
            args=(),
            exc_info=None,
        )
        record.rows = 42
        record.status = "ok"
        parsed = json.loads(formatter.format(record))
        assert parsed.get("rows") == 42
        assert parsed.get("status") == "ok"

    def test_exception_info_serialised(self):
        """Exception info is captured as a string in the JSON output."""
        formatter = _JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="x",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="oops",
            args=(),
            exc_info=exc_info,
        )
        parsed = json.loads(formatter.format(record))
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]

    def test_timestamp_is_iso_format(self):
        """Timestamp must be a valid ISO-8601 string (UTC)."""
        from datetime import datetime

        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="x",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="ts",
            args=(),
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        dt = datetime.fromisoformat(parsed["timestamp"])
        assert dt.tzinfo is not None


class TestBuildHandler:
    def test_returns_stream_handler(self):
        handler = _build_handler(stream=StringIO())
        assert isinstance(handler, logging.StreamHandler)

    def test_handler_has_json_formatter(self):
        handler = _build_handler(stream=StringIO())
        assert isinstance(handler.formatter, _JsonFormatter)


class TestGetLogger:
    def test_returns_structured_logger(self):
        """get_logger should return a _StructuredLogger (or any callable logger)."""
        log = get_logger("test.module")
        # Must have the standard log-level methods
        for method in ("debug", "info", "warning", "error", "critical"):
            assert callable(getattr(log, method)), f"missing method: {method}"

    def test_same_name_wraps_same_underlying_logger(self):
        """Two calls with the same name should wrap the same underlying Logger."""
        a = get_logger("shared.module.x")
        b = get_logger("shared.module.x")
        # Both wrappers delegate to the same underlying logging.Logger instance
        assert a._logger is b._logger  # type: ignore[attr-defined]

    def test_logger_has_handler(self):
        """The underlying logger should have at least one handler attached."""
        log = get_logger("handler.test.x")
        # Access the internal _logger attribute (delegate pattern)
        inner = getattr(log, "_logger", log)
        assert len(inner.handlers) >= 1

    def test_logger_does_not_propagate(self):
        """The underlying logger should not propagate to the root logger."""
        log = get_logger("no.propagate.x")
        inner = getattr(log, "_logger", log)
        assert inner.propagate is False

    def test_info_method_emits_without_error(self):
        """Calling .info() should not raise."""
        log = get_logger("emit.test")
        log.info("test message", key="value")  # must not raise

    def test_warning_method_emits_without_error(self):
        log = get_logger("emit.warn")
        log.warning("warn msg", detail="x")

    def test_error_method_emits_without_error(self):
        log = get_logger("emit.err")
        log.error("err msg")

    def test_debug_method_emits_without_error(self):
        log = get_logger("emit.debug")
        log.debug("debug msg", a=1, b=2)

    def test_fallback_on_import_error(self):
        """get_logger must still work when config module is unavailable."""
        with patch.dict("sys.modules", {"config": None}):
            import importlib

            import logger as logger_mod

            importlib.reload(logger_mod)
            log = logger_mod.get_logger("fallback.test")
            assert callable(log.info)

    def test_logger_level_is_set(self):
        """Logger level should be one of the standard levels."""
        log = get_logger("level.test.x")
        inner = getattr(log, "_logger", log)
        assert inner.level in (
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        )
