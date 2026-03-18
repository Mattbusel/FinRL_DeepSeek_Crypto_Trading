"""Tests for the exceptions.py module.

Verifies the LARSA custom exception hierarchy: inheritance, message
formatting, cause chaining, and that each subclass is independently
catch-able.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exceptions import (
    ConfigError,
    DataFetchError,
    LARSAError,
    ModelError,
    SignalError,
)


class TestLARSAError:
    """Tests for the base :class:`LARSAError`."""

    def test_message_stored(self) -> None:
        """The message is accessible via str()."""
        err = LARSAError("base error")
        assert "base error" in str(err)

    def test_cause_none_by_default(self) -> None:
        """Cause defaults to None."""
        err = LARSAError("msg")
        assert err.cause is None

    def test_cause_stored(self) -> None:
        """Cause is stored and accessible."""
        original = ValueError("root cause")
        err = LARSAError("wrapped", cause=original)
        assert err.cause is original

    def test_str_with_cause_includes_cause(self) -> None:
        """str() includes the cause type and message when cause is set."""
        cause = RuntimeError("something bad")
        err = LARSAError("outer error", cause=cause)
        text = str(err)
        assert "RuntimeError" in text
        assert "something bad" in text

    def test_str_without_cause_clean(self) -> None:
        """str() without a cause is just the message."""
        err = LARSAError("clean message")
        assert str(err) == "clean message"

    def test_is_exception(self) -> None:
        """LARSAError is a subclass of Exception."""
        assert issubclass(LARSAError, Exception)

    def test_raise_and_catch(self) -> None:
        """LARSAError can be raised and caught."""
        with pytest.raises(LARSAError):
            raise LARSAError("test raise")


class TestSubclasses:
    """Tests that all subclasses inherit LARSAError and can be caught."""

    @pytest.mark.parametrize(
        "exc_class",
        [DataFetchError, ModelError, SignalError, ConfigError],
    )
    def test_inherits_larsa_error(self, exc_class) -> None:
        """Each subclass is a subclass of LARSAError."""
        assert issubclass(exc_class, LARSAError)

    @pytest.mark.parametrize(
        "exc_class",
        [DataFetchError, ModelError, SignalError, ConfigError],
    )
    def test_caught_by_larsa_error(self, exc_class) -> None:
        """Each subclass instance is caught by except LARSAError."""
        with pytest.raises(LARSAError):
            raise exc_class("test")

    @pytest.mark.parametrize(
        "exc_class",
        [DataFetchError, ModelError, SignalError, ConfigError],
    )
    def test_caught_by_own_class(self, exc_class) -> None:
        """Each subclass instance is caught by its own class."""
        with pytest.raises(exc_class):
            raise exc_class("test")

    @pytest.mark.parametrize(
        "exc_class",
        [DataFetchError, ModelError, SignalError, ConfigError],
    )
    def test_message_preserved(self, exc_class) -> None:
        """Message is preserved in subclass instances."""
        err = exc_class("specific message")
        assert "specific message" in str(err)

    @pytest.mark.parametrize(
        "exc_class",
        [DataFetchError, ModelError, SignalError, ConfigError],
    )
    def test_cause_preserved(self, exc_class) -> None:
        """Cause is preserved in subclass instances."""
        cause = OSError("io issue")
        err = exc_class("wrapper", cause=cause)
        assert err.cause is cause
        # IOError is an alias for OSError in Python 3.3+; accept either name
        assert "OSError" in str(err) or "IOError" in str(err)


class TestExceptionIsolation:
    """Ensure catching one subclass does not catch another."""

    def test_data_fetch_not_caught_as_model_error(self) -> None:
        """DataFetchError is not caught by except ModelError."""
        with pytest.raises(DataFetchError):
            try:
                raise DataFetchError("data")
            except ModelError:
                pass  # should not reach here

    def test_signal_error_not_caught_as_config_error(self) -> None:
        """SignalError is not caught by except ConfigError."""
        with pytest.raises(SignalError):
            try:
                raise SignalError("signal")
            except ConfigError:
                pass
