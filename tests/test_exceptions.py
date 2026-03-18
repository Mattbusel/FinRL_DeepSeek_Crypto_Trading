"""Tests for the custom exception hierarchy in exceptions.py."""

from __future__ import annotations

import pytest

from exceptions import (
    ConfigError,
    DataFetchError,
    LARSAError,
    ModelError,
    SignalError,
)


class TestLARSAError:
    def test_message_is_stored(self):
        err = LARSAError("base error")
        assert str(err) == "base error"

    def test_cause_none_by_default(self):
        err = LARSAError("no cause")
        assert err.cause is None

    def test_cause_is_stored(self):
        inner = ValueError("inner")
        err = LARSAError("outer", cause=inner)
        assert err.cause is inner

    def test_str_with_cause_contains_both_messages(self):
        inner = ValueError("inner detail")
        err = LARSAError("outer message", cause=inner)
        rendered = str(err)
        assert "outer message" in rendered
        assert "inner detail" in rendered

    def test_is_exception(self):
        assert issubclass(LARSAError, Exception)

    def test_can_raise_and_catch(self):
        with pytest.raises(LARSAError, match="raised"):
            raise LARSAError("raised")


class TestDataFetchError:
    def test_is_larsa_error(self):
        assert issubclass(DataFetchError, LARSAError)

    def test_message_stored(self):
        err = DataFetchError("file not found")
        assert "file not found" in str(err)

    def test_cause_propagated(self):
        inner = OSError("disk error")
        err = DataFetchError("cannot load data", cause=inner)
        assert err.cause is inner
        assert "disk error" in str(err)

    def test_can_raise_and_catch_as_larsa(self):
        with pytest.raises(LARSAError):
            raise DataFetchError("data missing")


class TestModelError:
    def test_is_larsa_error(self):
        assert issubclass(ModelError, LARSAError)

    def test_message_stored(self):
        err = ModelError("checkpoint missing")
        assert "checkpoint missing" in str(err)

    def test_cause_propagated(self):
        inner = FileNotFoundError("actor.pth")
        err = ModelError("model load failed", cause=inner)
        assert err.cause is inner

    def test_can_raise_and_catch_as_larsa(self):
        with pytest.raises(LARSAError):
            raise ModelError("bad model")


class TestSignalError:
    def test_is_larsa_error(self):
        assert issubclass(SignalError, LARSAError)

    def test_message_stored(self):
        err = SignalError("api failed")
        assert "api failed" in str(err)

    def test_cause_propagated(self):
        inner = RuntimeError("connection refused")
        err = SignalError("deepseek unreachable", cause=inner)
        assert err.cause is inner
        assert "connection refused" in str(err)

    def test_can_raise_and_catch_as_larsa(self):
        with pytest.raises(LARSAError):
            raise SignalError("signal extraction failed")


class TestConfigError:
    def test_is_larsa_error(self):
        assert issubclass(ConfigError, LARSAError)

    def test_message_stored(self):
        err = ConfigError("invalid learning rate")
        assert "invalid learning rate" in str(err)

    def test_cause_propagated(self):
        inner = KeyError("DEEPSEEK_API_KEY")
        err = ConfigError("missing env var", cause=inner)
        assert err.cause is inner

    def test_can_raise_and_catch_as_larsa(self):
        with pytest.raises(LARSAError):
            raise ConfigError("bad config")

    def test_can_catch_specifically(self):
        with pytest.raises(ConfigError):
            raise ConfigError("specific")
