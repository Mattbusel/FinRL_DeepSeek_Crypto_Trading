"""Tests for the config.py module.

Covers default values, environment-variable overrides, field validation,
and the :class:`LARSASettings` singleton.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestLARSASettingsDefaults:
    """Verify that default values are applied when no env vars are set."""

    def test_deepseek_base_url_default(self) -> None:
        """Default base URL points to the DeepSeek v1 endpoint."""
        from config import LARSASettings

        s = LARSASettings()
        assert s.deepseek_base_url == "https://api.deepseek.com/v1"

    def test_deepseek_model_default(self) -> None:
        """Default model is deepseek-chat."""
        from config import LARSASettings

        s = LARSASettings()
        assert s.deepseek_model == "deepseek-chat"

    def test_temperature_default(self) -> None:
        """Default temperature is 0.0 (deterministic)."""
        from config import LARSASettings

        s = LARSASettings()
        assert s.deepseek_temperature == 0.0

    def test_max_tokens_default(self) -> None:
        """Default max tokens is 300."""
        from config import LARSASettings

        s = LARSASettings()
        assert s.deepseek_max_tokens == 300

    def test_max_retries_default(self) -> None:
        """Default max retries is 5."""
        from config import LARSASettings

        s = LARSASettings()
        assert s.max_retries == 5

    def test_checkpoint_interval_default(self) -> None:
        """Default checkpoint interval is 10."""
        from config import LARSASettings

        s = LARSASettings()
        assert s.checkpoint_interval == 10

    def test_min_confidence_threshold_default(self) -> None:
        """Default confidence threshold is 0.3."""
        from config import LARSASettings

        s = LARSASettings()
        assert s.min_confidence_threshold == 0.3

    def test_log_level_default(self) -> None:
        """Default log level is INFO."""
        from config import LARSASettings

        s = LARSASettings()
        assert s.log_level == "INFO"

    def test_rl_gamma_default(self) -> None:
        """Default RL discount factor is 0.995."""
        from config import LARSASettings

        s = LARSASettings()
        assert s.rl_gamma == 0.995

    def test_starting_cash_default(self) -> None:
        """Default starting cash is 1,000,000."""
        from config import LARSASettings

        s = LARSASettings()
        assert s.starting_cash == 1e6

    def test_data_dir_default(self) -> None:
        """Default data directory is ./data."""
        from config import LARSASettings

        s = LARSASettings()
        assert s.data_dir == "./data"

    def test_output_dir_default(self) -> None:
        """Default output directory is ./output."""
        from config import LARSASettings

        s = LARSASettings()
        assert s.output_dir == "./output"

    def test_retry_backoff_length(self) -> None:
        """Default backoff list has 5 entries matching max_retries."""
        from config import LARSASettings

        s = LARSASettings()
        assert len(s.retry_backoff_seconds) == 5

    def test_rl_net_dims_length(self) -> None:
        """Default RL network dims has 3 layers."""
        from config import LARSASettings

        s = LARSASettings()
        assert len(s.rl_net_dims) == 3


class TestLARSASettingsValidation:
    """Verify that field validators reject out-of-range values."""

    def test_invalid_log_level_raises(self) -> None:
        """An unknown log level raises a validation error."""
        from pydantic import ValidationError
        from config import LARSASettings

        with pytest.raises(ValidationError):
            LARSASettings(log_level="VERBOSE")

    def test_valid_log_levels_accepted(self) -> None:
        """All standard Python log level names are accepted."""
        from config import LARSASettings

        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            s = LARSASettings(log_level=level)
            assert s.log_level == level

    def test_log_level_case_insensitive(self) -> None:
        """Log level is normalised to uppercase."""
        from config import LARSASettings

        s = LARSASettings(log_level="debug")
        assert s.log_level == "DEBUG"

    def test_temperature_out_of_range_raises(self) -> None:
        """Temperature above 2.0 raises a validation error."""
        from pydantic import ValidationError
        from config import LARSASettings

        with pytest.raises(ValidationError):
            LARSASettings(deepseek_temperature=3.0)

    def test_temperature_negative_raises(self) -> None:
        """Negative temperature raises a validation error."""
        from pydantic import ValidationError
        from config import LARSASettings

        with pytest.raises(ValidationError):
            LARSASettings(deepseek_temperature=-0.1)

    def test_max_tokens_zero_raises(self) -> None:
        """Zero max_tokens raises a validation error."""
        from pydantic import ValidationError
        from config import LARSASettings

        with pytest.raises(ValidationError):
            LARSASettings(deepseek_max_tokens=0)

    def test_min_confidence_out_of_range_raises(self) -> None:
        """Confidence threshold above 1.0 raises a validation error."""
        from pydantic import ValidationError
        from config import LARSASettings

        with pytest.raises(ValidationError):
            LARSASettings(min_confidence_threshold=1.5)


class TestSettingsSingleton:
    """Verify that the module-level settings singleton is a LARSASettings."""

    def test_singleton_type(self) -> None:
        """The module-level settings object is a LARSASettings instance."""
        from config import LARSASettings, settings

        assert isinstance(settings, LARSASettings)

    def test_singleton_log_level_valid(self) -> None:
        """The singleton log level is a valid Python log level string."""
        from config import settings

        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        assert settings.log_level in valid
