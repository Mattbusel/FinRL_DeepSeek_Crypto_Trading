"""Custom exception hierarchy for the LARSA trading system.

All project-specific exceptions derive from LARSAError, allowing callers to
catch the entire family with a single except clause when appropriate.
"""

from __future__ import annotations


class LARSAError(Exception):
    """Base exception for all LARSA trading system errors.

    Args:
        message: Human-readable description of the error.
        cause: Optional underlying exception that triggered this one.
    """

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        base = super().__str__()
        if self.cause is not None:
            return f"{base} (caused by: {type(self.cause).__name__}: {self.cause})"
        return base


class DataFetchError(LARSAError):
    """Raised when market or news data cannot be loaded or fetched.

    Examples include missing CSV files, corrupt numpy arrays, and network
    failures when downloading price data.
    """


class ModelError(LARSAError):
    """Raised when a model cannot be loaded, saved, or produces invalid output.

    Examples include missing checkpoint files, shape mismatches, and NaN
    outputs from the RNN regression network.
    """


class SignalError(LARSAError):
    """Raised when sentiment or risk signal extraction fails.

    Examples include DeepSeek API failures, malformed JSON responses, missing
    required keys, and repeated rate-limit exhaustion.
    """


class ConfigError(LARSAError):
    """Raised when configuration is invalid or missing required fields.

    Examples include unset environment variables, out-of-range hyperparameters,
    and incompatible combinations of settings.
    """
