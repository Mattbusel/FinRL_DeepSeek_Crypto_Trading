"""YAML-based configuration loader for the LARSA trading system.

Reads ``config.yaml`` and merges values into the pydantic settings singleton,
allowing YAML to serve as the primary configuration source while still
respecting environment variable overrides.

Example::

    from config_yaml import load_config
    cfg = load_config()
    print(cfg.deepseek_model)
    print(cfg.paper_trade_starting_cash)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_YAML_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def _load_yaml_file(path: str = _YAML_PATH) -> dict[str, Any]:
    """Parse a YAML configuration file.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ImportError: If PyYAML is not installed.
        ValueError: If the YAML content cannot be parsed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to use load_config(). Install it with:\n"
            "  pip install pyyaml"
        ) from exc

    with open(path, encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ValueError(f"Cannot parse YAML file '{path}': {exc}") from exc

    return data or {}


# ---------------------------------------------------------------------------
# Extended settings with YAML-sourced paper trading / backtest params
# ---------------------------------------------------------------------------


class YAMLSettings:
    """Merged configuration: YAML defaults + pydantic environment settings.

    Exposes all fields from :class:`config.LARSASettings` plus the extra
    paper-trading and backtesting fields defined in ``config.yaml``.

    Args:
        yaml_path: Path to the YAML file.  Defaults to ``config.yaml`` next
            to this module.
    """

    def __init__(self, yaml_path: str = _YAML_PATH) -> None:
        self._raw = _load_yaml_file(yaml_path)

        # Pull the pydantic singleton (env vars already applied)
        from config import settings as _base_settings
        self._base = _base_settings

    def __getattr__(self, name: str) -> Any:
        """Attribute lookup: pydantic settings → YAML → KeyError."""
        # 1. Try pydantic settings (env-var overrides already baked in)
        try:
            return getattr(self._base, name)
        except AttributeError:
            pass
        # 2. Try YAML dict
        if name in self._raw:
            return self._raw[name]
        raise AttributeError(f"YAMLSettings has no attribute '{name}'")

    # Convenience properties for paper trading
    @property
    def paper_trade_starting_cash(self) -> float:
        env = os.environ.get("PAPER_TRADE_STARTING_CASH")
        return float(env) if env else float(self._raw.get("paper_trade_starting_cash", 100_000.0))

    @property
    def paper_trade_interval_seconds(self) -> float:
        env = os.environ.get("PAPER_TRADE_INTERVAL_SECONDS")
        return float(env) if env else float(self._raw.get("paper_trade_interval_seconds", 10.0))

    @property
    def paper_trade_slippage(self) -> float:
        env = os.environ.get("PAPER_TRADE_SLIPPAGE")
        return float(env) if env else float(self._raw.get("paper_trade_slippage", 0.0005))

    @property
    def paper_trade_trade_fraction(self) -> float:
        env = os.environ.get("PAPER_TRADE_TRADE_FRACTION")
        return float(env) if env else float(self._raw.get("paper_trade_trade_fraction", 0.10))

    @property
    def paper_trade_short_window(self) -> int:
        env = os.environ.get("PAPER_TRADE_SHORT_WINDOW")
        return int(env) if env else int(self._raw.get("paper_trade_short_window", 5))

    @property
    def paper_trade_long_window(self) -> int:
        env = os.environ.get("PAPER_TRADE_LONG_WINDOW")
        return int(env) if env else int(self._raw.get("paper_trade_long_window", 20))

    @property
    def paper_trade_log_file(self) -> str:
        env = os.environ.get("PAPER_TRADE_LOG_FILE")
        return env if env else str(self._raw.get("paper_trade_log_file", "paper_trades.jsonl"))

    # Convenience properties for backtesting
    @property
    def backtest_slippage(self) -> float:
        env = os.environ.get("BACKTEST_SLIPPAGE")
        return float(env) if env else float(self._raw.get("backtest_slippage", 0.0005))

    @property
    def backtest_trade_fraction(self) -> float:
        env = os.environ.get("BACKTEST_TRADE_FRACTION")
        return float(env) if env else float(self._raw.get("backtest_trade_fraction", 0.10))

    @property
    def backtest_short_window(self) -> int:
        env = os.environ.get("BACKTEST_SHORT_WINDOW")
        return int(env) if env else int(self._raw.get("backtest_short_window", 5))

    @property
    def backtest_long_window(self) -> int:
        env = os.environ.get("BACKTEST_LONG_WINDOW")
        return int(env) if env else int(self._raw.get("backtest_long_window", 20))

    def as_dict(self) -> dict[str, Any]:
        """Return a flat dictionary of all resolved settings."""
        result = dict(self._raw)
        # Overlay env-aware pydantic settings
        try:
            result.update(self._base.model_dump())
        except Exception:
            pass
        return result


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def load_config(yaml_path: str = _YAML_PATH) -> YAMLSettings:
    """Load YAML configuration and merge with environment variables.

    Args:
        yaml_path: Path to ``config.yaml``.

    Returns:
        A :class:`YAMLSettings` instance with all settings accessible as
        attributes.

    Example::

        from config_yaml import load_config
        cfg = load_config()
        print(cfg.deepseek_model)                 # from env or YAML
        print(cfg.paper_trade_starting_cash)      # YAML-only field
    """
    return YAMLSettings(yaml_path=yaml_path)
