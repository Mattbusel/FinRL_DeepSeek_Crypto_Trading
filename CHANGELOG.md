# Changelog

All notable changes to LARSA are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-03-17

### Added

- Full Google-style docstrings (Args/Returns/Raises) on `TradeSimulator`,
  `EvalTradeSimulator`, `check_simulator`, `Ensemble`, `can_buy`, `winloss`,
  `train_agent`, `run`, and `ConfigData`.
- `logging.getLogger` integration in `trade_simulator.py` replacing all
  `print()` calls inside the simulator internals.
- Type hints on all public method signatures in `trade_simulator.py`,
  `task1_ensemble.py`, `data_config.py`, and `task1_eval.py`.
- `TypeError` guard in `ConfigData.__init__` for non-string `data_dir`.
- `ValueError` guard in `Ensemble.__init__` for empty `agent_classes`.
- Specific `ModelError` (replacing bare `AssertionError`) for invalid
  environment state shape and type in `train_agent`.
- `from __future__ import annotations` in all modified modules.

### Changed

- `trade_simulator.py`: all `print()` calls replaced with `logger.info` /
  `logger.debug`; docstrings added to every method.
- `task1_ensemble.py`: bare `print(";;;", ...)` statements replaced with
  `log.debug`; `can_buy` and `winloss` return early instead of nested if/else.
- `data_config.py`: module docstring and class docstring added.

---

## [0.1.0] - 2026-03-17

### Added

- Custom exception hierarchy (`LARSAError`, `DataFetchError`, `ModelError`,
  `SignalError`, `ConfigError`) in `exceptions.py`.
- Centralised configuration module (`config.py`) using `pydantic-settings`
  with environment-variable binding for all hyperparameters.
- Structured JSON logging factory (`logger.py`) replacing all `print()` and
  bare `logging.basicConfig` calls.
- Full type annotations and Google-style docstrings across all public modules:
  `deepseek_signals.py`, `metrics.py`, `task1_ensemble.py`, `task1_eval.py`.
- `tests/` directory with `pytest` tests covering:
  - `deepseek_signals.py`: signal extraction, API error handling, edge cases,
    checkpoint round-trips.
  - `seq_run.py`: `SeqData` initialisation, data splitting, tensor shapes,
    NaN replacement, missing-file code path.
  - `task1_ensemble.py`: `can_buy`, `winloss`, `_majority_vote`,
    `save_ensemble` error propagation.
  - `task1_eval.py`: evaluator construction, `load_agents` error handling,
    `_ensemble_action`, performance metrics.
  - `conftest.py` with shared fixtures and a `mock_deepseek_client` that never
    hits the real DeepSeek API.
- `pyproject.toml` with project metadata, dependency list, and tool
  configuration for `pytest`, `ruff`, and `mypy`.
- Updated `.github/workflows/ci.yml` with multi-Python-version matrix (3.10,
  3.11), `pytest --cov`, `mypy`, `ruff`, `bandit`, and pip dependency caching.
- `mkdocs.yml` and `docs/` with `index.md`, `architecture.md`, and
  `api-reference.md` for auto-generated API documentation.
- Rewritten `README.md` with project overview, architecture diagram,
  quickstart guide, configuration reference, and development guide.

### Changed

- `deepseek_signals.py`: replaced bare `except` blocks with typed
  `SignalError` raises; replaced `print()` with structured logger;
  replaced module-level magic numbers with `settings` references;
  `client` is now lazily initialised so tests can patch without side effects.
- `metrics.py`: added `numpy` array coercion, empty-input guard raising
  `SignalError`, and full type annotations.
- `task1_ensemble.py`: replaced `print()` with structured logger; sourced
  all hyperparameters from `config.settings`; added type annotations and
  docstrings; `ModelError` and `DataFetchError` replace bare exceptions.
- `task1_eval.py`: replaced `print()` with structured logger; added type
  annotations and docstrings; hardcoded dataset path replaced with
  `os.path.join(__file__, ...)` relative resolution; `ModelError` and
  `DataFetchError` raised on failures.

### Fixed

- Hardcoded absolute Windows dataset path in `task1_eval.py` replaced with
  a portable relative path.
- `deepseek_signals.py` no longer silently swallows API errors; all failures
  surface as logged warnings or typed exceptions.

---

[0.1.0]: https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading/releases/tag/v0.1.0
