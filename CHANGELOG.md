# Changelog

All notable changes to LARSA are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.1.0] - 2026-03-17

### Added
- `tests/__init__.py`, `tests/test_metrics.py`, `tests/test_exceptions.py`,
  `tests/test_config.py`: comprehensive pytest test suite covering all public
  functions in `metrics.py`, the full `LARSAError` hierarchy, and all
  `LARSASettings` defaults and validators.
- `.github/workflows/ci.yml`: three-job CI pipeline — lint (`ruff`),
  type-check (`mypy`), and test (`pytest`) across Python 3.10/3.11/3.12.
- `pyproject.toml`: added `[project.dependencies]`, `[project.optional-dependencies]`
  dev extras, `[tool.pytest.ini_options]` with custom markers,
  `[tool.ruff.lint.per-file-ignores]`, and `[[tool.mypy.overrides]]` for
  third-party stubs.

### Changed
- `erl_run.py`: replaced all `print()` logging with structured `get_logger`
  calls; added comprehensive docstrings and type annotations to `Learner`,
  `Worker`, `EvaluatorProc`, `train_agent`, `valid_agent`, and `run`.
- `erl_agent.py`: added full Google-style docstrings and type annotations to
  all methods; replaced bare `[int]` list type hints with `List[int]`.
- `erl_config.py`: switched from `logging.getLogger` to `get_logger`;
  replaced all `print()` calls with `logger.warning`; added type annotations
  to all methods.
- `data_config.py`: added `input_ary_path`, `label_ary_path`, and
  `predict_net_path` attributes; added module docstring and class docstring.
- `seq_run.py`: added module docstring, class docstring for `SeqData`, and
  function docstrings with type annotations for `_update_network`,
  `train_model`, and `valid_model`.
- `README.md`: expanded with full configuration reference table, metrics
  table, development workflow section, and license section.

---

## [1.0.0] - 2026-03-17

### Added
- `tests/test_logger.py`: full unit-test coverage of `_JsonFormatter`, `_build_handler`,
  and `get_logger` including exception serialisation and fallback behaviour.
- `tests/test_data_config.py`: coverage of `data_config.ConfigData` path construction,
  type guard, and all attribute types.
- `tests/test_metrics_edge_cases.py`: additional edge cases for `sharpe_ratio`,
  `max_drawdown`, `return_over_max_drawdown`, and `cumulative_returns`.
- `tests/test_seq_run_extended.py`: covers `_update_network` gradient clipping and
  `SeqData` with extreme split ratios (0.0 and 1.0).
- `tests/test_deepseek_signals_extended.py`: error-path tests for `_build_client`,
  `_get_client`, `_call_api`, `_call_with_retry`, and `process_news_analysis`.

### Changed
- `trade_simulator.py`: switched from `logging.getLogger` to `get_logger` (project
  JSON logger); all `print()` calls in `check_simulator` replaced with structured
  `logger.info` / `logger.debug` calls.
- `seq_run.py`: switched to `get_logger`; all hardcoded training hyperparameters in
  `train_model` now read from `config.settings`; removed redundant `logging.basicConfig`
  from `__main__`.
- `erl_config.py`: switched to `get_logger` for consistent structured log output.
- `task1_eval.py`: removed redundant `print()` metric output (already emitted via
  structured logger).
- `logger.py`: bare `except Exception` in `get_logger` narrowed to
  `(ImportError, AttributeError, ValueError)`.
- `.github/workflows/ci.yml`: expanded to multi-Python matrix (3.10 + 3.11), added
  `ruff` linting job, `bandit` security-scan job, and `pytest-mock` install.

---

## [0.1.0] - 2025-03-17

### Added
- `deepseek_signals.py`: DeepSeek V3 API signal extractor with exponential-backoff retry and CSV checkpointing.
- `seq_net.py`: `RnnRegNet` combining LSTM and GRU layers with MLP projections for factor mining.
- `erl_agent.py`: `AgentDoubleDQN`, `AgentD3QN`, and `AgentTwinD3QN` off-policy DRL agents.
- `erl_net.py`: `QNetTwin` (Double DQN) and `QNetTwinDuel` (D3QN) Q-network architectures.
- `erl_replay_buffer.py`: Circular off-policy replay buffer supporting vectorized environments.
- `erl_config.py`: `Config` dataclass with on-policy / off-policy hyperparameter sets.
- `task1_ensemble.py`: `Ensemble` class training multiple agents and coordinating majority-vote decisions.
- `task1_eval.py`: Backtest evaluation pipeline for held-out test data.
- `config.py`: Pydantic `LARSASettings` for validated, environment-variable-backed configuration.
- `exceptions.py`: Custom exception hierarchy (`LARSAError`, `DataFetchError`, `ModelError`, `SignalError`, `ConfigError`).
- `logger.py`: Structured logging via structlog.
- `metrics.py`: Sharpe ratio, Sortino ratio, and max drawdown implementations.

[Unreleased]: https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading/releases/tag/v0.1.0
