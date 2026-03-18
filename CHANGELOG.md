# Changelog

All notable changes to LARSA are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `tests/` directory with pytest suite covering agents, networks, config, exceptions, and data pipeline.
- `pyproject.toml` with package metadata, ruff, mypy, and pytest configuration.
- Google-style docstrings on all public functions and classes.
- Updated CI workflow to run pytest, ruff, and mypy on Python 3.10 and 3.11.
- Rewritten README with ASCII architecture diagram, quickstart, and environment variable reference.

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
