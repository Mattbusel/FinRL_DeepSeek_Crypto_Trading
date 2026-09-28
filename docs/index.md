# LARSA: LLM-Augmented Regime-Switching Agent

LARSA is a FinRL contest Task 1 research entry: three DQN agents (D3QN, DoubleDQN,
TwinD3QN) trade Bitcoin on one-second limit-order-book data, with DeepSeek V3
news sentiment and risk scores in their state. It is research code, not
financial advice, and it has never traded real money.

## Quick Navigation

- [Reference](REFERENCE.md) - Files, configuration, metrics, tests
- [Architecture](architecture.md) - System design and data flow
- [API Reference](api-reference.md) - Auto-generated module documentation

## Quickstart

```bash
git clone https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading
cd FinRL_DeepSeek_Crypto_Trading
pip install -r requirements-demo.txt && python demo.py
```

`demo.py` replays the trained agents over 4 bundled hours of the contest test
data. The full data download, training and 68-hour replay are in the README.

## Project Layout

```
deepseek_signals.py    DeepSeek V3 signal extraction
seq_data.py            Alpha101 factor generation and data loading
seq_run.py             RNN regression model training
seq_net.py             RNN architecture (LSTM + GRU)
task1_ensemble.py      Multi-agent ensemble training orchestration
task1_eval.py          Evaluation and performance metrics
metrics.py             Sharpe ratio, max drawdown, RoMaD
config.py              Centralised pydantic-settings configuration
exceptions.py          Typed exception hierarchy
logger.py              Structured JSON logging factory
tests/                 pytest test suite
demo.py                One-command demo on bundled sample data
docs/                  This documentation
```
