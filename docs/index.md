# LARSA: LLM-Augmented Regime-Switching Agent

LARSA is a hybrid reinforcement-learning and large-language-model trading system
that ingests real-time Bitcoin news, extracts structured sentiment and risk signals
via DeepSeek V3, and routes those signals through a D3QN ensemble agent to produce
live trade decisions.

## Quick Navigation

- [Architecture](architecture.md) - System design and data flow
- [API Reference](api-reference.md) - Auto-generated module documentation

## Quickstart

```bash
git clone https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading
cd FinRL_DeepSeek_Crypto_Trading
pip install -e ".[dev]"
export DEEPSEEK_API_KEY="your_key_here"

# Step 1: extract news signals
python deepseek_signals.py --input ./data/news_train.csv --output ./data/news_with_signals.csv

# Step 2: train the RNN factor model
python seq_run.py

# Step 3: train the agent ensemble
python task1_ensemble.py

# Step 4: evaluate on test data
python task1_eval.py
```

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
docs/                  This documentation
```
