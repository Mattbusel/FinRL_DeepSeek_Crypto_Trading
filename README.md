[![CI](https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading/actions/workflows/ci.yml/badge.svg)](https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# LARSA: LLM-Augmented Regime-Switching Agent

LARSA is a hybrid reinforcement learning and large language model trading system for Bitcoin.
It extracts structured sentiment and risk signals from news articles using DeepSeek V3, feeds
those signals through a recurrent factor model, and passes the combined feature vector to a
D3QN-based ensemble of trading agents.

---

## Architecture Overview

```
BTC News Articles
       |
       v
 deepseek_signals.py          -- DeepSeek V3 API: sentiment + risk scores
       |
       v
   seq_data.py                -- Alpha-101 factor computation + CSV -> NPY
       |
       v
    seq_run.py                -- RNN regression (LSTM + GRU) training on factors
       |
       v
BTC_1sec_predict.npy          -- 8-dimensional factor predictions per second
       |
       v
 trade_simulator.py           -- Vectorised environment (TradeSimulator / EvalTradeSimulator)
   state = [position, holding, 8 factors, sentiment, risk]
       |
       v
 task1_ensemble.py            -- Trains AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
       |
       v
 task1_eval.py                -- Majority-vote ensemble evaluation, Sharpe / MDD / RoMaD
```

All configuration is loaded from environment variables via `config.py` (backed by
`pydantic-settings`). Structured JSON logs are emitted by every module through `logger.py`.
Custom exceptions (`exceptions.py`) form a hierarchy rooted at `LARSAError`.

---

## Quickstart

### Prerequisites

- Python 3.10 or later
- A DeepSeek API key (only required for signal extraction)

### Installation

```bash
git clone https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading.git
cd FinRL_DeepSeek_Crypto_Trading

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

If the editable install fails due to missing `torch` extras, install manually:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov ruff mypy pydantic pydantic-settings openai
```

### Run the Pipeline

**Step 1 -- Extract news signals** (requires `DEEPSEEK_API_KEY`):

```bash
export DEEPSEEK_API_KEY=sk-...
python deepseek_signals.py \
    --input ./data/news_train.csv \
    --output ./data/BTC_1sec_with_sentiment_risk_train.csv
```

**Step 2 -- Preprocess market data and train the RNN factor model**:

```bash
python seq_run.py          # accepts optional GPU id: python seq_run.py 0
```

**Step 3 -- Train the agent ensemble**:

```bash
python task1_ensemble.py   # CPU by default; pass GPU id as first arg
```

**Step 4 -- Evaluate on held-out data**:

```bash
python task1_eval.py
```

---

## Configuration Reference

All settings can be overridden via environment variables.  The defaults below are active when
the variable is not set.

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | `""` | API key for the DeepSeek inference endpoint. |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | DeepSeek endpoint URL. |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model identifier. |
| `DEEPSEEK_TEMPERATURE` | `0.0` | Sampling temperature (0 = deterministic). |
| `MAX_RETRIES` | `5` | API retry attempts before giving up. |
| `MIN_CONFIDENCE_THRESHOLD` | `0.3` | Minimum signal confidence to accept. |
| `CHECKPOINT_INTERVAL` | `10` | Save a progress checkpoint every N rows. |
| `RNN_BATCH_SIZE` | `256` | Mini-batch size for RNN training. |
| `RNN_MID_DIM` | `128` | Hidden dimension of the RNN. |
| `RNN_NUM_LAYERS` | `4` | Number of stacked recurrent layers. |
| `RNN_EPOCHS` | `256` | Training epochs for the RNN. |
| `RL_LEARNING_RATE` | `2e-6` | Learning rate for actor/critic networks. |
| `RL_BATCH_SIZE` | `512` | Mini-batch size for RL network updates. |
| `RL_GAMMA` | `0.995` | Discount factor for future rewards. |
| `RL_BREAK_STEP` | `32` | Stop RL training after this many steps. |
| `NUM_SIMS` | `4096` | Parallel simulation environments during training. |
| `STARTING_CASH` | `1000000` | Initial cash for evaluation episodes. |
| `DATA_DIR` | `./data` | Directory for price and news data files. |
| `OUTPUT_DIR` | `./output` | Directory for model checkpoints and artefacts. |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG/INFO/WARNING/ERROR/CRITICAL). |

---

## Project Structure

```
.
+-- config.py                  # Pydantic-settings configuration singleton
+-- data_config.py             # Resolved data file paths
+-- deepseek_signals.py        # DeepSeek V3 news signal extraction pipeline
+-- ensemble_npy_evaluator.py  # Offline NPY result evaluator
+-- erl_agent.py               # AgentDoubleDQN, AgentD3QN, AgentTwinD3QN
+-- erl_config.py              # Config dataclass and build_env utility
+-- erl_evaluator.py           # Training evaluator and learning curve plotter
+-- erl_net.py                 # QNetTwin, QNetTwinDuel, build_mlp
+-- erl_replay_buffer.py       # Off-policy experience replay buffer
+-- erl_run.py                 # Multi-process Learner/Worker/Evaluator pipeline
+-- exceptions.py              # LARSAError hierarchy
+-- logger.py                  # JSON structured logging factory
+-- metrics.py                 # Sharpe ratio, max drawdown, RoMaD
+-- seq_data.py                # Alpha-101 factor computation from CSV
+-- seq_net.py                 # RnnRegNet (LSTM + GRU regression network)
+-- seq_record.py              # RNN training evaluator and curve plotter
+-- seq_run.py                 # RNN training and inference entry point
+-- task1_ensemble.py          # Ensemble training pipeline
+-- task1_eval.py              # Ensemble evaluation pipeline
+-- trade_simulator.py         # Vectorised trade environment
+-- tests/                     # pytest test suite
+-- pyproject.toml             # Project metadata, dependencies, tool config
+-- requirements.txt           # Pinned dependency list
+-- CHANGELOG.md               # Version history
```

---

## Running Tests

```bash
pytest tests/ -q --tb=short
```

With coverage:

```bash
pytest tests/ --cov=. --cov-report=term-missing -q
```

---

## Code Quality

```bash
# Linting
ruff check . --select E,F,W,I,UP,B --ignore E501,B008

# Type checking
mypy --ignore-missing-imports config.py exceptions.py logger.py metrics.py

# Security scan
bandit -r . --exclude tests,output,data -ll
```

---

## Contributing

1. Fork the repository and create a branch from `main`.
2. Install dev dependencies: `pip install -e ".[dev]"`.
3. Write tests for any new logic in `tests/`.
4. Ensure `ruff check` and `pytest` pass before opening a pull request.
5. Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## Contact

Author: Matthew C. Busel
Email: [mattbusel@gmail.com](mailto:mattbusel@gmail.com)
GitHub: [github.com/mattbusel](https://github.com/mattbusel)
