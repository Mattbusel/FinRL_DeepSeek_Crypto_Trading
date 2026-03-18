# LARSA: LLM-Augmented Regime-Switching Agent

A hybrid reinforcement learning and large language model trading system for
Bitcoin. LARSA extracts structured sentiment and risk signals from BTC news via
the DeepSeek V3 API, mines predictive factors with a recurrent neural network,
and trains an ensemble of DQN-family agents that vote on trade decisions.

---

## System Requirements

- Python 3.10, 3.11, or 3.12
- pip >= 23
- CUDA-capable GPU (optional; all components run on CPU)
- Internet access for DeepSeek API calls during signal extraction
- ~4 GB disk space for data, model checkpoints, and dependencies

---

## Architecture

```
                     LARSA System Architecture
  +---------------+    +---------------------+    +--------------+
  |  BTC News     |    | DeepSeek V3 Signals |    |  RNN Factor  |
  |  CSV data     |--->|  (sentiment + risk) |--->|    Miner     |
  +---------------+    +---------------------+    +------+-------+
                                                         |
                             +---------------------------+
                             v
                    +-----------------+
                    |  TradeSimulator |<--- BTC price ticks
                    |  Environment   |
                    +--------+--------+
                             |
             +---------------+---------------+
             v               v               v
        +---------+   +----------+   +--------------+
        | AgentD3QN|  |AgentDDQN |   |AgentTwinD3QN |
        +----+----+   +----+-----+   +------+-------+
             +---------------+---------------+
                             v
                    +-----------------+
                    |    Ensemble     |  majority-vote coordinator
                    |  Coordinator   |  regime-aware weighting
                    +--------+--------+
                             v
                    +-----------------+
                    |   Evaluation    |  Sharpe, drawdown, RoMaD
                    +-----------------+
```

Detailed module flow:

```
deepseek_signals.py         Calls DeepSeek V3 API, extracts
(sentiment + risk JSON)     sentiment_score and risk_score per article
        |
        v
seq_net.py / seq_run.py     RnnRegNet trained on Alpha101 + news signals
(recurrent factor miner)    outputs predictive feature vectors
        |
        v
erl_agent.py                AgentD3QN / AgentDoubleDQN / AgentTwinD3QN
(DQN-family agents)         each trained independently on TradeSimulator-v0
        |
        v
task1_ensemble.py           Majority-vote ensemble coordinator,
(Ensemble class)            regime-aware signal weighting
        |
        v
task1_eval.py               Backtest on held-out data, reports
(evaluation)                Sharpe, drawdown, return metrics
```

---

## Installation

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading.git
cd FinRL_DeepSeek_Crypto_Trading

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

To also install development tools (pytest, mypy, ruff, black, isort):

```bash
pip install -e ".[dev]"
```

### 3. Set API keys

```bash
export DEEPSEEK_API_KEY="your-key-here"
```

Or create a `.env` file in the project root:

```
DEEPSEEK_API_KEY=your-key-here
```

---

## Configuration Reference

All settings are read from environment variables (or a `.env` file) and
validated at startup via `config.py`. Defaults are shown below.

| Variable | Required | Default | Description |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | Yes | (none) | API key for the DeepSeek inference endpoint |
| `DEEPSEEK_BASE_URL` | No | `https://api.deepseek.com/v1` | Endpoint URL |
| `DEEPSEEK_MODEL` | No | `deepseek-chat` | Model identifier |
| `DEEPSEEK_TEMPERATURE` | No | `0.0` | Sampling temperature (0 = deterministic) |
| `DEEPSEEK_MAX_TOKENS` | No | `300` | Max tokens per completion |
| `MAX_RETRIES` | No | `5` | API retry attempts before giving up |
| `CHECKPOINT_INTERVAL` | No | `10` | Save progress every N rows |
| `MIN_CONFIDENCE_THRESHOLD` | No | `0.3` | Discard signals below this confidence |
| `NUM_SIMS` | No | `4096` | Parallel simulation environments during training |
| `RL_BREAK_STEP` | No | `32` | Stop RL training after this many steps |
| `RL_GAMMA` | No | `0.995` | Discount factor for future rewards |
| `RL_LEARNING_RATE` | No | `2e-6` | Actor/critic learning rate |
| `RL_BATCH_SIZE` | No | `512` | Mini-batch size for RL updates |
| `DATA_DIR` | No | `./data` | Directory containing price and news data |
| `OUTPUT_DIR` | No | `./output` | Directory for model checkpoints |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity: DEBUG, INFO, WARNING, ERROR |

---

## Running Instructions

### Extract news signals

```bash
python deepseek_signals.py \
  --input ./data/news_train.csv \
  --output ./data/news_with_signals.csv
```

Optional flags:

```
--min-confidence 0.5     # filter low-confidence results (default 0.3)
--checkpoint-interval 20 # checkpoint every 20 rows (default 10)
```

### Train the RNN factor miner

```bash
python seq_run.py
```

### Train the agent ensemble

```bash
python task1_ensemble.py          # CPU
python task1_ensemble.py 0        # GPU 0
```

### Evaluate on held-out data

```bash
python task1_eval.py
```

---

## Testing Instructions

Install test dependencies:

```bash
pip install -e ".[dev]"
```

Run the full test suite:

```bash
pytest tests/ -v
```

Run with coverage report:

```bash
pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=60
```

Run only fast tests (exclude torch-dependent tests if torch is not installed):

```bash
pytest tests/test_config.py tests/test_exceptions.py tests/test_logger.py \
       tests/test_deepseek_signals.py tests/test_deepseek_signals_extended.py \
       -v
```

---

## Project Structure

```
deepseek_signals.py       DeepSeek V3 API signal extractor
seq_net.py                RnnRegNet and NnSeqBnMLP definitions
seq_run.py                Trains the RNN on factor + signal data
erl_agent.py              DQN-family agent implementations
erl_net.py                QNetTwin and QNetTwinDuel network definitions
erl_replay_buffer.py      Off-policy replay buffer
erl_config.py             Config dataclass and build_env helpers
erl_evaluator.py          Training loop evaluator
task1_ensemble.py         Ensemble training orchestration
task1_eval.py             Backtest evaluation on test data
trade_simulator.py        TradeSimulator-v0 environment
config.py                 Pydantic settings (all hyperparameters)
exceptions.py             Custom exception hierarchy (LARSAError)
logger.py                 Structured JSON logging setup
metrics.py                Sharpe, max drawdown, RoMaD helpers
data_config.py            Data path configuration
tests/                    Pytest test suite (21+ test files)
requirements.txt          Pinned dependencies
pyproject.toml            Package metadata, ruff, mypy, pytest config
.github/workflows/ci.yml  CI: lint, type-check, test with coverage
```

---

## Logging

All modules use structured JSON logging via `logger.py`:

```python
from logger import get_logger
log = get_logger(__name__)
log.info("pipeline.start", rows=1024, input="news.csv")
```

Log level is controlled by the `LOG_LEVEL` environment variable (default `INFO`).
Log records are emitted as single-line JSON to stdout, suitable for ingestion by
log aggregators (Datadog, CloudWatch, Splunk, etc.).

---

## Contact

Author: Matthew C. Busel
Email: mattbusel@gmail.com
GitHub: https://github.com/mattbusel
