# LARSA: LLM-Augmented Regime-Switching Agent

A hybrid reinforcement learning and large language model trading system for
Bitcoin. LARSA extracts structured sentiment and risk signals from BTC news via
the DeepSeek V3 API, mines predictive factors with a recurrent neural network,
and trains an ensemble of DQN-family agents that vote on trade decisions.

---

## Architecture

```
[BTC News] -> [DeepSeek V3 Signals] -> [RNN Factor Mining] ->
[DQN Agent] -> [Ensemble Coordinator] -> [Trade Decisions]
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

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | Yes | (none) | API key for the DeepSeek inference endpoint |
| `DEEPSEEK_BASE_URL` | No | `https://api.deepseek.com/v1` | Endpoint URL |
| `DEEPSEEK_MODEL` | No | `deepseek-chat` | Model identifier |
| `DEEPSEEK_TEMPERATURE` | No | `0.0` | Sampling temperature |
| `NUM_SIMS` | No | `4096` | Parallel simulation environments during training |
| `RL_BREAK_STEP` | No | `32` | Stop training after this many steps |
| `DATA_DIR` | No | `./data` | Directory containing price and news data |
| `OUTPUT_DIR` | No | `./output` | Directory for model checkpoints |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity: DEBUG, INFO, WARNING, ERROR |

All settings are defined in `config.py` and can be set as environment variables
or in a `.env` file in the project root.

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set API keys

```bash
export DEEPSEEK_API_KEY="your-key-here"
```

Or create a `.env` file:

```
DEEPSEEK_API_KEY=your-key-here
```

### 3. Extract news signals

```bash
python deepseek_signals.py \
  --input ./data/news_train.csv \
  --output ./data/news_with_signals.csv
```

### 4. Train the RNN factor miner

```bash
python seq_run.py
```

### 5. Train the agent ensemble

```bash
python task1_ensemble.py          # CPU
python task1_ensemble.py 0        # GPU 0
```

### 6. Evaluate on held-out data

```bash
python task1_eval.py
```

---

## Training Pipeline

**Signal extraction** (`deepseek_signals.py`): Each news article is sent to the
DeepSeek V3 API with a structured prompt requesting a `sentiment_score` (1-5)
and a `risk_score` (1-5). Results are checkpointed every 10 rows so interrupted
runs can resume. Low-confidence outputs are logged and optionally filtered.

**Factor mining** (`seq_net.py`, `seq_run.py`): An `RnnRegNet` model combines
LSTM and GRU layers with MLP projections. It is trained on Alpha101 factors
augmented with the DeepSeek sentiment and risk signals to predict future price
movements. Outputs serve as the state representation for the RL agents.

**Agent training** (`erl_agent.py`, `task1_ensemble.py`): Three agent
architectures are trained independently on `TradeSimulator-v0`:
- `AgentD3QN` (Dueling Double DQN)
- `AgentDoubleDQN` (Double DQN with twin Q-heads)
- `AgentTwinD3QN` (Twin-network D3QN)

Each agent uses an off-policy replay buffer with soft target-network updates.
The `Ensemble` class collects all trained agents and performs majority-vote
action selection at evaluation time.

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
logger.py                 Structured logging setup
metrics.py                Sharpe, Sortino, max drawdown helpers
data_config.py            Data path configuration
tests/                    Pytest test suite
requirements.txt          Pinned dependencies
pyproject.toml            Package metadata, ruff, mypy, pytest config
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Configuration Reference

All settings are managed by `config.py` via Pydantic. Each field maps to an
upper-cased environment variable of the same name (e.g. `DEEPSEEK_API_KEY`).
A `.env` file in the project root is also supported.

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | `""` | API key for the DeepSeek inference endpoint (**required** for signal extraction) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | DeepSeek endpoint URL |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model identifier |
| `DEEPSEEK_TEMPERATURE` | `0.0` | Sampling temperature (0 = deterministic) |
| `DEEPSEEK_MAX_TOKENS` | `300` | Max tokens per completion |
| `MAX_RETRIES` | `5` | Max API retry attempts |
| `MIN_CONFIDENCE_THRESHOLD` | `0.3` | Discard signals below this confidence |
| `CHECKPOINT_INTERVAL` | `10` | Save checkpoint every N rows |
| `RNN_BATCH_SIZE` | `256` | RNN training mini-batch size |
| `RNN_EPOCHS` | `256` | RNN training epochs |
| `RNN_LEARNING_RATE` | `0.001` | RNN AdamW learning rate |
| `RL_LEARNING_RATE` | `2e-6` | RL agent learning rate |
| `RL_BATCH_SIZE` | `512` | RL mini-batch size |
| `RL_GAMMA` | `0.995` | Discount factor for future rewards |
| `RL_BREAK_STEP` | `32` | Stop RL training after this many steps (×1e4) |
| `NUM_SIMS` | `4096` | Parallel simulation environments during training |
| `MAX_POSITION` | `1` | Maximum absolute BTC position |
| `SLIPPAGE` | `7e-7` | Per-trade slippage fraction |
| `STARTING_CASH` | `1000000` | Initial cash for evaluation |
| `DATA_DIR` | `./data` | Directory containing price and news data |
| `OUTPUT_DIR` | `./output` | Directory for checkpoints and artefacts |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

---

## Metrics

The evaluation pipeline reports three financial metrics:

| Metric | Description |
|---|---|
| **Sharpe Ratio** | `(mean_return - risk_free) / std_return` — risk-adjusted return |
| **Max Drawdown** | Largest peak-to-trough decline in cumulative returns |
| **RoMaD** | Return-over-Max-Drawdown (Calmar-style ratio) |

---

## Development

```bash
# Install with dev extras
pip install -e ".[dev]"

# Run linter
ruff check .

# Type-check
mypy config.py exceptions.py metrics.py logger.py deepseek_signals.py

# Run tests with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

---

## License

This project is released under the **MIT License**. See the
[LICENSE](LICENSE) file for details.

---

## Contact

Author: Matthew C. Busel
Email: mattbusel@gmail.com
GitHub: https://github.com/mattbusel
