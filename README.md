# LARSA: LLM-Augmented Regime-Switching Agent

[![CI](https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading/actions/workflows/ci.yml/badge.svg)](https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://docs.astral.sh/ruff/)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy.readthedocs.io/)

A hybrid reinforcement learning and large language model trading system for Bitcoin - and now, for entire crypto portfolios. LARSA extracts structured sentiment and risk signals from BTC news via the DeepSeek V3 API, mines predictive factors with a recurrent neural network, trains an ensemble of DQN-family agents that vote on trade decisions, and supports live paper trading on Binance price feeds.

---

## Features

- **DeepSeek V3 signal extraction** -- structured sentiment + risk JSON from raw news articles
- **RNN factor miner** -- `RnnRegNet` trained on Alpha101 + LLM signals outputs predictive feature vectors
- **DQN ensemble** -- `AgentD3QN`, `AgentDoubleDQN`, `AgentTwinD3QN` trained independently and combined via majority vote
- **Regime-aware weighting** -- `RegimeDetector` shifts agent weights based on detected market regime
- **Multi-asset portfolio** -- `MultiAssetEnvironment` handles N crypto assets simultaneously with rolling correlation features
- **Paper trading mode** -- `PaperTrader` connects to Binance WebSocket, executes simulated trades, logs to SQLite, and prints a live terminal dashboard
- **Comprehensive backtesting** -- Sharpe, max drawdown, RoMaD, annual return on held-out data

---

## Architecture

### Full Pipeline

```
  BTC News CSV          DeepSeek V3                 RNN Factor Miner
  (raw articles)  --->  sentiment_score   -------->  RnnRegNet trained on
                        risk_score                   Alpha101 + LLM signals
                                                          |
                                          +---------------+
                                          v
                               TradeSimulator-v0          <--- BTC price ticks
                               (vectorised env,                (1-sec resolution)
                                4096 parallel sims)
                                          |
                     +--------------------+--------------------+
                     v                   v                     v
               AgentD3QN          AgentDoubleDQN         AgentTwinD3QN
               (Dueling +         (Double DQN)           (Twin-network
                Double DQN)                               Dueling D3QN)
                     +--------------------+--------------------+
                                          v
                              Ensemble Coordinator
                              majority-vote + regime weighting
                                          v
                              Evaluation / Paper Trading
                              Sharpe, drawdown, RoMaD, P&L
```

### Multi-Asset Extension

```
  BTC prices  \
  ETH prices   +--> MultiAssetEnvironment
  SOL prices  /       - PortfolioState (prices, positions, cash, total_value)
                      - CorrelationMatrix (rolling Pearson, upper-tri features)
                      - PortfolioAction (allocation weights, rebalance logic)
                              |
               +--------------+--------------+
               v              v              v
          BTC Agent      ETH Agent      SOL Agent
          (D3QN)         (D3QN)         (D3QN)
               +--------------+--------------+
                              v
                   MultiAssetEnsemble
                   meta-agent optimises allocation
                   using per-asset Q-values + correlation
```

### Paper Trading Pipeline

```
  Binance WebSocket  --->  PaperTrader
  (live prices)              |
                             | ensemble vote (buy / sell / hold)
                             v
                         PaperPortfolio
                         - positions, cash, P&L
                         - SQLite TradeLog
                         - max drawdown, win rate
                             |
                             v
                      Terminal Dashboard
                      compare_vs_backtest()
```

### Module Flow

```
deepseek_signals.py         DeepSeek V3 API; extracts sentiment_score + risk_score
        |
seq_net.py / seq_run.py     RnnRegNet trained on Alpha101 + news signals
        |
erl_agent.py                AgentD3QN / AgentDoubleDQN / AgentTwinD3QN
        |
task1_ensemble.py           Majority-vote ensemble; regime-aware signal weighting
        |
task1_eval.py               Backtest on held-out data (Sharpe, drawdown, RoMaD)
        |
multi_asset.py              Multi-asset portfolio environment + ensemble meta-agent
        |
paper_trading.py            Live paper trading via Binance WS + SQLite trade log
```

---

## System Requirements

- Python 3.10, 3.11, or 3.12
- pip >= 23
- CUDA-capable GPU (optional; all components run on CPU)
- Internet access for DeepSeek API calls during signal extraction
- ~4 GB disk space for data, model checkpoints, and dependencies

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

All settings are read from environment variables (or a `.env` file) and validated at startup via `config.py`. Defaults are shown below.

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
| `MULTI_ASSET_SYMBOLS` | No | `["BTC","ETH","SOL"]` | Assets for portfolio mode |
| `MULTI_ASSET_STARTING_CASH` | No | `100000.0` | Starting cash for portfolio simulations |
| `MULTI_ASSET_CORRELATION_WINDOW` | No | `120` | Rolling window for correlation matrix |
| `MULTI_ASSET_REBALANCE_THRESHOLD` | No | `0.02` | Min weight drift to trigger rebalance |
| `PAPER_TRADING_ASSETS` | No | `["BTCUSDT"]` | Binance symbols for paper trading |
| `PAPER_TRADING_CASH` | No | `100000.0` | Initial simulated cash |
| `PAPER_TRADING_QTY_FRACTION` | No | `0.05` | Fraction of cash deployed per buy |
| `PAPER_TRADING_DB_PATH` | No | `./paper_trades.db` | SQLite path for trade log |
| `PAPER_TRADING_WS_URL` | No | `wss://stream.binance.com:9443/ws` | Binance WebSocket URL |

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

### Train the agent ensemble (single-asset BTC)

```bash
python task1_ensemble.py          # CPU
python task1_ensemble.py 0        # GPU 0
```

### Evaluate on held-out data

```bash
python task1_eval.py
```

### Multi-asset portfolio example

```python
from multi_asset import MultiAssetEnvironment, MultiAssetEnsemble
import numpy as np

# Create environment (uses synthetic prices if no price_data provided)
env = MultiAssetEnvironment(
    assets=["BTC", "ETH", "SOL"],
    starting_cash=100_000.0,
    max_step=500,
)
state = env.reset()
print(f"State dim: {env.state_dim}")  # per-asset features + portfolio + correlation

for _ in range(500):
    # Random allocation weights (softmax-normalised inside env.step)
    action = np.random.randn(env.n_assets).astype(np.float32)
    state, reward, done, info = env.step(action)
    if done:
        state = env.reset()

pstate = env.get_portfolio_state()
print(f"Final value: ${pstate.total_value:,.2f}")
print(f"Weights:     {pstate.weights()}")
```

### Paper trading -- offline mock mode

```python
import numpy as np
from paper_trading import PaperPortfolio, PaperTrader, print_dashboard

portfolio = PaperPortfolio(starting_cash=100_000.0)

# Generate mock BTC prices (replace with real data or live WebSocket)
prices = (30_000.0 * np.exp(np.cumsum(np.random.normal(0.0001, 0.002, 300)))).tolist()

trader = PaperTrader(
    portfolio=portfolio,
    assets=["BTCUSDT"],
    mock_prices={"BTCUSDT": prices},
    trade_qty_fraction=0.05,
)
trader.run(max_ticks=300)

# Show terminal dashboard
print_dashboard(portfolio, current_prices={"BTCUSDT": prices[-1]})
```

Dashboard output example:

```
------------------------------------------------------------------------
  LARSA Paper Trading Dashboard - 2026-03-22 10:15:43 UTC
------------------------------------------------------------------------
  Total Value   :    $101,423.88
  Cash          :     $92,100.00
  Total P&L     : +1.424%  ($+1,423.88)
  Realised P&L  :       $+820.50
  Max Drawdown  : 0.312%
  Win Rate      : 58.3%  (24 trades)
------------------------------------------------------------------------
  POSITIONS
    BTCUSDT       qty=0.030000  price=$30,800.00  val=$924.00  unreal=+24.00
------------------------------------------------------------------------
  RECENT TRADES (last 5)
    [10:14:11] SELL  BTCUSDT       qty=0.015000  @$30,750.00  pnl=+11.25
    [10:12:33] BUY   BTCUSDT       qty=0.030000  @$30,600.00  pnl=+0.00
    ...
------------------------------------------------------------------------
```

### Paper trading -- live Binance WebSocket

```python
import asyncio
from paper_trading import PaperPortfolio, PaperTrader

portfolio = PaperPortfolio(starting_cash=100_000.0)
trader = PaperTrader(
    portfolio=portfolio,
    assets=["BTCUSDT", "ETHUSDT"],
    agents=my_trained_agents,   # list of LARSA ensemble agents
)
asyncio.run(trader.run_async(max_ticks=1000))
```

### Compare paper trading vs backtest

```python
from paper_trading import compare_vs_backtest

result = compare_vs_backtest(
    paper_portfolio=portfolio,
    backtest_returns=backtest_step_returns,  # list[float]
    current_prices={"BTCUSDT": 31_500.0},
)
# {
#   "paper": {"total_pnl_pct": 2.3, "win_rate": 0.54, ...},
#   "backtest": {"total_return_pct": 1.8, "sharpe": 1.2, ...},
#   "delta_return_pct": 0.5
# }
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
trade_simulator.py        TradeSimulator-v0 environment (single-asset BTC)
multi_asset.py            MultiAssetEnvironment + CorrelationMatrix + MultiAssetEnsemble
paper_trading.py          PaperTrader, PaperPortfolio, TradeLog, dashboard, compare
regime_detector.py        Market regime detection (bull/bear/sideways/volatile)
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

## Results

Run `python task1_eval.py` to populate these values after training.

| Metric | Single-Asset BTC | Multi-Asset (BTC/ETH/SOL) |
|--------|-----------------|--------------------------|
| Sharpe Ratio | See training output | See training output |
| Max Drawdown | See training output | See training output |
| Annual Return | See training output | See training output |

Paper trading results are logged to SQLite and displayed via `print_dashboard()`.

---

## Contact

Author: Matthew C. Busel
Email: mattbusel@gmail.com
GitHub: https://github.com/mattbusel
