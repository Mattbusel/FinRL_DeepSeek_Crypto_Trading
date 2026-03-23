# LARSA: LLM-Augmented Regime-Switching Agent

[![CI](https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading/actions/workflows/ci.yml/badge.svg)](https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://docs.astral.sh/ruff/)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy.readthedocs.io/)

---

> **DISCLAIMER: LARSA is experimental research software. It is NOT financial advice. Crypto trading carries substantial risk of loss. Never deploy real capital without fully understanding the system, the risks, and the applicable laws in your jurisdiction. Always start with dry-run or paper trading mode. Past simulated performance does not guarantee future results.**

---

## What is LARSA?

LARSA (LLM-Augmented Regime-Switching Agent) is a hybrid reinforcement learning and large language model trading system for crypto assets. It extracts structured sentiment and risk signals from BTC news via the DeepSeek V3 API, mines predictive factors with a recurrent neural network trained on Alpha101 features, trains an ensemble of three DQN-family agents (D3QN, DoubleDQN, TwinD3QN) that vote on trade decisions, and dynamically shifts ensemble weights based on a detected market regime (bull, bear, sideways, volatile). The system supports single-asset BTC trading, multi-asset crypto portfolios, live paper trading on Binance price feeds, production-safe live trading via the Live Trading Bridge, per-trade explainability reports, and automated hyperparameter search.

---

## Architecture Diagram

```
                         +------------------------------------------------------+
                         |                   LARSA Pipeline                     |
                         +------------------------------------------------------+

  BTC/Crypto News CSV
  (raw headlines)
        |
        v
  +------------------+      structured JSON        +--------------------------+
  | deepseek_        |  sentiment_score (1-5)  --> |  RNN Factor Miner        |
  | signals.py       |  risk_score      (1-5)      |  (RnnRegNet)             |
  | DeepSeek V3 API  |  confidence      (0-1)      |  trained on Alpha101     |
  +------------------+                             |  + LLM signals           |
                                                   +-----------+--------------+
                                                               |  factor vectors
                                                               v
  BTC price ticks --------------------------------------------> TradeSimulator-v0
  (1-sec resolution,                                           vectorised env
   4096 parallel sims)                                         (num_sims=4096)
                                                               |
                             +----------------------------------+
                             |               |                 |
                             v               v                 v
                       AgentD3QN      AgentDoubleDQN     AgentTwinD3QN
                       (Dueling +     (Double DQN)       (Twin-network
                        Double DQN)                       Dueling D3QN)
                             |               |                 |
                             +---------------+-----------------+
                                             |  majority vote
                                             v
                                  Ensemble Coordinator
                                  + RegimeDetector
                                  (regime-aware weights)
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
                    v                        v                        v
             Backtest / Eval         Live Trading Bridge       Explainability
             (Sharpe, RoMaD,         (LiveTradingBridge)       Dashboard
              Sortino, MDD)          risk-gated execution      HTML + JSON audit
                                     dry-run / testnet /
                                     live exchange APIs

  ---------------------------------------------------------------------------

  Multi-Asset Extension:

  BTC prices -+
  ETH prices -+-> MultiAssetEnvironment --> per-asset D3QN agents
  SOL prices -+    PortfolioState               |
                   CorrelationMatrix      MultiAssetEnsemble
                   rebalance logic        meta-agent + Q-values

  ---------------------------------------------------------------------------

  Paper Trading Pipeline:

  Binance WebSocket --> PaperTrader --> PaperPortfolio --> Terminal Dashboard
  (live prices)         ensemble vote    SQLite TradeLog    compare_vs_backtest
```

### Module Flow

```
deepseek_signals.py       DeepSeek V3 API; extracts sentiment_score + risk_score
        |
seq_net.py / seq_run.py   RnnRegNet trained on Alpha101 + news signals
        |
erl_agent.py              AgentD3QN / AgentDoubleDQN / AgentTwinD3QN
        |
task1_ensemble.py         Majority-vote ensemble; regime-aware signal weighting
        |
task1_eval.py             Backtest on held-out data (Sharpe, drawdown, RoMaD)
        |
multi_asset.py            Multi-asset portfolio environment + ensemble meta-agent
        |
paper_trading.py          Paper trading via Binance WS + SQLite trade log
        |
live_trading_bridge.py    Production-safe exchange bridge (dry-run / testnet / live)
        |
explainability.py         SHAP attribution + per-trade HTML/JSON reports
        |
hyperparameter_search.py  Grid and random search over key hyperparameters
```

---

## 5-Minute Quickstart

### 1. Clone and install

```bash
git clone https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading.git
cd FinRL_DeepSeek_Crypto_Trading

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export DEEPSEEK_API_KEY="your-key-here"
# Or create a .env file:
echo "DEEPSEEK_API_KEY=your-key-here" > .env
```

### 3. Run the full pipeline

```bash
# Extract news signals
python deepseek_signals.py --input ./data/news_train.csv --output ./data/news_with_signals.csv

# Train the RNN factor miner
python seq_run.py

# Train the agent ensemble
python task1_ensemble.py          # CPU
python task1_ensemble.py 0        # GPU 0

# Evaluate on held-out data
python task1_eval.py
```

### 4. Paper trade (offline mock, no API key needed)

```python
import numpy as np
from paper_trading import PaperPortfolio, PaperTrader, print_dashboard

portfolio = PaperPortfolio(starting_cash=100_000.0)
prices = (30_000.0 * np.exp(np.cumsum(np.random.normal(0.0001, 0.002, 300)))).tolist()

trader = PaperTrader(
    portfolio=portfolio,
    assets=["BTCUSDT"],
    mock_prices={"BTCUSDT": prices},
    trade_qty_fraction=0.05,
)
trader.run(max_ticks=300)
print_dashboard(portfolio, current_prices={"BTCUSDT": prices[-1]})
```

---

## Training Guide

### Step 1: Signal extraction

```bash
python deepseek_signals.py \
  --input ./data/news_train.csv \
  --output ./data/news_with_signals.csv \
  --min-confidence 0.5 \
  --checkpoint-interval 20
```

Produces a CSV with `sentiment_score`, `risk_score`, and `confidence_score_*` columns appended to each news row.

### Step 2: RNN factor miner

```bash
python seq_run.py
```

Trains `RnnRegNet` on Alpha101 factors plus the LLM-derived signals. Saves factor prediction arrays as `.npy` files in `./output/`.

Key hyperparameters (set via environment variables or `.env`):

| Variable | Default | Effect |
|---|---|---|
| `RNN_EPOCHS` | 256 | Training epochs |
| `RNN_BATCH_SIZE` | 256 | Mini-batch size |
| `RNN_LEARNING_RATE` | 1e-3 | AdamW learning rate |
| `RNN_NUM_LAYERS` | 4 | Stacked recurrent layers |
| `RNN_MID_DIM` | 128 | Hidden layer width |

### Step 3: Ensemble training

```bash
python task1_ensemble.py          # all three agents sequentially
python task1_ensemble.py 0        # all three agents on GPU 0
```

Each of the three DQN-family agents trains independently on `TradeSimulator-v0` using the factor vectors from Step 2. Model checkpoints are saved to `./output/` or `./TradeSimulator-v0_*/`.

Key hyperparameters:

| Variable | Default | Effect |
|---|---|---|
| `RL_LEARNING_RATE` | 2e-6 | Actor/critic learning rate |
| `RL_BATCH_SIZE` | 512 | RL mini-batch size |
| `RL_GAMMA` | 0.995 | Discount factor |
| `NUM_SIMS` | 4096 | Parallel simulation envs |
| `RL_BREAK_STEP` | 32 | Training steps before stopping |

### Step 4: Evaluation

```bash
python task1_eval.py
```

Runs the trained ensemble on the held-out test split and prints Sharpe ratio, max drawdown, RoMaD, and annualised return.

---

## Live Trading Safety Guide

LARSA includes `live_trading_bridge.py` — a production-safe bridge from ensemble signals to exchange APIs.

### Safety architecture

```
TradeSignal (from ensemble)
        |
        v
  Risk Gate (LiveTradingBridge._passes_risk_checks)
  +-- confidence >= min_confidence (default 0.65)
  +-- cooldown: >= min_trade_interval_secs since last trade (default 300s)
  +-- open positions < max_open_positions (default 3)
  +-- position_size_pct <= max_position_pct (default 10%)
  +-- daily loss < max_daily_loss_pct (default 5%) -- else HALT
        |
        v (passes)
  [dry_run=True]  --> Log signal, return DRY-RUN result (NO order sent)
  [dry_run=False] --> Exchange order (MockClient / BinanceTestnet / live)
        |
        v
  OrderResult + JSONL audit log (live_trades.jsonl)
        |
        v
  Background stop-loss loop (every 30s checks open positions)
```

### Start in dry-run mode (always recommended first)

```python
import asyncio
from live_trading_bridge import LiveTradingBridge, MockExchangeClient, RiskLimits, TradeSignal
from datetime import datetime

exchange = MockExchangeClient(initial_balance=10_000.0)
limits = RiskLimits(dry_run=True)   # DRY RUN -- no real orders
queue: asyncio.Queue = asyncio.Queue()
bridge = LiveTradingBridge(exchange=exchange, risk_limits=limits, signal_queue=queue)

signal = TradeSignal(
    timestamp=datetime.utcnow(),
    symbol="BTC/USDT",
    direction="buy",
    confidence=0.82,
    regime="bull",
    position_size_pct=0.05,
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    source_model="AgentD3QN",
)
asyncio.run(bridge.process_signal(signal))
print(bridge.generate_daily_report())
```

### Paper trading with MockExchangeClient

`MockExchangeClient` simulates realistic fills including random slippage and taker fees. It never touches a real exchange.

```python
from live_trading_bridge import MockExchangeClient, RiskLimits

exchange = MockExchangeClient(
    initial_balance=10_000.0,
    fee_pct=0.001,        # 0.1% taker fee
    slippage_pct=0.0005,  # up to 0.05% random slippage
)
limits = RiskLimits(dry_run=False)  # mock exchange, orders execute locally
```

### Binance testnet (safe, no real money)

```python
from live_trading_bridge import BinanceTestnetClient, RiskLimits

# Get testnet keys from https://testnet.binance.vision/
exchange = BinanceTestnetClient(api_key="YOUR_TESTNET_KEY", secret_key="YOUR_TESTNET_SECRET")
limits = RiskLimits(
    dry_run=False,
    max_position_pct=0.05,
    max_daily_loss_pct=0.02,
    min_confidence=0.70,
    min_trade_interval_secs=600,
)
```

### RiskLimits reference

| Parameter | Default | Description |
|---|---|---|
| `dry_run` | `True` | Log signals but never execute orders |
| `max_position_pct` | 0.10 | Max 10% of capital per position |
| `max_daily_loss_pct` | 0.05 | Halt trading if daily loss exceeds 5% |
| `min_confidence` | 0.65 | Reject signals below this confidence |
| `min_trade_interval_secs` | 300 | 5-minute cooldown between trades |
| `max_open_positions` | 3 | Max simultaneous open positions |

Every signal — accepted or rejected — is appended to `live_trades.jsonl` as a single JSON line providing a full audit trail.

---

## Explainability Dashboard Guide

`explainability.py` provides two complementary tools:

### 1. LarsaExplainer -- per-signal HTML + JSON audit

```python
from explainability import LarsaExplainer

explainer = LarsaExplainer()

explanation = explainer.explain_signal(
    signal_data={
        "symbol": "BTC/USDT",
        "direction": "buy",
        "confidence": 0.82,
        "regime": "bull",
        "regime_confidence": 0.78,
        "sentiment_score": 4.2,
        "current_drawdown_pct": -0.8,
        "days_since_last_trade": 2,
    },
    news_items=[
        "Bitcoin ETF receives SEC approval",
        "Institutional inflows reach record high",
        "Fed signals pause on rate hikes",
    ],
    factor_values={
        "alpha_001": 0.45,
        "alpha_012": -0.12,
        "alpha_028": 0.31,
        "sentiment_score": 0.84,
    },
    agent_outputs=[
        {"agent_type": "D3QN",     "agent_id": 0, "action": "buy",  "q_value": 1.23},
        {"agent_type": "DDQN",     "agent_id": 1, "action": "buy",  "q_value": 0.98},
        {"agent_type": "TwinD3QN", "agent_id": 2, "action": "hold", "q_value": 0.55},
    ],
)

# Write HTML report
html = explainer.generate_html_report(explanation)
with open("trade_report.html", "w") as f:
    f.write(html)

# Append JSON audit entry
audit = explainer.generate_json_audit(explanation)
with open("audit.jsonl", "a") as f:
    f.write(audit + "\n")
```

The HTML report is self-contained (no external CSS/JS) and includes:
- Decision banner (BUY / SELL / HOLD) with confidence and consensus
- Key metrics: sentiment score, regime, drawdown, days since last trade
- DeepSeek sentiment section: headlines, market-moving keywords
- Alpha101 factor attribution table with visual contribution bars
- Ensemble vote breakdown with Q-values and per-agent reasoning
- Similar historical signal patterns and their outcomes

### 2. TradeExplainer -- SHAP attribution for DRL agents

```python
from explainability import TradeExplainer

explainer = TradeExplainer(
    agent=my_trained_agent,
    feature_names=["position", "btc_price", "sentiment_score", ...],
    top_k=5,
    symbol="BTC/USDT",
)
report = explainer.explain_decision(state=current_state, action=1)
print(report.to_text())
# Traded BUY BTC/USDT because:
#   sentiment_score           +0.3412  (bullish)
#   alpha_001                 +0.2198  (bullish)
#   volatility                -0.1503  (bearish)
```

---

## Hyperparameter Search Guide

`hyperparameter_search.py` provides grid and random search over LARSA hyperparameters.

### Define an evaluation function

```python
from hyperparameter_search import HyperparamSearch

def evaluate(params: dict) -> dict:
    """Train and evaluate a LARSA config; return metrics."""
    sharpe = run_backtest(params)["sharpe"]
    return {"sharpe": sharpe, "max_drawdown": -0.12, "total_return": 0.08}

searcher = HyperparamSearch(eval_fn=evaluate, n_jobs=1)
```

### Random search (recommended)

```python
results = searcher.random_search(n_trials=20)
best = searcher.best_config(metric="sharpe")
print(f"Best config: {best.params}")
print(f"Best Sharpe: {best.sharpe:.4f}")
print(searcher.summary_table())
searcher.save_results("my_search.json")
```

### Grid search

```python
results = searcher.grid_search(param_grid={
    "learning_rate": [1e-4, 3e-4],
    "batch_size": [64, 128],
    "gamma": [0.97, 0.99],
})
```

### Default search space

| Parameter | Candidates |
|---|---|
| `learning_rate` | 1e-4, 3e-4, 1e-3 |
| `batch_size` | 32, 64, 128 |
| `gamma` | 0.95, 0.97, 0.99 |
| `hidden_sizes` | [256,256], [512,256], [256,128,64] |
| `confidence_threshold` | 0.5, 0.6, 0.7 |
| `regime_switch_threshold` | 0.3, 0.4, 0.5 |

### Load and compare saved results

```python
searcher2 = HyperparamSearch(eval_fn=evaluate)
searcher2.load_results("my_search.json")
print(searcher2.top_k(k=3, metric="sharpe"))
```

### Parallel search

```python
searcher = HyperparamSearch(eval_fn=evaluate, n_jobs=4)
# Evaluates 4 configs simultaneously via ThreadPoolExecutor
```

---

## Performance Metrics Explained

All metrics are computed in `metrics.py` using the `empyrical` library.

### Sharpe Ratio

Measures risk-adjusted return:

```
Sharpe = (mean_return - risk_free_rate) / std_return
```

A Sharpe > 1.0 is generally considered good for discretionary strategies. Higher is better.

### Maximum Drawdown (MDD)

The largest peak-to-trough decline in portfolio value over the evaluation period:

```
MDD = min over all t of (V_t - peak_t) / peak_t
```

Reported as a non-positive fraction (e.g. -0.15 = -15%). Closer to 0 is better.

### Return over Max Drawdown (RoMaD)

Also known as the Calmar ratio:

```
RoMaD = cumulative_return / |max_drawdown|
```

Returns `inf` if no drawdown occurred. Higher is better. RoMaD > 2.0 indicates the system recovered its worst loss more than twice over.

### Sortino Ratio

Similar to Sharpe but penalises only downside volatility:

```
Sortino = (mean_return - risk_free_rate) / downside_std
```

More appropriate for asymmetric return distributions typical in crypto.

### Usage

```python
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown

returns = [0.001, -0.002, 0.003, ...]  # per-step percentage returns

sr   = sharpe_ratio(returns)
mdd  = max_drawdown(returns)
romd = return_over_max_drawdown(returns)

print(f"Sharpe: {sr:.3f}  MDD: {mdd:.3f}  RoMaD: {romd:.3f}")
```

---

## Configuration Reference

All settings are read from environment variables (or a `.env` file) and validated at startup via `config.py`.

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | (required) | API key for the DeepSeek inference endpoint |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | Endpoint URL |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model identifier |
| `DEEPSEEK_TEMPERATURE` | `0.0` | Sampling temperature |
| `DEEPSEEK_MAX_TOKENS` | `300` | Max tokens per completion |
| `MAX_RETRIES` | `5` | API retry attempts before giving up |
| `CHECKPOINT_INTERVAL` | `10` | Save progress every N rows |
| `MIN_CONFIDENCE_THRESHOLD` | `0.3` | Discard signals below this confidence |
| `NUM_SIMS` | `4096` | Parallel simulation environments during training |
| `RL_BREAK_STEP` | `32` | Stop RL training after this many steps |
| `RL_GAMMA` | `0.995` | Discount factor for future rewards |
| `RL_LEARNING_RATE` | `2e-6` | Actor/critic learning rate |
| `RL_BATCH_SIZE` | `512` | Mini-batch size for RL updates |
| `DATA_DIR` | `./data` | Directory containing price and news data |
| `OUTPUT_DIR` | `./output` | Directory for model checkpoints |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MULTI_ASSET_SYMBOLS` | `["BTC","ETH","SOL"]` | Assets for portfolio mode |
| `MULTI_ASSET_STARTING_CASH` | `100000.0` | Starting cash for portfolio simulations |
| `PAPER_TRADING_ASSETS` | `["BTCUSDT"]` | Binance symbols for paper trading |
| `PAPER_TRADING_CASH` | `100000.0` | Initial simulated cash |
| `PAPER_TRADING_QTY_FRACTION` | `0.05` | Fraction of cash deployed per buy |
| `PAPER_TRADING_DB_PATH` | `./paper_trades.db` | SQLite path for trade log |

---

## System Requirements

- Python 3.10, 3.11, or 3.12
- pip >= 23
- CUDA-capable GPU (optional; all components run on CPU)
- Internet access for DeepSeek API calls during signal extraction
- ~4 GB disk space for data, model checkpoints, and dependencies
- `httpx` (optional, required only for `BinanceTestnetClient`): `pip install httpx`

---

## Installation

```bash
git clone https://github.com/mattbusel/FinRL_DeepSeek_Crypto_Trading.git
cd FinRL_DeepSeek_Crypto_Trading

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# Optional: dev tools (pytest, mypy, ruff, black, isort)
pip install -e ".[dev]"
```

---

## Testing Instructions

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run the full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=60

# Run only fast tests (no torch required)
pytest tests/test_config.py tests/test_exceptions.py tests/test_logger.py \
       tests/test_deepseek_signals.py tests/test_deepseek_signals_extended.py \
       -v
```

---

## Project Structure

```
deepseek_signals.py         DeepSeek V3 API signal extractor
seq_net.py                  RnnRegNet and NnSeqBnMLP definitions
seq_run.py                  Trains the RNN on factor + signal data
erl_agent.py                DQN-family agent implementations
erl_net.py                  QNetTwin and QNetTwinDuel network definitions
erl_replay_buffer.py        Off-policy replay buffer
erl_config.py               Config dataclass and build_env helpers
erl_evaluator.py            Training loop evaluator
task1_ensemble.py           Ensemble training orchestration
task1_eval.py               Backtest evaluation on test data
trade_simulator.py          TradeSimulator-v0 environment (single-asset BTC)
multi_asset.py              MultiAssetEnvironment + CorrelationMatrix + MultiAssetEnsemble
paper_trading.py            PaperTrader, PaperPortfolio, TradeLog, dashboard, compare
regime_detector.py          Market regime detection (bull/bear/sideways/volatile)
live_trading_bridge.py      Production-safe exchange bridge with full risk gates
explainability.py           SHAP attribution + per-trade HTML/JSON reports (LarsaExplainer)
hyperparameter_search.py    Grid/random search over LARSA hyperparameters
config.py                   Pydantic settings (all hyperparameters)
exceptions.py               Custom exception hierarchy (LARSAError)
logger.py                   Structured JSON logging setup
metrics.py                  Sharpe, max drawdown, RoMaD helpers
data_config.py              Data path configuration
tests/                      Pytest test suite (21+ test files)
requirements.txt            Pinned dependencies
pyproject.toml              Package metadata, ruff, mypy, pytest config
.github/workflows/ci.yml    CI: lint, type-check, test with coverage
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
Records are emitted as single-line JSON to stdout, suitable for log aggregators
(Datadog, CloudWatch, Splunk, etc.).

---

## Results

Run `python task1_eval.py` to populate these values after training.

| Metric | Single-Asset BTC | Multi-Asset (BTC/ETH/SOL) |
|---|---|---|
| Sharpe Ratio | See training output | See training output |
| Max Drawdown | See training output | See training output |
| Annual Return | See training output | See training output |
| RoMaD | See training output | See training output |

Paper trading results are logged to SQLite and displayed via `print_dashboard()`.

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository and create a feature branch from `main`.
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Write tests for any new functionality in `tests/`.
4. Ensure all checks pass before submitting a pull request:

```bash
ruff check .
mypy --ignore-missing-imports .
pytest tests/ --cov=. --cov-fail-under=60
```

5. Open a pull request with a clear description of what the change does and why.

Do not commit API keys, model weights, or large data files.

---

## Disclaimer

LARSA is experimental research software developed for educational and research purposes. It is NOT financial advice and should NOT be used to make real investment decisions without thorough independent evaluation. Cryptocurrency markets are highly volatile and largely unregulated. You can lose all of your invested capital. The authors make no warranty of any kind regarding performance, fitness for purpose, or accuracy. By using this software you agree that the authors are not liable for any financial losses incurred.

**Always start with `dry_run=True` (the default) and paper trading before considering any live deployment.**

---

## Contact

Author: Matthew C. Busel
Email: mattbusel@gmail.com
GitHub: https://github.com/mattbusel
