# Reference

Everything that used to live in the README below the fold. Back to the [README](../README.md).

- [Pipeline stages and files](#pipeline-stages-and-files)
- [News signals (DeepSeek)](#news-signals-deepseek)
- [Configuration](#configuration)
- [Metrics](#metrics)
- [Extra modules](#extra-modules)
- [Tests](#tests)

## Pipeline stages and files

| Stage | File | What it does |
|---|---|---|
| Data | `scripts/prepare_data.py` | Downloads the contest data, merges optional news signals, writes a time-ordered 70/30 split to `data/train` and `data/test` |
| News signals | `deepseek_signals.py` | Two JSON prompts per article (sentiment 1-5, risk 1-5, each with a confidence and one-line reasoning); retries, checkpoints, drops low-confidence rows |
| Factors | `seq_data.py`, `seq_run.py`, `seq_net.py` | 101 formulaic alphas on the order book, then an MLP + 4-layer LSTM + 4-layer GRU that outputs 8 factors. `data/BTC_1sec_predict.pth` holds trained weights; the quick start uses the organisers' precomputed factors |
| Environment | `trade_simulator.py` | Vectorised market replay from the contest starter kit; state = position, holding time, 8 factors, 2 LLM signals |
| Agents | `erl_agent.py`, `erl_net.py`, `erl_replay_buffer.py` | ElegantRL DoubleDQN, D3QN (dueling) and TwinD3QN. Note: `AgentTwinD3QN` uses the same `QNetTwin` network as `AgentDoubleDQN`, so with the same seed the two train to identical weights (checked on the bundled 2026-09-25 agents) |
| Training | `task1_ensemble.py` | Trains each agent class and saves it under `ensemble_teamname/ensemble_models/` |
| Evaluation | `task1_eval.py`, `scripts/replay_test.py`, `metrics.py` | Contest-style single episode with Sharpe, max drawdown and RoMaD; full-split replay with per-step logs |

`TradeSimulator-v0_D3QN_0/` is the D3QN checkpoint from the original submission. It still loads (`scripts/replay_test.py --agents TradeSimulator-v0_D3QN_0`); on the test split used in the README it also holds on every step.

## News signals (DeepSeek)

The contest data ships no news, and the organisers note that the timestamps in `BTC_1sec.csv` were processed and are not the true times. To use the LLM channel:

```bash
export DEEPSEEK_API_KEY=...
python deepseek_signals.py --input data/news_train.csv --output data/news_with_signals.csv
python scripts/prepare_data.py --signals data/news_with_signals.csv
```

The news CSV needs `title`, `text` and a time column (`date`, `timestamp`, `datetime`, `published_at` or `system_time`). Each second takes the latest scored article before it; seconds before the first article stay neutral at 3. Because the data timestamps are synthetic, any alignment with real headlines is approximate, which is why the README run leaves the channel neutral rather than guess.

## Configuration

Environment variables or `.env`, validated by `config.py`.

| Variable | Default | Description |
|---|---|---|
| `LARSA_DATA_DIR` | `./data` | Data directory the simulator reads (`data/train`, `data/test` after `prepare_data.py`) |
| `LARSA_ENSEMBLE_DIR` | `ensemble_teamname/ensemble_models` | Where `task1_eval.py` loads agents from |
| `DEEPSEEK_API_KEY` | (required for signals) | API key for the DeepSeek endpoint |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | Endpoint URL |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model identifier |
| `DEEPSEEK_TEMPERATURE` | `0.0` | Sampling temperature |
| `DEEPSEEK_MAX_TOKENS` | `300` | Max tokens per completion |
| `MAX_RETRIES` | `5` | API retry attempts before giving up |
| `CHECKPOINT_INTERVAL` | `10` | Save progress every N rows |
| `MIN_CONFIDENCE_THRESHOLD` | `0.3` | Discard signals below this confidence |
| `NUM_SIMS` | `4096` | Parallel simulations during training (256 is plenty on CPU) |
| `RL_BREAK_STEP` | `32` | Stop training after this many steps (32 means one rollout; the README run used 60000) |
| `RL_GAMMA` | `0.995` | Discount factor |
| `RL_LEARNING_RATE` | `2e-6` | Learning rate |
| `RL_BATCH_SIZE` | `512` | Mini-batch size |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

Settings for the extra modules (`MULTI_ASSET_*`, `PAPER_TRADING_*`) are documented in `config.py`.

## Metrics

`metrics.py`, numpy only.

All take per-step fractional returns.

| Function | Definition |
|---|---|
| `sharpe_ratio` | `(mean - risk_free) / std(ddof=1)`, not annualised; `inf` when returns do not vary |
| `max_drawdown` | `min((V_t - peak_t) / peak_t)` on the wealth path, a non-positive fraction |
| `return_over_max_drawdown` | cumulative return / abs(max drawdown); `inf` with no drawdown |
| `cumulative_returns` | `cumprod(1 + r) - 1` as a pandas Series |

```python
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown
returns = [0.001, -0.002, 0.003]
print(sharpe_ratio(returns), max_drawdown(returns), return_over_max_drawdown(returns))
```

## Extra modules

Outside the contest pipeline.

These were added after the contest. They have their own tests but are not used by `task1_ensemble.py` or the replay, and none of them has been run against live markets here.

| Module | Purpose |
|---|---|
| `multi_agent_debate.py` | Bull, bear and neutral DeepSeek agents debate a trade before an arbiter decides |
| `alt_data.py`, `onchain_signals.py` | Fear and greed index, blockchain.info on-chain stats, social and macro feeds as a feature vector |
| `continuous_learner.py` | Replay buffer, online fine-tuning and drift detection for an agent in a live loop |
| `portfolio_opt.py`, `risk_parity.py`, `multi_asset.py` | Mean-variance, hierarchical risk parity, min-correlation and multi-asset allocation |
| `live_trading_bridge.py`, `paper_trading.py`, `paper_trader.py` | Dry-run by default bridge to exchange APIs, Binance-feed paper trading, Alpaca paper trading |
| `explainability.py` | Gradient or KernelSHAP attribution per decision, HTML and JSON audit reports |
| `hyperparameter_search.py`, `walk_forward.py`, `regime_detector.py` | Grid and random search, rolling walk-forward validation, HMM regime detection |
| `src/` | Standalone utilities: paper trader, risk limits, ensemble voter, report generator, live WebSocket feeds, rebalancer, BHB attribution, toy environments and strategies |

## Tests

```bash
pip install pytest pytest-mock pytest-asyncio
python -m pytest -q tests/test_trade_simulator.py tests/test_task1_eval.py tests/test_metrics.py tests/test_erl_agent.py tests/test_prepare_data.py
```

CI byte-compiles every module and runs the contest-pipeline tests on each push. A handful of tests for the extra modules fail or need optional packages (`shap`, `hmmlearn`); they are not part of CI.
