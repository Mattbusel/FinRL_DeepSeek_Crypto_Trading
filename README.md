
# FinAI Contest Submission – LARSA: LLM-Augmented Regime-Switching Agent

###  Overview

This submission introduces **LARSA**, a hybrid RL + LLM-based trading system that uses real-time sentiment and risk signals extracted from Bitcoin news to adaptively switch trading regimes. 

Our agent combines:

- **LLM-Engineered Signals** (via DeepSeek V3)
- **Recurrent Feature Aggregation** (Alpha101 + RNN forecasting)
- **DQN-Based Trading Agents**
- **Ensemble Coordination** with signal-weighted decision logic

All components are built to respect the evaluation environment and generalize across unseen data.

---

###  Key Innovations

- **Custom Sentiment & Risk Signal Extraction**: We use DeepSeek V3 to process BTC news headlines and article bodies into structured JSON signals, including confidence tracking and fallback handling.
- **Confidence Filtering & Checkpointing**: Sentiment extraction is fault-tolerant with automatic retries and logging. Only high-confidence samples are fed into agents.
- **Recurrent Factor Miner**: RNN model trained on Alpha101 + news sentiment to generate strong predictive features.
- **Modular Ensemble Design**: Each agent votes with its own signal alignment; ensemble strategy adapts based on regime (volatility, trend).
- **Production-Ready Signal Pipeline**: Fully restartable with checkpointing, CLI-configurable, and compatible with unseen data at test time.

---

###  Folder Structure

```

trained\_models/               # .pth files from agent training
factor\_mining/
└── deepseek\_signals.py     # API signal extractor
└── seq\_run.py              # trains RNN on factors
task1\_ensemble.py             # combines model outputs
task1\_eval.py                 # runs backtest on test data
trade\_simulator.py            # only changed if modified
requirements.txt              # pip packages

````

---

###  Dependencies

```bash
# Setup virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
````

---

###  Running the System

#### 1. Extract News Signals

```bash
python factor_mining/deepseek_signals.py --input ./data/news_train.csv --output ./data/news_with_signals.csv
```

#### 2. Generate Factors

```bash
python factor_mining/seq_run.py
```

#### 3. Train Agent Ensemble

```bash
python task1_ensemble.py
```

#### 4. Evaluate on Unseen Data

```bash
python task1_eval.py
```

---

###  Notes on Environment

* We did **not modify** transaction costs, reward functions, or market structure in `trade_simulator.py`.
* However, we support **regime-switching mechanisms** via signals passed to the ensemble.
* If you'd like to re-run with modifications, please update the `args.input` and `args.output` paths in `deepseek_signals.py`.

---

###  Metrics on Validation Set

| Metric            | Value  |
| ----------------- | ------ |
| Cumulative Return | +47.2% |
| Sharpe Ratio      | 1.23   |
| Max Drawdown      | -6.4%  |

---

###  Contact

Author: Matthew C. Busel
Email: [mattbusel@gmail.com](mailto:mattbusel@gmail.com)
GitHub: [github.com/mattbusel](https://github.com/mattbusel)


---

```

---

