<picture>
  <source media="(prefers-color-scheme: light)" srcset="assets/hero-light.png">
  <img alt="比特币 DQN 交易智能体：留出测试集上的 BTC 中间价、智能体看到的 8 个 RNN 订单簿因子和 2 个 DeepSeek 新闻信号，以及 D3QN 智能体的卖出和买入仓位" src="assets/hero-dark.png" width="100%">
</picture>

# FinRL DeepSeek Crypto Trading (LARSA)

[English](README.md) | 简体中文 | [日本語](README.ja.md) | [한국어](README.ko.md)

**三个小型 AI 智能体，学会了每 2 秒对比特币做一次买入、卖出或持有决策，外加一份如实回放它们表现的记录。**

适合正在学习深度强化学习量化交易的人、FinRL 竞赛参赛者，以及想知道大模型（LLM）新闻情绪信号如何接入交易智能体的人。

LARSA 是一个面向 FinRL 竞赛 Task 1 的加密货币交易机器人研究项目：一个 DQN 集成模型（D3QN、DoubleDQN、TwinD3QN），在比特币逐秒限价订单簿数据上训练，状态中包含 DeepSeek 大模型给出的情绪分和风险分。智能体投票选择卖出、持有或买入，再由竞赛提供的市场回放模拟器执行投票结果。

&nbsp; **[项目主页](https://finrl-deepseek-crypto-trading.vercel.app/)**

> [!WARNING]
> **这是研究代码，不构成投资建议。** 这里的任何东西都没有用真金白银测试过，结果来自一个训练随机种子、2021 年九天的数据。加密货币交易可能让你投入的本金全部亏光。

## 一条命令试用

无需下载数据，无需 API 密钥，在笔记本 CPU 上大约 15 秒。它会在仓库自带的 4 小时真实竞赛测试数据上回放训练好的智能体：

```bash
git clone https://gitlab.com/mattbusel/FinRL_DeepSeek_Crypto_Trading && cd FinRL_DeepSeek_Crypto_Trading
pip install -r requirements-demo.txt && python demo.py
```

<details open>
<summary>真实输出（2026-09-28，Python 3.12，CPU）</summary>

```text
LARSA demo: 3 DQN agents trading BTC on the contest's 1-second order book
data   17,762 seconds from the held-out test split, 4 episodes x 3540 s
price  63,313.23 -> 62,300.00 USD over the traded hours
agents AgentD3QN, AgentDoubleDQN, AgentTwinD3QN (trained 2026-09-25, CPU)

1) ONE DECISION, TAKEN APART

   episode 1, second 3,642, BTC mid 63,321.23
   state the agents see (12 numbers):
           position +0.000     holding time +0.000     RNN factor 1 -0.941     RNN factor 2 -0.914
       RNN factor 3 -0.726     RNN factor 4 -0.577     RNN factor 5 -0.712     RNN factor 6 -0.824
       RNN factor 7 -0.862     RNN factor 8 -0.814   news sentiment +3.000        news risk +3.000
   each agent scores sell / hold / buy, then votes for its highest score:
    AgentD3QN       sell +0.222  hold +0.214  buy +0.208   -> SELL
    AgentDoubleDQN  sell -0.349  hold -0.345  buy -0.353   -> HOLD
    AgentTwinD3QN   sell -0.349  hold -0.345  buy -0.353   -> HOLD
   majority vote -> HOLD (simulator then applies position limit, stop-loss, slippage)

2) THE SAME HOURS, REPLAYED  (USD P&L, 1 BTC position limit, after slippage)
   episode    3-agent vote   D3QN alone   buy and hold
   1                 +0.00      +308.41        -531.06
   2                 +0.00      -169.21        +743.33
   3                 +0.00        +8.62       -1000.34
   4                 +0.00      -328.78        -242.11
   total             +0.00      -180.95       -1030.18

   3-agent vote: unanimous on 92.9% of steps. DoubleDQN and TwinD3QN learned
   'always hold', so they outvote D3QN every time and the ensemble never trades.
   TwinD3QN is built on the same QNetTwin network as DoubleDQN (erl_agent.py) and,
   trained with the same seed, has bit-identical weights: one vote counted twice.
   D3QN alone: sell 619, hold 6389, buy 72 of 7080 steps, in the market 7.4% of the time.
   4 hours is an anecdote. The full 68-hour test replay is in the README.

3) per-step log of the 3-agent run: runs/demo/steps.csv

Research code, not financial advice.
```
</details>

## 一次决策是怎样做出的

<img alt="一次真实决策的动画示意图：12 个数字（仓位、持仓时间、从订单簿提取的 8 个 RNN 因子、DeepSeek 情绪和风险）输入 D3QN、DoubleDQN 和 TwinD3QN；D3QN 给卖出打分最高，另外两个给持有打分最高；2 比 1 的多数投票结果为持有；模拟器施加 1 BTC 上限、止损、1 小时限制和滑点，然后记入奖励" src="docs/img/decision.svg" width="100%">

- **状态**（`trade_simulator.py`，`get_state`）：你的仓位和持仓时间、RNN 从 101 个订单簿 alpha 因子中压缩出的 8 个因子，以及最新的 DeepSeek 情绪分和风险分（1 到 5，没有新闻标题时取中性值 3）。
- **智能体**（`erl_net.py`）：每个都是 3 层、宽度 128 的神经网络，分别给卖出、持有和买入打分，然后投票给自己打分最高的动作。
- **投票与模拟器**（`scripts/replay_test.py`，`trade_simulator.py`）：多数票胜出，随后模拟器把仓位上限设为 1 BTC，价格相对最优价出现任何不利波动就止损，持仓满一小时平仓，扣除滑点，并把价值变化记为奖励。
- **演示揭示了什么**：D3QN 想交易，另外两个学会了永远持有，而 TwinD3QN 原来是 DoubleDQN 的完全复制品，所以集成模型从不交易。这是关于这次运行的真实发现，不是演示程序的 bug。

从原始数据到一次投票的完整流水线：

<img alt="流水线：BTC_1sec.csv 经 Alpha101 进入 RNN 因子挖掘器（8 个因子）；新闻 CSV 进入 DeepSeek V3（情绪、风险）；两者组成 12 个数字的状态；D3QN、DoubleDQN 和 TwinD3QN 投票；EvalTradeSimulator 执行" src="assets/pipeline.svg" width="100%">

## 完整测试集上的结果（68 小时）

同样的智能体在全部留出的 30% 数据上运行，于 2026-09-25 按[下面的 3 个步骤](#reproduce-the-full-run-in-3-steps)完成（Python 3.12，torch 2.14 CPU）：

<img alt="终端：prepare_data.py 划分出 576,577 秒训练数据和 247,105 秒测试数据；3 智能体回放保持持平，+0.00 USD，对比买入持有的 -2698.19；D3QN 单独运行赚了 +798.15 USD" src="assets/terminal-replay.png" width="100%">

<img alt="68 个测试回合的累计盈亏：D3QN 单独运行最终为 +798 USD，3 智能体投票为 0，买入持有为 -2,698" src="assets/test-replay.svg" width="100%">

这些数字说明了什么，又没有说明什么：

| | 盈亏，USD（每 1 BTC 上限） | 回合 盈 / 亏 / 平 | 持仓时间占比 |
|---|---:|---|---:|
| 3 智能体多数投票 | 0.00 | 0 / 0 / 68 | 0.0% |
| D3QN 单独运行 | +798.15 | 25 / 27 / 16 | 2.0% |
| 买入持有，相同时间窗口 | -2,698.19 | | 100% |

- **集成模型从未交易**。在这次运行中，DoubleDQN 和 TwinD3QN 收敛到了“永远持有”（它们的评估收益在整个训练过程中一直是 0.00），所以每一步都以多数票压过 D3QN。对同样的智能体运行 `task1_eval.py`，出于同样的原因会报告夏普比率 `inf`、回撤 `0.0`：没有交易，就没有方差。
- **D3QN 单独运行**时大部分时间做空（2.3% 的步数卖出，0.2% 买入），这三天里 BTC 从 63,734 跌到了 61,371。盈利回合和亏损回合几乎持平。一个随机种子、一个时间段、没有交易成本压力测试：请把它当作一条能跑通的流水线，而不是一个优势（edge）。
- 回放使用竞赛自带的 `EvalTradeSimulator`（1 BTC 上限、7e-7 滑点、止损、回合结束时强制平仓），以 68 个首尾相接、每个 59 分钟的回合走完整个测试集，因此结果是确定性的。每一步的投票、仓位和奖励都在 `runs/replay*/steps.csv` 中。

<a id="reproduce-the-full-run-in-3-steps"></a>
## 3 步复现完整运行

在笔记本 CPU 上即可运行。行情数据是公开的：竞赛主办方把它托管在 Google Drive 上（1.4 GB）。

```bash
git clone https://gitlab.com/mattbusel/FinRL_DeepSeek_Crypto_Trading
cd FinRL_DeepSeek_Crypto_Trading
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scipy matplotlib pydantic pydantic-settings pyyaml openai gdown

# 1. download BTC_1sec.csv + BTC_1sec_predict.npy, split 70/30 in time
python scripts/prepare_data.py --download --source data/raw

# 2. train D3QN, DoubleDQN and TwinD3QN on the first 70%  (about 30 min on CPU)
LARSA_DATA_DIR=data/train NUM_SIMS=256 RL_BREAK_STEP=60000 python task1_ensemble.py

# 3. replay the ensemble over the last 30%, hour by hour
LARSA_DATA_DIR=data/test python scripts/replay_test.py
```

在 Windows PowerShell 中，请先用 `$env:LARSA_DATA_DIR = "data/train"` 设置变量。如果没有新闻文件，每一行都会得到中性的大模型信号 3（见[新闻信号](docs/REFERENCE.md#news-signals-deepseek)）。

## 文档

| 阅读 | 内容 |
|---|---|
| [docs/REFERENCE.md](docs/REFERENCE.md) | 每个文件和每个阶段、DeepSeek 新闻信号、全部配置变量、评估指标、附加模块、测试 |
| [docs/architecture.md](docs/architecture.md) | 逐个模块的设计说明 |
| [CHANGELOG.md](CHANGELOG.md) | 改了什么、什么时候改的 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 如何运行测试并提交改动 |

## 许可证

MIT，见 [LICENSE](LICENSE)。基于 Open Finance Lab 的 [FinRL Contest 2024 Task 1 starter kit](https://github.com/Open-Finance-Lab/FinRL_Contest_2024/tree/main/Task_1_starter_kit)（ElegantRL 智能体和市场回放模拟器）构建。

作者：Matthew C. Busel（[mattbusel@gmail.com](mailto:mattbusel@gmail.com)，[gitlab.com/mattbusel](https://gitlab.com/mattbusel)）
