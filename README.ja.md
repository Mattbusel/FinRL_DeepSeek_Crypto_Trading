<picture>
  <source media="(prefers-color-scheme: light)" srcset="assets/hero-light.png">
  <img alt="ビットコインの DQN トレーディングエージェント：ホールドアウトしたテスト区間の BTC 仲値、エージェントが見る 8 個の RNN 板情報ファクターと 2 個の DeepSeek ニュースシグナル、D3QN エージェントの売りと買いのポジション" src="assets/hero-dark.png" width="100%">
</picture>

# FinRL DeepSeek Crypto Trading (LARSA)

[English](README.md) | [简体中文](README.zh-CN.md) | 日本語 | [한국어](README.ko.md)

**2 秒ごとにビットコインを買うか、売るか、持つかを学習した 3 つの小さな AI エージェントと、その成績を正直にリプレイした記録です。**

トレーディングのための深層強化学習を学んでいる方、FinRL コンテストの参加者、そして LLM によるニュースのセンチメントシグナルがトレーディングエージェントにどう組み込まれるのか興味のある方に向けたプロジェクトです。

LARSA は FinRL コンテストの Task 1 に向けた暗号資産の自動売買ボットの研究プロジェクトです。DQN のアンサンブル（D3QN、DoubleDQN、TwinD3QN）をビットコインの 1 秒単位の板情報（指値注文板）データで学習させ、状態には DeepSeek LLM によるセンチメントスコアとリスクスコアを含めています。エージェントが売り、ホールド、買いに投票し、コンテストのマーケットリプレイ・シミュレーターがその投票結果を執行します。

&nbsp; **[プロジェクトページ](https://finrl-deepseek-crypto-trading.vercel.app/)**

> [!WARNING]
> **研究用のコードであり、投資助言ではありません。** ここにあるものはどれも実際のお金で試されておらず、結果は 1 つの学習シードと 2021 年の 9 日分のデータによるものです。暗号資産の取引では、投じた資金をすべて失う可能性があります。

## コマンドひとつで試す

データのダウンロードも API キーも不要で、ノートPC の CPU で約 15 秒です。同梱されている実際のコンテスト用テストデータ 4 時間分で、学習済みエージェントをリプレイします。

```bash
git clone https://gitlab.com/mattbusel/FinRL_DeepSeek_Crypto_Trading && cd FinRL_DeepSeek_Crypto_Trading
pip install -r requirements-demo.txt && python demo.py
```

<details open>
<summary>実際の出力（2026-09-28、Python 3.12、CPU）</summary>

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

## 1 回の判断はこうして決まる

<img alt="実際の 1 回の判断を示すアニメーション図：12 個の数値（ポジション、保有時間、板情報から得た 8 個の RNN ファクター、DeepSeek のセンチメントとリスク）が D3QN、DoubleDQN、TwinD3QN に入力される。D3QN は売りに最高スコアを付け、残りの 2 つはホールドに最高スコアを付ける。2 対 1 の多数決でホールドとなり、シミュレーターが 1 BTC の上限、損切り、1 時間の制限、スリッページを適用して報酬を計上する" src="docs/img/decision.svg" width="100%">

- **状態**（`trade_simulator.py`、`get_state`）：自分のポジションと保有時間、101 個の板情報アルファから RNN が圧縮した 8 個のファクター、そして最新の DeepSeek センチメントスコアとリスクスコア（1 から 5、ニュースの見出しがないときは中立の 3）。
- **エージェント**（`erl_net.py`）：それぞれ 3 層、幅 128 のネットワークで、売り、ホールド、買いにスコアを付けます。各エージェントは最もスコアの高い行動に投票します。
- **投票とシミュレーター**（`scripts/replay_test.py`、`trade_simulator.py`）：多数決で決まったあと、シミュレーターがポジションを 1 BTC に制限し、最良価格に対して逆行したら即座に損切りし、1 時間で手仕舞いし、スリッページを差し引いて、評価額の変化を報酬として計上します。
- **デモから分かること**：D3QN は取引したがっていますが、残りの 2 つは常にホールドすることを学習しており、さらに TwinD3QN は DoubleDQN の完全なコピーだと判明しました。そのためアンサンブルは一度も取引しません。これはこの実行についての実際の発見であり、デモのバグではありません。

生データから投票までのパイプライン全体：

<img alt="パイプライン：BTC_1sec.csv から Alpha101、RNN ファクターマイナー（8 ファクター）へ。ニュース CSV から DeepSeek V3（センチメント、リスク）へ。両者が 12 個の数値からなる状態を作り、D3QN、DoubleDQN、TwinD3QN が投票し、EvalTradeSimulator が執行する" src="assets/pipeline.svg" width="100%">

## テスト区間全体での結果（68 時間）

同じエージェントを、ホールドアウトしたデータの 30% 全体で動かしました。2026-09-25 に[下の 3 ステップ](#reproduce-the-full-run-in-3-steps)で実行しています（Python 3.12、torch 2.14 CPU）。

<img alt="ターミナル：prepare_data.py が学習用 576,577 秒とテスト用 247,105 秒に分割。3 エージェントのリプレイは +0.00 USD で横ばい、バイ・アンド・ホールドは -2698.19。D3QN 単独では +798.15 USD" src="assets/terminal-replay.png" width="100%">

<img alt="68 個のテストエピソードにおける累積損益：D3QN 単独は最終的に +798 USD、3 エージェントの投票は 0、バイ・アンド・ホールドは -2,698" src="assets/test-replay.svg" width="100%">

数字が示していること、示していないこと：

| | 損益（USD、1 BTC 上限あたり） | エピソード 勝ち / 負け / 横ばい | 市場にいた時間 |
|---|---:|---|---:|
| 3 エージェントの多数決 | 0.00 | 0 / 0 / 68 | 0.0% |
| D3QN 単独 | +798.15 | 25 / 27 / 16 | 2.0% |
| バイ・アンド・ホールド、同じ期間 | -2,698.19 | | 100% |

- **アンサンブルは一度も取引しませんでした**。この実行では DoubleDQN と TwinD3QN が「常にホールド」に収束しており（学習中ずっと評価リターンが 0.00 のままでした）、すべてのステップで D3QN を多数決で上回ります。同じエージェントに対して `task1_eval.py` を実行すると、同じ理由でシャープレシオ `inf`、ドローダウン `0.0` と報告されます。取引がなければ分散もありません。
- **D3QN 単独**では、BTC が 63,734 から 61,371 に下落した 3 日間、ほとんどショートしていました（売りがステップの 2.3%、買いが 0.2%）。勝ちエピソードと負けエピソードはほぼ同数です。シード 1 つ、期間 1 つ、取引コストのストレステストなし。エッジではなく、動くパイプラインとして見てください。
- リプレイはコンテスト付属の `EvalTradeSimulator`（1 BTC 上限、7e-7 のスリッページ、損切り、エピソード終了時に強制的にポジションをクローズ）を使い、59 分のエピソードを 68 個連続させてテスト区間全体をたどるので、結果は決定的です。ステップごとの投票、ポジション、報酬は `runs/replay*/steps.csv` にあります。

<a id="reproduce-the-full-run-in-3-steps"></a>
## 3 ステップでフル実行を再現する

ノートPC の CPU で動きます。マーケットデータは公開されており、コンテストの主催者が Google Drive で配布しています（1.4 GB）。

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

Windows PowerShell では、先に `$env:LARSA_DATA_DIR = "data/train"` で変数を設定してください。ニュースファイルがない場合、すべての行に中立の LLM シグナル 3 が入ります（[ニュースシグナル](docs/REFERENCE.md#news-signals-deepseek)を参照）。

## ドキュメント

| 読むもの | 内容 |
|---|---|
| [docs/REFERENCE.md](docs/REFERENCE.md) | すべてのファイルと各ステージ、DeepSeek ニュースシグナル、すべての設定変数、評価指標、追加モジュール、テスト |
| [docs/architecture.md](docs/architecture.md) | モジュールごとの設計メモ |
| [CHANGELOG.md](CHANGELOG.md) | いつ何が変わったか |
| [CONTRIBUTING.md](CONTRIBUTING.md) | テストの実行方法と変更の送り方 |

## ライセンス

MIT。[LICENSE](LICENSE) を参照してください。Open Finance Lab の [FinRL Contest 2024 Task 1 starter kit](https://github.com/Open-Finance-Lab/FinRL_Contest_2024/tree/main/Task_1_starter_kit)（ElegantRL のエージェントとマーケットリプレイ・シミュレーター）をベースにしています。

作者：Matthew C. Busel（[mattbusel@gmail.com](mailto:mattbusel@gmail.com)、[gitlab.com/mattbusel](https://gitlab.com/mattbusel)）
