<picture>
  <source media="(prefers-color-scheme: light)" srcset="assets/hero-light.png">
  <img alt="비트코인 DQN 트레이딩 에이전트: 홀드아웃 테스트 구간의 BTC 중간가, 에이전트가 보는 8개의 RNN 호가창 팩터와 2개의 DeepSeek 뉴스 신호, 그리고 D3QN 에이전트의 매도 및 매수 포지션" src="assets/hero-dark.png" width="100%">
</picture>

# FinRL DeepSeek Crypto Trading (LARSA)

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | 한국어

**2초마다 비트코인을 살지, 팔지, 그대로 들고 있을지 학습한 작은 AI 에이전트 세 개와, 이들이 실제로 어떻게 했는지 정직하게 다시 돌려 본 기록입니다.**

트레이딩용 심층 강화학습을 공부하는 분, FinRL 대회 참가자, 그리고 LLM 뉴스 감성 신호가 트레이딩 에이전트에 어떻게 연결되는지 궁금한 분들을 위한 프로젝트입니다.

LARSA는 FinRL 대회 Task 1을 위한 암호화폐 트레이딩 봇 연구 프로젝트입니다. DQN 앙상블(D3QN, DoubleDQN, TwinD3QN)을 1초 단위 비트코인 지정가 호가창(오더북) 데이터로 학습시켰고, 상태(state)에는 DeepSeek LLM이 매긴 감성 점수와 리스크 점수가 들어갑니다. 에이전트들이 매도, 보유, 매수 중 하나에 투표하면 대회에서 제공하는 시장 리플레이 시뮬레이터가 그 결과를 실행합니다.

&nbsp; **[프로젝트 페이지](https://finrl-deepseek-crypto-trading.vercel.app/)**

> [!WARNING]
> **연구용 코드이며 투자 조언이 아닙니다.** 여기 있는 어떤 것도 실제 돈으로 테스트되지 않았고, 결과는 학습 시드 하나와 2021년 9일치 데이터에서 나온 것입니다. 암호화폐 거래로 투자한 자본을 모두 잃을 수 있습니다.

## 명령어 하나로 실행해 보기

데이터 다운로드도 API 키도 필요 없고, 노트북 CPU에서 약 15초 걸립니다. 저장소에 들어 있는 실제 대회 테스트 데이터 4시간 분량으로 학습된 에이전트를 리플레이합니다.

```bash
git clone https://gitlab.com/mattbusel/FinRL_DeepSeek_Crypto_Trading && cd FinRL_DeepSeek_Crypto_Trading
pip install -r requirements-demo.txt && python demo.py
```

<details open>
<summary>실제 출력(2026-09-28, Python 3.12, CPU)</summary>

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

## 결정 하나가 내려지는 과정

<img alt="실제 결정 하나를 보여 주는 애니메이션 다이어그램: 12개의 숫자(포지션, 보유 시간, 호가창에서 뽑은 8개의 RNN 팩터, DeepSeek 감성과 리스크)가 D3QN, DoubleDQN, TwinD3QN에 들어간다. D3QN은 매도에 가장 높은 점수를, 나머지 둘은 보유에 가장 높은 점수를 준다. 2대 1 다수결 결과는 보유이며, 시뮬레이터가 1 BTC 한도, 손절, 1시간 제한, 슬리피지를 적용한 뒤 보상을 기록한다" src="docs/img/decision.svg" width="100%">

- **상태**(`trade_simulator.py`, `get_state`): 내 포지션과 보유 시간, RNN이 101개의 호가창 알파에서 압축해 낸 8개 팩터, 그리고 최신 DeepSeek 감성 점수와 리스크 점수(1에서 5, 뉴스 헤드라인이 없으면 중립값 3).
- **에이전트**(`erl_net.py`): 각각 3층, 폭 128의 신경망으로 매도, 보유, 매수에 점수를 매깁니다. 각자 가장 높은 점수를 받은 행동에 투표합니다.
- **투표와 시뮬레이터**(`scripts/replay_test.py`, `trade_simulator.py`): 다수결로 정해지면, 시뮬레이터가 포지션을 1 BTC로 제한하고, 최우선 호가 대비 불리하게 움직이면 바로 손절하고, 한 시간이 지나면 청산하고, 슬리피지를 차감한 뒤 가치 변화를 보상으로 기록합니다.
- **데모가 보여 주는 것**: D3QN은 거래하고 싶어 하지만 나머지 둘은 항상 보유하도록 학습했고, TwinD3QN은 DoubleDQN의 완전한 복제본으로 드러났습니다. 그래서 앙상블은 한 번도 거래하지 않습니다. 이것은 이번 실행에 대한 실제 발견이지, 데모의 버그가 아닙니다.

원시 데이터에서 투표까지 이어지는 전체 파이프라인:

<img alt="파이프라인: BTC_1sec.csv에서 Alpha101을 거쳐 RNN 팩터 마이너(8개 팩터)로, 뉴스 CSV에서 DeepSeek V3(감성, 리스크)로. 둘이 합쳐 12개 숫자로 된 상태를 만들고, D3QN, DoubleDQN, TwinD3QN이 투표하고, EvalTradeSimulator가 실행한다" src="assets/pipeline.svg" width="100%">

## 전체 테스트 구간 결과(68시간)

같은 에이전트를 홀드아웃한 데이터 30% 전체에서 돌린 결과로, 2026-09-25에 [아래 3단계](#reproduce-the-full-run-in-3-steps)로 실행했습니다(Python 3.12, torch 2.14 CPU).

<img alt="터미널: prepare_data.py가 학습용 576,577초와 테스트용 247,105초로 나눈다. 3-에이전트 리플레이는 +0.00 USD로 제자리, 매수 후 보유는 -2698.19. D3QN 단독은 +798.15 USD" src="assets/terminal-replay.png" width="100%">

<img alt="68개 테스트 에피소드의 누적 손익: D3QN 단독은 최종 +798 USD, 3-에이전트 투표는 0, 매수 후 보유는 -2,698" src="assets/test-replay.svg" width="100%">

숫자가 말해 주는 것과 말해 주지 않는 것:

| | 손익(USD, 1 BTC 한도 기준) | 에피소드 승 / 패 / 무 | 시장 참여 시간 |
|---|---:|---|---:|
| 3-에이전트 다수결 | 0.00 | 0 / 0 / 68 | 0.0% |
| D3QN 단독 | +798.15 | 25 / 27 / 16 | 2.0% |
| 매수 후 보유, 같은 구간 | -2,698.19 | | 100% |

- **앙상블은 한 번도 거래하지 않았습니다.** 이번 실행에서 DoubleDQN과 TwinD3QN은 "항상 보유"로 수렴했고(학습 내내 평가 수익률이 0.00에 머물렀습니다), 그래서 매 스텝마다 다수결로 D3QN을 이깁니다. 같은 에이전트에 `task1_eval.py`를 돌리면 같은 이유로 샤프 지수 `inf`, 낙폭 `0.0`이 나옵니다. 거래가 없으니 분산도 없습니다.
- **D3QN 단독**은 BTC가 63,734에서 61,371로 떨어진 사흘 동안 대부분 숏 포지션이었습니다(스텝의 2.3%에서 매도, 0.2%에서 매수). 이긴 에피소드와 진 에피소드는 거의 비슷합니다. 시드 하나, 기간 하나, 거래 비용 스트레스 테스트 없음. 엣지가 아니라 돌아가는 파이프라인으로 봐 주세요.
- 리플레이는 대회 자체의 `EvalTradeSimulator`(1 BTC 한도, 7e-7 슬리피지, 손절, 에피소드 종료 시 강제 청산)로 59분짜리 에피소드 68개를 연달아 돌려 테스트 구간 전체를 훑으므로 결과가 결정적입니다. 스텝별 투표, 포지션, 보상은 `runs/replay*/steps.csv`에 있습니다.

<a id="reproduce-the-full-run-in-3-steps"></a>
## 3단계로 전체 실행 재현하기

노트북 CPU에서 돌아갑니다. 시장 데이터는 공개되어 있으며, 대회 주최 측이 Google Drive에 올려 두었습니다(1.4 GB).

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

Windows PowerShell에서는 먼저 `$env:LARSA_DATA_DIR = "data/train"`으로 변수를 설정하세요. 뉴스 파일이 없으면 모든 행에 중립 LLM 신호 3이 들어갑니다([뉴스 신호](docs/REFERENCE.md#news-signals-deepseek) 참고).

## 문서

| 읽을 문서 | 내용 |
|---|---|
| [docs/REFERENCE.md](docs/REFERENCE.md) | 모든 파일과 단계, DeepSeek 뉴스 신호, 전체 설정 변수, 지표, 추가 모듈, 테스트 |
| [docs/architecture.md](docs/architecture.md) | 모듈별 설계 노트 |
| [CHANGELOG.md](CHANGELOG.md) | 무엇이 언제 바뀌었는지 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 테스트 실행 방법과 변경 사항 보내는 법 |

## 라이선스

MIT, [LICENSE](LICENSE) 참고. Open Finance Lab의 [FinRL Contest 2024 Task 1 starter kit](https://github.com/Open-Finance-Lab/FinRL_Contest_2024/tree/main/Task_1_starter_kit)(ElegantRL 에이전트와 시장 리플레이 시뮬레이터)를 바탕으로 만들었습니다.

작성자: Matthew C. Busel([mattbusel@gmail.com](mailto:mattbusel@gmail.com), [gitlab.com/mattbusel](https://gitlab.com/mattbusel))
