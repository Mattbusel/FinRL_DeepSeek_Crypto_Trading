# FinRL DeepSeek Crypto Trading (LARSA)

English | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

AI agents that trade Bitcoin and read the news.

68 test hours: D3QN made +$798. Buy and hold lost $2,698.

- Trains 3 DQN agents on 1-second BTC data
- Scores headlines with DeepSeek V3
- Votes sell, hold or buy
- Replays it in the FinRL contest simulator

Research code. Not financial advice.

## Install

```bash
git clone https://gitlab.com/mattbusel/FinRL_DeepSeek_Crypto_Trading && cd FinRL_DeepSeek_Crypto_Trading
pip install -r requirements-demo.txt
```

## Use

```bash
python demo.py
```

No API key. About 15 seconds on a laptop CPU.

[Project page](https://finrl-deepseek-crypto-trading.vercel.app/)

More: [docs/DETAILS.md](docs/DETAILS.md)

License: MIT. Built on the [FinRL Contest 2024 Task 1 starter kit](https://github.com/Open-Finance-Lab/FinRL_Contest_2024/tree/main/Task_1_starter_kit).
