"""Replay a trained ensemble over the whole held-out split, hour by hour.

``task1_eval.py`` scores one 59-minute episode at a random offset.  This
script walks the test split end to end instead: back-to-back episodes of the
same length, the same ``EvalTradeSimulator`` (stop-loss, slippage, forced
flat at episode end) and the same majority vote, so every number it prints is
reproducible from a seed.

For every step it records the mid price, each agent's vote, the ensemble
action, the resulting position, the reward (USD P&L for a 1 BTC position
limit) and the two LLM signal inputs.  It writes::

    runs/replay/steps.csv      one row per environment step
    runs/replay/summary.json   totals, per-episode P&L, vote agreement

Usage::

    LARSA_DATA_DIR=data/test python scripts/replay_test.py
    LARSA_DATA_DIR=data/test python scripts/replay_test.py --agents TradeSimulator-v0_D3QN_0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN  # noqa: E402
from erl_config import Config  # noqa: E402
from trade_simulator import EvalTradeSimulator  # noqa: E402

AGENT_CLASSES = {"AgentD3QN": AgentD3QN, "AgentDoubleDQN": AgentDoubleDQN, "AgentTwinD3QN": AgentTwinD3QN}


def load_agents(paths: list[str]) -> list[tuple[str, object]]:
    """Load one agent per checkpoint directory; the class comes from the dir name."""
    cfg = Config()
    agents = []
    for path in paths:
        base = os.path.basename(os.path.normpath(path))
        name = next((n for n in AGENT_CLASSES if n == base), None)
        if name is None:  # e.g. TradeSimulator-v0_D3QN_0
            name = "AgentTwinD3QN" if "TwinD3QN" in base else "AgentD3QN" if "D3QN" in base else "AgentDoubleDQN"
        agent = AGENT_CLASSES[name]((128, 128, 128), 12, 3, gpu_id=-1, args=cfg)
        agent.save_or_load_agent(path, if_save=False)
        agent.act.eval()
        agents.append((f"{name}@{base}", agent))
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--agents", nargs="+", default=None,
                        help="checkpoint dirs (default: every dir under ensemble_teamname/ensemble_models)")
    parser.add_argument("--out", default="runs/replay")
    parser.add_argument("--slippage", type=float, default=7e-7)
    args = parser.parse_args()

    if args.agents is None:
        root = "ensemble_teamname/ensemble_models"
        if not os.path.isdir(root):
            sys.exit(f"no {root}/ yet: run task1_ensemble.py first, or pass --agents DIR")
        args.agents = [os.path.join(root, d) for d in sorted(os.listdir(root))]

    torch.set_grad_enabled(False)
    agents = load_agents(args.agents)
    env = EvalTradeSimulator(num_sims=1, slippage=args.slippage, max_position=1, step_gap=2)
    ep_len = env.max_step * env.step_gap  # seconds per episode
    first = env.seq_len
    starts = list(range(first, env.full_seq_len - ep_len - 1, ep_len))
    print(f"| {len(agents)} agents  {len(starts)} episodes x {ep_len}s  on {env.full_seq_len:,} seconds of test data")

    rows = []
    episode_pnl = []
    agree = Counter()
    for ep, start in enumerate(starts):
        env.reset(slippage=args.slippage)
        env.step_is[:] = start
        env.step_i = 0
        state = env.get_state(env.step_is.cpu())
        pnl = 0.0
        for k in range(env.max_step):
            votes = [int(a.act(state).argmax(dim=1)[0]) for _, a in agents]
            action = Counter(votes).most_common(1)[0][0]
            agree[len(set(votes))] += 1
            idx = int(start + (k + 1) * env.step_gap)
            state, reward, _done, _ = env.step(torch.tensor([[action]]))
            r = float(reward[0])
            pnl += r
            rows.append((ep, idx, float(env.price_ary[idx, 2]), action - 1, int(env.position[0]) if k + 1 < env.max_step else 0,
                         r, *votes, *env.llm_signals[idx].tolist()))
        episode_pnl.append(pnl)

    os.makedirs(args.out, exist_ok=True)
    names = [n for n, _ in agents]
    header = ["episode", "second", "mid", "action", "position", "reward_usd"] + [f"vote_{i}" for i in range(len(agents))] + ["sentiment_score", "risk_score"]
    with open(os.path.join(args.out, "steps.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + "\n")

    pnl = np.array(episode_pnl)
    actions = np.array([r[3] for r in rows])
    positions = np.array([r[4] for r in rows])
    mids = np.array([r[2] for r in rows])
    first_mid = [mids[i * env.max_step] for i in range(len(starts))]
    last_mid = [mids[(i + 1) * env.max_step - 1] for i in range(len(starts))]
    hold_pnl = np.array(last_mid) - np.array(first_mid)
    summary = {
        "agents": names,
        "episodes": len(starts),
        "episode_seconds": ep_len,
        "steps": len(rows),
        "total_pnl_usd_per_btc": float(pnl.sum()),
        "mean_episode_pnl_usd": float(pnl.mean()),
        "winning_episodes": int((pnl > 0).sum()),
        "losing_episodes": int((pnl < 0).sum()),
        "flat_episodes": int((pnl == 0).sum()),
        "buy_and_hold_pnl_usd_per_btc": float(hold_pnl.sum()),
        "action_share": {k: float((actions == v).mean()) for k, v in (("sell", -1), ("hold", 0), ("buy", 1))},
        "time_in_market": float((positions != 0).mean()),
        "vote_unanimous_share": float(agree[1] / sum(agree.values())),
        "llm_signal_unique_values": sorted({(r[-2], r[-1]) for r in rows}),
        "episode_pnl_usd": [round(float(x), 2) for x in pnl],
        "buy_and_hold_episode_pnl_usd": [round(float(x), 2) for x in hold_pnl],
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"| ensemble P&L  {summary['total_pnl_usd_per_btc']:>+10.2f} USD  (1 BTC limit, after slippage)")
    print(f"| buy and hold  {summary['buy_and_hold_pnl_usd_per_btc']:>+10.2f} USD  (same windows)")
    print(f"| episodes won {summary['winning_episodes']}  lost {summary['losing_episodes']}  flat {summary['flat_episodes']}")
    print(f"| in market {summary['time_in_market']:.1%}   unanimous votes {summary['vote_unanimous_share']:.1%}")
    print(f"| wrote {args.out}/steps.csv and summary.json")


if __name__ == "__main__":
    main()
