"""One-command demo: watch the trained DQN ensemble trade 4 hours of real BTC data.

Runs on a fresh clone in about half a minute on a laptop CPU, with no download
and no API key::

    pip install -r requirements-demo.txt
    python demo.py

What it uses (all committed under ``examples/sample/``):

* 17,762 seconds (about 4.9 hours) of the contest's held-out BTC order-book
  data, 2021-04-14 13:49 to 18:45 UTC, cut from the same test split the
  README's full replay uses (seconds 35,400 to 53,162), plus the matching
  rows of the organisers' 8 RNN factors. The first hour only warms up the
  factors, so 4 one-hour episodes are traded.
* ``act.pth`` of the three agents trained on 2026-09-25 (D3QN, DoubleDQN,
  TwinD3QN, 60,000 steps on the first 70% of the data).

It prints three things:

1. One decision taken apart: the 12 numbers the agents see, each agent's
   score for sell / hold / buy, the votes, and what the simulator executes.
2. The same 4 hours replayed with the 3-agent vote, D3QN alone and buy and hold.
3. Where the per-step log was written.

The news channel is neutral (sentiment 3, risk 3) on every row, as in the
README's full run: the contest data ships no headlines.

Research code, not financial advice.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "examples", "sample")
AGENT_NAMES = ("AgentD3QN", "AgentDoubleDQN", "AgentTwinD3QN")
ACTIONS = ("sell", "hold", "buy")
STATE_LABELS = (
    ["position", "holding time"]
    + [f"RNN factor {i}" for i in range(1, 9)]
    + ["news sentiment", "news risk"]
)


def _fail(msg: str) -> None:
    sys.exit(f"demo.py: {msg}")


try:
    import numpy as np
    import torch
except ImportError as exc:  # friendly first run
    _fail(f"missing package {exc.name!r}. Run:  pip install -r requirements-demo.txt")

sys.path.insert(0, HERE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the trained ensemble over 4 bundled hours of BTC order-book data.",
        epilog="examples:\n  python demo.py\n  python demo.py --steps 3 --out runs/demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", default="runs/demo", help="where steps.csv goes (default: runs/demo)")
    parser.add_argument("--steps", type=int, default=1, help="how many split votes to print in part 1 (default: 1)")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(SAMPLE, "BTC_1sec_predict.npy")):
        _fail(f"sample data not found in {SAMPLE}. Clone the full repo (git clone, not a partial download).")
    os.environ["LARSA_DATA_DIR"] = SAMPLE

    from scripts.replay_test import load_agents  # noqa: E402
    from trade_simulator import EvalTradeSimulator  # noqa: E402

    for name in list(logging.root.manager.loggerDict):
        logging.getLogger(name).setLevel(logging.WARNING)

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    np.random.seed(0)
    agents = load_agents([os.path.join(SAMPLE, "agents", n) for n in AGENT_NAMES])
    for _, a in agents:
        a.act.explore_rate = 0.0
    env = EvalTradeSimulator(num_sims=1, slippage=7e-7, max_position=1, step_gap=2)
    ep_len = env.max_step * env.step_gap
    starts = list(range(env.seq_len, env.full_seq_len - ep_len - 1, ep_len))

    def replay(members: list[int]) -> tuple[list[float], list[tuple]]:
        """Run every episode with a majority vote of the given agents."""
        pnl, rows = [], []
        for ep, start in enumerate(starts):
            env.reset(slippage=7e-7)
            env.step_is[:] = start
            env.step_i = 0
            state = env.get_state(env.step_is.cpu())
            total = 0.0
            for k in range(env.max_step):
                qs = [agents[i][1].act(state)[0] for i in members]
                votes = [int(q.argmax()) for q in qs]
                action = Counter(votes).most_common(1)[0][0]
                idx = int(start + (k + 1) * env.step_gap)
                before = state[0].clone()
                state, reward, _done, _ = env.step(torch.tensor([[action]]))
                r = float(reward[0])
                total += r
                pos = int(env.position[0]) if k + 1 < env.max_step else 0
                rows.append((ep, idx, float(env.price_ary[idx, 2]), action - 1, pos, r, votes, qs, before))
            pnl.append(total)
        return pnl, rows

    first_mid = float(env.price_ary[starts[0], 2])
    last_mid = float(env.price_ary[starts[-1] + ep_len, 2])
    print("LARSA demo: 3 DQN agents trading BTC on the contest's 1-second order book")
    print(f"data   {env.full_seq_len:,} seconds from the held-out test split, {len(starts)} episodes x {ep_len} s")
    print(f"price  {first_mid:,.2f} -> {last_mid:,.2f} USD over the traded hours")
    print("agents " + ", ".join(n.split("@")[0] for n, _ in agents) + " (trained 2026-09-25, CPU)\n")

    ens_pnl, ens_rows = replay([0, 1, 2])
    d3_pnl, d3_rows = replay([0])

    # Part 1: one decision taken apart. Pick the first steps where the agents disagree.
    split = [r for r in ens_rows if len(set(r[6])) > 1][: args.steps] or ens_rows[: args.steps]
    print("1) ONE DECISION, TAKEN APART")
    for n, row in enumerate(split):
        ep, idx, mid, action, _pos, _r, votes, qs, st = row
        print(f"\n   episode {ep + 1}, second {idx:,}, BTC mid {mid:,.2f}")
        print("   state the agents see (12 numbers):")
        vals = st.tolist()
        for i in range(0, 12, 4):
            cells = [f"{STATE_LABELS[j]:>15} {vals[j]:+.3f}" for j in range(i, i + 4)]
            print("    " + "  ".join(cells))
        print("   each agent scores sell / hold / buy, then votes for its highest score:")
        for (name, _), q, v in zip(agents, qs, votes):
            scores = "  ".join(f"{ACTIONS[j]} {float(q[j]):+.3f}" for j in range(3))
            print(f"    {name.split('@')[0]:<15} {scores}   -> {ACTIONS[v].upper()}")
        print(f"   majority vote -> {ACTIONS[action + 1].upper()} (simulator then applies position limit, stop-loss, slippage)")
        if n == 0 and len(split) > 1:
            print("   ...")

    # Part 2: the replay.
    hold = []
    for start in starts:
        hold.append(float(env.price_ary[start + ep_len, 2] - env.price_ary[start + env.step_gap, 2]))
    print("\n2) THE SAME HOURS, REPLAYED  (USD P&L, 1 BTC position limit, after slippage)")
    print(f"   {'episode':<9}{'3-agent vote':>14}{'D3QN alone':>13}{'buy and hold':>15}")
    for i in range(len(starts)):
        print(f"   {i + 1:<9}{ens_pnl[i]:>+14.2f}{d3_pnl[i]:>+13.2f}{hold[i]:>+15.2f}")
    print(f"   {'total':<9}{sum(ens_pnl):>+14.2f}{sum(d3_pnl):>+13.2f}{sum(hold):>+15.2f}")
    acts = Counter(r[3] for r in d3_rows)
    in_mkt = sum(1 for r in d3_rows if r[4] != 0) / len(d3_rows)
    agree = sum(1 for r in ens_rows if len(set(r[6])) == 1) / len(ens_rows)
    print(f"\n   3-agent vote: unanimous on {agree:.1%} of steps. DoubleDQN and TwinD3QN learned")
    print("   'always hold', so they outvote D3QN every time and the ensemble never trades.")
    twin = zip(agents[1][1].act.state_dict().values(), agents[2][1].act.state_dict().values())
    if all(torch.equal(p, q) for p, q in twin):
        print("   TwinD3QN is built on the same QNetTwin network as DoubleDQN (erl_agent.py) and,")
        print("   trained with the same seed, has bit-identical weights: one vote counted twice.")
    print(f"   D3QN alone: sell {acts[-1]}, hold {acts[0]}, buy {acts[1]} of {len(d3_rows)} steps, in the market {in_mkt:.1%} of the time.")
    print("   4 hours is an anecdote. The full 68-hour test replay is in the README.")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "steps.csv").replace(os.sep, "/")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("episode,second,mid,action,position,reward_usd,vote_d3qn,vote_doubledqn,vote_twind3qn\n")
        for r in ens_rows:
            fh.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]},{r[4]},{r[5]:.4f},{','.join(str(v - 1) for v in r[6])}\n")
    print(f"\n3) per-step log of the 3-agent run: {path}")
    print("\nResearch code, not financial advice.")


if __name__ == "__main__":
    main()
