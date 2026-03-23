"""Options trading environment for reinforcement learning."""
import math
from typing import List, Tuple, Optional

def bsm_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """Black-Scholes-Merton option pricing."""
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    if option_type == 'call':
        return S * norm_cdf(d1) - K * math.exp(-r*T) * norm_cdf(d2)
    return K * math.exp(-r*T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def bsm_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    if T <= 0:
        return 1.0 if (option_type == 'call' and S > K) else 0.0
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    if option_type == 'call':
        return norm_cdf(d1)
    return norm_cdf(d1) - 1

class OptionsEnv:
    """Delta hedging environment for options trading RL."""

    def __init__(self, S0: float = 100.0, K: float = 100.0, T: float = 0.25,
                 r: float = 0.05, sigma: float = 0.2, n_steps: int = 63,
                 transaction_cost: float = 0.001, option_type: str = 'call'):
        self.S0 = S0
        self.K = K
        self.T_total = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.transaction_cost = transaction_cost
        self.option_type = option_type
        self.reset()

    def reset(self) -> List[float]:
        self.S = self.S0
        self.t = 0
        self.portfolio_delta = 0.0
        self.cash = 0.0
        self.done = False
        # Write option, collect premium
        T_rem = self.T_total
        self.option_value = bsm_price(self.S, self.K, T_rem, self.r, self.sigma, self.option_type)
        self.cash = self.option_value
        return self._state()

    def _state(self) -> List[float]:
        T_rem = max(self.T_total - self.t * self.dt, 0)
        delta = bsm_delta(self.S, self.K, T_rem, self.r, self.sigma, self.option_type)
        moneyness = self.S / self.K
        return [self.S / self.S0, T_rem / self.T_total, delta, moneyness, self.portfolio_delta, self.sigma]

    def step(self, action: float, seed: int = 0) -> Tuple[List[float], float, bool]:
        """Action: target delta hedge ratio (-1 to 1)."""
        # Simulate stock price (GBM)
        import random
        rng = random.Random(seed + self.t)
        z = rng.gauss(0, 1)
        self.S *= math.exp((self.r - 0.5*self.sigma**2)*self.dt + self.sigma*math.sqrt(self.dt)*z)
        self.S = max(self.S, 0.01)

        # Execute hedge
        target_shares = max(-1.0, min(1.0, action))
        delta_shares = target_shares - self.portfolio_delta
        trade_cost = abs(delta_shares) * self.S * self.transaction_cost
        self.cash -= delta_shares * self.S + trade_cost
        self.portfolio_delta = target_shares

        self.t += 1
        T_rem = max(self.T_total - self.t * self.dt, 0)

        if self.t >= self.n_steps:
            self.done = True
            # Settle option
            payoff = max(self.S - self.K, 0) if self.option_type == 'call' else max(self.K - self.S, 0)
            # Unwind hedge
            self.cash += self.portfolio_delta * self.S * (1 - self.transaction_cost)
            pnl = self.cash - payoff
            reward = pnl / self.option_value if self.option_value > 0 else pnl
        else:
            new_option_val = bsm_price(self.S, self.K, T_rem, self.r, self.sigma, self.option_type)
            delta_option = new_option_val - self.option_value
            pnl = self.portfolio_delta * (self.S - self.S0) - delta_option - trade_cost
            reward = pnl / self.option_value if self.option_value > 0 else pnl
            self.option_value = new_option_val

        return self._state(), reward, self.done

    def state_dim(self) -> int:
        return 6

    def action_dim(self) -> int:
        return 1
