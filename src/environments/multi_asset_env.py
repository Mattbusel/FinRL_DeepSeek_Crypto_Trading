"""Multi-asset portfolio RL environment."""
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


@dataclass
class MultiAssetConfig:
    n_assets: int = 5
    initial_cash: float = 100_000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    max_position_pct: float = 0.4    # max 40% in one asset
    episode_length: int = 252
    lookback: int = 20
    gbm_mu: List[float] = field(default_factory=lambda: [0.08, 0.12, 0.06, 0.15, 0.10])
    gbm_sigma: List[float] = field(default_factory=lambda: [0.15, 0.25, 0.12, 0.30, 0.20])
    gbm_corr: float = 0.3  # pairwise correlation


class MultiAssetEnv:
    """RL environment: allocate across N assets + cash."""

    def __init__(self, config: Optional[MultiAssetConfig] = None, seed: int = 42):
        self.config = config or MultiAssetConfig()
        self.rng = random.Random(seed)
        self.reset()

    @property
    def state_dim(self) -> int:
        return self.config.n_assets * self.config.lookback + self.config.n_assets + 1

    @property
    def action_dim(self) -> int:
        return self.config.n_assets + 1  # allocation to each asset + cash

    def reset(self) -> List[float]:
        n = self.config.n_assets
        self.prices = [[100.0] for _ in range(n)]
        self.weights = [0.0] * n + [1.0]  # all cash initially
        self.portfolio_value = self.config.initial_cash
        self.step_count = 0

        # Pre-generate correlated GBM paths
        self._generate_paths()
        return self._get_state()

    def _generate_paths(self):
        n = self.config.n_assets
        length = self.config.episode_length + self.config.lookback + 1
        cfg = self.config

        dt = 1.0 / 252
        self.price_paths = [[100.0] * length for _ in range(n)]

        for t in range(1, length):
            # Correlated normals via Cholesky (simple pairwise correlation)
            eps = [self.rng.gauss(0, 1) for _ in range(n)]
            # Apply correlation: Z_i = sqrt(rho)*M + sqrt(1-rho)*eps_i
            M = self.rng.gauss(0, 1)
            corr = cfg.gbm_corr
            for i in range(n):
                z = math.sqrt(corr) * M + math.sqrt(1-corr) * eps[i]
                ret = (cfg.gbm_mu[i] - 0.5*cfg.gbm_sigma[i]**2)*dt + cfg.gbm_sigma[i]*math.sqrt(dt)*z
                self.price_paths[i][t] = self.price_paths[i][t-1] * math.exp(ret)

        self._t = self.config.lookback

    def _get_state(self) -> List[float]:
        n = self.config.n_assets
        lb = self.config.lookback
        state = []
        # Lookback returns for each asset
        for i in range(n):
            for k in range(lb):
                t = self._t - lb + k + 1
                if t <= 0:
                    state.append(0.0)
                else:
                    ret = (self.price_paths[i][t] - self.price_paths[i][t-1]) / self.price_paths[i][t-1]
                    state.append(ret)
        # Current weights
        state.extend(self.weights)
        return state

    def _softmax(self, x: List[float]) -> List[float]:
        m = max(x)
        exp_x = [math.exp(v - m) for v in x]
        s = sum(exp_x)
        return [v/s for v in exp_x]

    def step(self, action: List[float]) -> Tuple[List[float], float, bool, Dict]:
        # Convert action to target weights via softmax
        target_weights = self._softmax(action)

        # Clip to max position
        max_pos = self.config.max_position_pct
        clipped = [min(w, max_pos) for w in target_weights]
        total = sum(clipped)
        clipped = [w/total for w in clipped]

        # Compute transaction costs
        turnover = sum(abs(clipped[i] - self.weights[i]) for i in range(len(self.weights)))
        tc = turnover * self.config.transaction_cost * self.portfolio_value

        # Advance prices
        self._t += 1
        returns = []
        for i in range(self.config.n_assets):
            r = (self.price_paths[i][self._t] - self.price_paths[i][self._t-1]) / self.price_paths[i][self._t-1]
            returns.append(r)

        # Portfolio return (using old weights for simplicity)
        asset_weights = self.weights[:-1]  # exclude cash
        port_return = sum(asset_weights[i] * returns[i] for i in range(self.config.n_assets))

        self.portfolio_value = self.portfolio_value * (1 + port_return) - tc
        self.weights = clipped
        self.step_count += 1

        reward = math.log(max(self.portfolio_value / self.config.initial_cash, 1e-6))
        done = self.step_count >= self.config.episode_length

        info = {
            'portfolio_value': self.portfolio_value,
            'turnover': turnover,
            'transaction_cost': tc,
            'returns': returns,
        }

        return self._get_state(), reward, done, info
