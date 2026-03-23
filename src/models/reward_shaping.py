"""Reward engineering and curriculum learning for financial RL."""
import math
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict
from collections import deque


@dataclass
class RewardComponent:
    name: str
    weight: float
    fn: Callable[[Dict], float]


class CompositeReward:
    """Combine multiple reward signals."""

    def __init__(self):
        self.components: List[RewardComponent] = []

    def add(self, name: str, weight: float, fn: Callable[[Dict], float]) -> 'CompositeReward':
        self.components.append(RewardComponent(name, weight, fn))
        return self

    def compute(self, info: Dict) -> float:
        return sum(c.weight * c.fn(info) for c in self.components)

    def breakdown(self, info: Dict) -> Dict[str, float]:
        return {c.name: c.weight * c.fn(info) for c in self.components}


def sharpe_reward(window_returns: List[float], scale: float = 1.0) -> float:
    """Reward based on rolling Sharpe ratio."""
    if len(window_returns) < 5:
        return 0.0
    m = sum(window_returns) / len(window_returns)
    var = sum((r-m)**2 for r in window_returns) / len(window_returns)
    std = math.sqrt(var) + 1e-8
    return scale * m / std


def drawdown_penalty(portfolio_values: List[float], penalty_scale: float = 2.0) -> float:
    """Negative reward proportional to current drawdown."""
    if len(portfolio_values) < 2:
        return 0.0
    peak = max(portfolio_values)
    current = portfolio_values[-1]
    dd = (peak - current) / peak if peak > 0 else 0.0
    return -penalty_scale * dd


def turnover_penalty(turnover: float, threshold: float = 0.1, scale: float = 0.5) -> float:
    """Penalize excessive trading."""
    excess = max(turnover - threshold, 0.0)
    return -scale * excess


def tail_risk_penalty(returns: List[float], var_confidence: float = 0.95) -> float:
    """Penalize based on CVaR (expected shortfall)."""
    if len(returns) < 10:
        return 0.0
    sorted_r = sorted(returns)
    cutoff_idx = int(len(sorted_r) * (1 - var_confidence))
    tail = sorted_r[:max(cutoff_idx, 1)]
    cvar = sum(tail) / len(tail)
    return cvar  # negative if losses are bad


class CurriculumScheduler:
    """Progressive difficulty curriculum for financial RL training."""

    def __init__(self, n_stages: int = 5, episodes_per_stage: int = 500):
        self.n_stages = n_stages
        self.episodes_per_stage = episodes_per_stage
        self.episode = 0
        self.stage = 0

    def step(self) -> int:
        self.episode += 1
        self.stage = min(self.episode // self.episodes_per_stage, self.n_stages - 1)
        return self.stage

    def market_difficulty(self) -> Dict:
        """Return env config modifiers based on current stage."""
        progress = self.stage / max(self.n_stages - 1, 1)
        return {
            'vol_multiplier': 0.5 + progress * 1.0,     # 0.5x to 1.5x vol
            'transaction_cost': 0.002 * (1 - progress * 0.5),  # reduce costs over time
            'n_assets': 2 + int(progress * 3),           # start with 2, add up to 5
            'episode_length': 50 + int(progress * 202),  # short to full year
        }

    def reward_scaling(self) -> float:
        """Scale reward to maintain consistent magnitude across stages."""
        progress = self.stage / max(self.n_stages - 1, 1)
        return 1.0 + progress * 2.0


class RewardNormalizer:
    """Online reward normalization via running statistics."""

    def __init__(self, alpha: float = 0.001, clip: float = 5.0):
        self.mean = 0.0
        self.var = 1.0
        self.alpha = alpha
        self.clip = clip
        self.n = 0

    def normalize(self, reward: float) -> float:
        self.n += 1
        self.mean = (1 - self.alpha) * self.mean + self.alpha * reward
        self.var = (1 - self.alpha) * self.var + self.alpha * (reward - self.mean)**2
        std = math.sqrt(self.var) + 1e-8
        normalized = (reward - self.mean) / std
        return max(-self.clip, min(self.clip, normalized))
