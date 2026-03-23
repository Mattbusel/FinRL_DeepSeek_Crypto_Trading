"""Strategy Optimizer — exhaustive grid search and random search over
trading strategy hyperparameters.

Typical usage::

    from src.strategy_optimizer import (
        StrategyParams,
        GridSearchOptimizer,
        RandomSearchOptimizer,
        Uniform, Choice,
    )

    def my_backtest(params: StrategyParams) -> dict:
        # ... run backtest, return metric dict
        return {"sharpe": 1.2, "win_rate": 0.55}

    grid = {
        "lookback": [10, 20, 50],
        "threshold": [0.01, 0.02],
        "position_size": [0.1, 0.2],
        "stop_loss_pct": [0.05],
        "take_profit_pct": [0.15],
    }

    opt = GridSearchOptimizer()
    result = opt.search(grid, my_backtest, metric="sharpe")
    print(result.best_params, result.best_score)
"""

from __future__ import annotations

import itertools
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# StrategyParams
# ---------------------------------------------------------------------------


@dataclass
class StrategyParams:
    """All tunable hyperparameters for a trading strategy."""

    lookback: int = 20
    threshold: float = 0.01
    position_size: float = 0.1
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lookback": self.lookback,
            "threshold": self.threshold,
            "position_size": self.position_size,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyParams":
        return cls(
            lookback=int(d.get("lookback", 20)),
            threshold=float(d.get("threshold", 0.01)),
            position_size=float(d.get("position_size", 0.1)),
            stop_loss_pct=float(d.get("stop_loss_pct", 0.05)),
            take_profit_pct=float(d.get("take_profit_pct", 0.15)),
        )

    def validate(self) -> None:
        """Raise ValueError if any parameter is out of a reasonable range."""
        if self.lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {self.lookback}")
        if not (0.0 < self.position_size <= 1.0):
            raise ValueError(f"position_size must be in (0, 1], got {self.position_size}")
        if self.stop_loss_pct <= 0:
            raise ValueError(f"stop_loss_pct must be > 0, got {self.stop_loss_pct}")
        if self.take_profit_pct <= 0:
            raise ValueError(f"take_profit_pct must be > 0, got {self.take_profit_pct}")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ParamResult:
    """Result for a single parameter combination."""

    params: StrategyParams
    score: float
    metrics: Dict[str, float]


@dataclass
class OptimizationResult:
    """Full result of a hyperparameter search."""

    best_params: StrategyParams
    best_score: float
    all_results: List[ParamResult]
    elapsed_s: float
    metric: str

    def n_evaluated(self) -> int:
        return len(self.all_results)

    def worst_score(self) -> float:
        if not self.all_results:
            return float("-inf")
        return min(r.score for r in self.all_results)

    def top_k(self, k: int) -> List[ParamResult]:
        """Return the top-k results by score."""
        return sorted(self.all_results, key=lambda r: r.score, reverse=True)[:k]


# ---------------------------------------------------------------------------
# Grid Search Optimizer
# ---------------------------------------------------------------------------


class GridSearchOptimizer:
    """Exhaustive grid search over a discrete parameter space.

    Args:
        verbose:  If True, log each evaluation.
        seed:     Not used (deterministic), kept for API consistency.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def search(
        self,
        param_grid: Dict[str, List[Any]],
        backtest_fn: Callable[[StrategyParams], Dict[str, float]],
        metric: str = "sharpe",
    ) -> OptimizationResult:
        """Run exhaustive grid search.

        Args:
            param_grid:   Dict mapping param name to list of candidate values.
                          All five StrategyParams keys are expected; missing keys
                          default to the StrategyParams default.
            backtest_fn:  Callable that accepts a StrategyParams and returns a
                          dict of metric name -> float.
            metric:       Key in the returned metrics dict to optimise (maximise).

        Returns:
            OptimizationResult with best params and all individual results.
        """
        t0 = time.perf_counter()
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        combos = list(itertools.product(*values))

        results: List[ParamResult] = []
        best_score = float("-inf")
        best_result: Optional[ParamResult] = None

        for combo in combos:
            d = dict(zip(keys, combo))
            params = StrategyParams.from_dict(d)
            try:
                metrics = backtest_fn(params)
            except Exception as exc:
                log.warning("backtest_fn raised for params %s: %s", d, exc)
                metrics = {metric: float("-inf")}

            score = float(metrics.get(metric, float("-inf")))
            if not math.isfinite(score):
                score = float("-inf")

            pr = ParamResult(params=params, score=score, metrics=metrics)
            results.append(pr)

            if score > best_score:
                best_score = score
                best_result = pr
                if self.verbose:
                    log.info("New best score=%.4f params=%s", score, d)

        elapsed = time.perf_counter() - t0

        if best_result is None:
            best_result = results[0] if results else ParamResult(
                params=StrategyParams(), score=float("-inf"), metrics={}
            )

        return OptimizationResult(
            best_params=best_result.params,
            best_score=best_result.score,
            all_results=results,
            elapsed_s=elapsed,
            metric=metric,
        )


# ---------------------------------------------------------------------------
# Param Distributions (for Random Search)
# ---------------------------------------------------------------------------


class ParamDistribution:
    """Abstract base for a parameter sampling distribution."""

    def sample(self, rng: random.Random) -> Any:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Uniform(ParamDistribution):
    """Uniform continuous distribution over [low, high]."""

    low: float
    high: float

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)


@dataclass
class LogUniform(ParamDistribution):
    """Log-uniform distribution: samples exp(Uniform(log(low), log(high)))."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low <= 0 or self.high <= 0:
            raise ValueError("LogUniform requires low > 0 and high > 0")
        if self.low >= self.high:
            raise ValueError("LogUniform requires low < high")

    def sample(self, rng: random.Random) -> float:
        log_low = math.log(self.low)
        log_high = math.log(self.high)
        return math.exp(rng.uniform(log_low, log_high))


@dataclass
class Choice(ParamDistribution):
    """Discrete uniform choice from a list of options."""

    options: List[Any]

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("Choice requires at least one option")

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.options)


# ---------------------------------------------------------------------------
# Random Search Optimizer
# ---------------------------------------------------------------------------


class RandomSearchOptimizer:
    """Random sampling from continuous or discrete parameter distributions.

    Args:
        n_iter:  Number of random parameter combinations to evaluate.
        seed:    Random seed for reproducibility.
        verbose: If True, log each evaluation.
    """

    def __init__(self, n_iter: int = 50, seed: Optional[int] = None, verbose: bool = False) -> None:
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")
        self.n_iter = n_iter
        self.seed = seed
        self.verbose = verbose
        self._rng = random.Random(seed)

    def search(
        self,
        param_distributions: Dict[str, ParamDistribution],
        backtest_fn: Callable[[StrategyParams], Dict[str, float]],
        metric: str = "sharpe",
    ) -> OptimizationResult:
        """Run random search.

        Args:
            param_distributions:  Dict mapping param name to a ParamDistribution.
            backtest_fn:          Callable accepting StrategyParams, returning
                                  metric dict.
            metric:               Metric key to maximise.

        Returns:
            OptimizationResult with the best sampled params.
        """
        t0 = time.perf_counter()
        results: List[ParamResult] = []
        best_score = float("-inf")
        best_result: Optional[ParamResult] = None

        for i in range(self.n_iter):
            d: Dict[str, Any] = {}
            for key, dist in param_distributions.items():
                d[key] = dist.sample(self._rng)

            params = StrategyParams.from_dict(d)
            try:
                metrics = backtest_fn(params)
            except Exception as exc:
                log.warning("Iteration %d: backtest_fn raised: %s", i, exc)
                metrics = {metric: float("-inf")}

            score = float(metrics.get(metric, float("-inf")))
            if not math.isfinite(score):
                score = float("-inf")

            pr = ParamResult(params=params, score=score, metrics=metrics)
            results.append(pr)

            if score > best_score:
                best_score = score
                best_result = pr
                if self.verbose:
                    log.info("Iteration %d new best score=%.4f params=%s", i, score, d)

        elapsed = time.perf_counter() - t0

        if best_result is None:
            best_result = results[0] if results else ParamResult(
                params=StrategyParams(), score=float("-inf"), metrics={}
            )

        return OptimizationResult(
            best_params=best_result.params,
            best_score=best_result.score,
            all_results=results,
            elapsed_s=elapsed,
            metric=metric,
        )
