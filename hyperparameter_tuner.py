"""Simple hyperparameter search utilities.

Provides random search and grid search over user-defined parameter spaces,
with importance analysis via score-variance decomposition.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# LCG random number generator (no external deps)
# ---------------------------------------------------------------------------

def _lcg_next(state: list[int]) -> float:
    """Advance an LCG state and return a float in [0, 1)."""
    # Parameters from Knuth / Numerical Recipes.
    state[0] = (1664525 * state[0] + 1013904223) & 0xFFFFFFFF
    return state[0] / 0x1_0000_0000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HyperparamSpace:
    """Specification of a single hyperparameter's search range."""

    name: str
    low: float
    high: float
    step: float = 0.0          # 0 = continuous; > 0 = discrete grid step
    log_scale: bool = False    # sample / enumerate in log space

    def sample(self, rng_state: list[int]) -> float:
        """Sample one value using the provided LCG state list (mutable)."""
        u = _lcg_next(rng_state)
        if self.log_scale:
            log_low = math.log(self.low) if self.low > 0 else -10.0
            log_high = math.log(self.high) if self.high > 0 else 0.0
            return math.exp(log_low + u * (log_high - log_low))
        return self.low + u * (self.high - self.low)

    def grid_values(self) -> list[float]:
        """Return all discrete values for grid search."""
        if self.step <= 0:
            # Treat as a single-point midpoint if no step specified.
            return [(self.low + self.high) / 2.0]

        if self.log_scale:
            log_low = math.log(self.low) if self.low > 0 else 1e-10
            log_high = math.log(self.high) if self.high > 0 else 1.0
            log_step = math.log(self.step) if self.step > 1 else self.step
            vals = []
            v = log_low
            while v <= log_high + 1e-9:
                vals.append(math.exp(v))
                v += log_step if log_step > 0 else (log_high - log_low)
            return vals if vals else [math.exp(log_low)]

        vals = []
        v = self.low
        while v <= self.high + 1e-9:
            vals.append(round(v, 10))
            v += self.step
        return vals


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""

    params: dict[str, float]
    score: float
    duration_seconds: float


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------

class HyperparamTuner:
    """Searches hyperparameter spaces and tracks results."""

    def __init__(self) -> None:
        self._spaces: list[HyperparamSpace] = []
        self._results: list[TrialResult] = []

    # ── Parameter registration ─────────────────────────────────────────────

    def add_param(self, space: HyperparamSpace) -> None:
        """Register a hyperparameter space."""
        self._spaces.append(space)

    # ── Random search ─────────────────────────────────────────────────────

    def random_search(
        self,
        objective_fn: Callable[[dict], float],
        n_trials: int,
        seed: int = 42,
    ) -> list[TrialResult]:
        """Sample *n_trials* random configurations and evaluate each.

        Uses an LCG PRNG seeded with *seed* — no external randomness needed.
        """
        rng_state = [seed & 0xFFFFFFFF]
        new_results: list[TrialResult] = []

        for _ in range(n_trials):
            params = {sp.name: sp.sample(rng_state) for sp in self._spaces}
            t0 = time.perf_counter()
            score = objective_fn(params)
            duration = time.perf_counter() - t0
            result = TrialResult(params=params, score=score, duration_seconds=duration)
            self._results.append(result)
            new_results.append(result)

        return new_results

    # ── Grid search ───────────────────────────────────────────────────────

    def grid_search(
        self,
        objective_fn: Callable[[dict], float],
    ) -> list[TrialResult]:
        """Enumerate all combinations and evaluate each.

        Only practical for small discrete spaces (step > 0 on all params).
        """
        if not self._spaces:
            return []

        grid_values = [sp.grid_values() for sp in self._spaces]
        names = [sp.name for sp in self._spaces]
        new_results: list[TrialResult] = []

        for combo in itertools.product(*grid_values):
            params = dict(zip(names, combo))
            t0 = time.perf_counter()
            score = objective_fn(params)
            duration = time.perf_counter() - t0
            result = TrialResult(params=params, score=score, duration_seconds=duration)
            self._results.append(result)
            new_results.append(result)

        return new_results

    # ── Best trial ────────────────────────────────────────────────────────

    def best_trial(self) -> Optional[TrialResult]:
        """Return the trial with the highest score, or None if no trials run."""
        if not self._results:
            return None
        return max(self._results, key=lambda r: r.score)

    # ── Importance plot ────────────────────────────────────────────────────

    def plot_importance(self) -> str:
        """ASCII bar chart of parameter importance.

        Importance is estimated as the variance of scores when a parameter
        varies, averaged across all configurations (sensitivity analysis).

        Returns a string suitable for printing to a terminal.
        """
        if not self._results or not self._spaces:
            return "No results to analyse.\n"

        all_scores = [r.score for r in self._results]
        global_mean = sum(all_scores) / len(all_scores)
        global_var = sum((s - global_mean) ** 2 for s in all_scores) / max(len(all_scores), 1)

        importances: dict[str, float] = {}

        for sp in self._spaces:
            # Group results by (quantised) parameter value.
            buckets: dict[str, list[float]] = {}
            for r in self._results:
                val = r.params.get(sp.name, 0.0)
                key = f"{val:.6g}"
                buckets.setdefault(key, []).append(r.score)

            # Compute between-group variance as importance proxy.
            between_var = 0.0
            for scores in buckets.values():
                mean = sum(scores) / len(scores)
                between_var += (mean - global_mean) ** 2 * len(scores)
            between_var /= max(len(self._results), 1)
            importances[sp.name] = between_var

        max_importance = max(importances.values()) if importances else 1.0
        bar_width = 30

        lines = ["Parameter Importance (score variance decomposition)"]
        lines.append("=" * 55)
        for name, imp in sorted(importances.items(), key=lambda kv: kv[1], reverse=True):
            ratio = imp / max_importance if max_importance > 0 else 0.0
            filled = int(ratio * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            lines.append(f"  {name:<20} [{bar}] {imp:.4f}")
        lines.append(f"  {'(global variance)':<20}  {global_var:.4f}")
        lines.append("=" * 55)
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    tuner = HyperparamTuner()
    tuner.add_param(HyperparamSpace("lr", low=1e-4, high=1e-1, log_scale=True))
    tuner.add_param(HyperparamSpace("batch_size", low=16, high=128, step=16))

    def objective(params: dict) -> float:
        lr = params["lr"]
        bs = params["batch_size"]
        # Toy function: peaks around lr=0.001, bs=64
        return -(math.log10(lr) + 3) ** 2 - (bs / 64 - 1) ** 2

    results = tuner.random_search(objective, n_trials=20, seed=42)
    assert len(results) == 20, f"Expected 20 results, got {len(results)}"

    best = tuner.best_trial()
    assert best is not None
    print(f"Best score: {best.score:.4f}", file=sys.stderr)

    grid_results = tuner.grid_search(objective)
    assert len(grid_results) > 0

    print(tuner.plot_importance(), file=sys.stderr)
    print("HyperparamTuner smoke test passed.", file=sys.stderr)
