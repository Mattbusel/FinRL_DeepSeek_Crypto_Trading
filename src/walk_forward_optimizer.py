"""Walk-forward optimization framework.

Splits time-series data into train/test windows, runs a strategy on each split,
and aggregates performance metrics to assess out-of-sample robustness.

This module provides WalkForwardConfig, WalkForwardSplit, and WalkForwardOptimizer
with full iterative-split support including expanding windows, gap periods, and
overfitting score computation.
"""

from __future__ import annotations

import math
import statistics
import unittest
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class WalkForwardConfig:
    """Configuration for a walk-forward validation run."""

    n_splits: int = 5
    """Number of train/test folds."""

    train_periods: int = 60
    """Length of each training window (in periods)."""

    test_periods: int = 20
    """Length of each test window (in periods)."""

    gap_periods: int = 0
    """Gap between train end and test start (avoids lookahead bias)."""

    expanding_window: bool = False
    """If True, grow the training window with each split instead of rolling."""


@dataclass
class WalkForwardSplit:
    """Index bounds for a single train/test fold."""

    split_id: int
    train_start: int
    train_end: int   # exclusive
    test_start: int
    test_end: int    # exclusive


class WalkForwardOptimizer:
    """Generate splits, run a strategy, and aggregate results."""

    # ------------------------------------------------------------------
    # Split generation
    # ------------------------------------------------------------------

    def generate_splits(
        self,
        n_periods: int,
        config: WalkForwardConfig,
    ) -> list[WalkForwardSplit]:
        """Generate walk-forward splits for a dataset of `n_periods` rows.

        For rolling windows the training window slides forward by `test_periods`
        each fold; for expanding windows it starts from period 0 each time.
        """
        splits: list[WalkForwardSplit] = []
        step = config.test_periods

        for split_id in range(config.n_splits):
            if config.expanding_window:
                train_start = 0
                train_end = config.train_periods + split_id * step
            else:
                train_start = split_id * step
                train_end = train_start + config.train_periods

            test_start = train_end + config.gap_periods
            test_end = test_start + config.test_periods

            if test_end > n_periods:
                break  # not enough data for this fold

            splits.append(
                WalkForwardSplit(
                    split_id=split_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

        return splits

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        data: list[Any],
        strategy_fn: Callable[[list[Any], list[Any]], dict[str, float]],
        config: WalkForwardConfig,
    ) -> list[dict[str, float]]:
        """Run `strategy_fn(train_data, test_data) -> metrics` for each split.

        Returns a list of metric dicts (one per fold).  Each dict also
        contains `split_id` for traceability.
        """
        splits = self.generate_splits(len(data), config)
        results: list[dict[str, float]] = []

        for split in splits:
            train_data = data[split.train_start : split.train_end]
            test_data = data[split.test_start : split.test_end]
            metrics = strategy_fn(train_data, test_data)
            metrics["split_id"] = float(split.split_id)
            results.append(metrics)

        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_results(
        self,
        results: list[dict[str, float]],
    ) -> dict[str, float]:
        """Compute mean and std for each metric across all splits.

        Returns a flat dict: ``{metric_mean: ..., metric_std: ...}``.
        """
        if not results:
            return {}

        all_keys: set[str] = set()
        for r in results:
            all_keys.update(r.keys())

        agg: dict[str, float] = {}
        for key in sorted(all_keys):
            values = [r[key] for r in results if key in r]
            if not values:
                continue
            agg[f"{key}_mean"] = sum(values) / len(values)
            agg[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0

        return agg

    # ------------------------------------------------------------------
    # Overfitting score
    # ------------------------------------------------------------------

    def overfitting_score(
        self,
        in_sample: list[float],
        out_sample: list[float],
    ) -> float:
        """Pearson correlation between IS and OOS performance across folds.

        A high positive correlation suggests the in-sample results generalise;
        a low or negative correlation suggests overfitting.
        """
        n = min(len(in_sample), len(out_sample))
        if n < 2:
            return 0.0

        xs = in_sample[:n]
        ys = out_sample[:n]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        std_x = math.sqrt(sum((v - mean_x) ** 2 for v in xs))
        std_y = math.sqrt(sum((v - mean_y) ** 2 for v in ys))
        denom = std_x * std_y
        if denom < 1e-12:
            return 0.0
        return cov / denom


# ─── TESTS ────────────────────────────────────────────────────────────────────

class TestWalkForwardOptimizer(unittest.TestCase):

    def _cfg(self, **kwargs: Any) -> WalkForwardConfig:
        defaults = dict(n_splits=3, train_periods=10, test_periods=5, gap_periods=0)
        defaults.update(kwargs)
        return WalkForwardConfig(**defaults)

    def setUp(self) -> None:
        self.opt = WalkForwardOptimizer()
        self.data = list(range(50))

    def test_generate_splits_count(self) -> None:
        cfg = self._cfg(n_splits=4, train_periods=10, test_periods=5)
        splits = self.opt.generate_splits(50, cfg)
        self.assertEqual(len(splits), 4)

    def test_generate_splits_rolling_train_end(self) -> None:
        cfg = self._cfg(n_splits=3, train_periods=10, test_periods=5)
        splits = self.opt.generate_splits(50, cfg)
        self.assertEqual(splits[0].train_start, 0)
        self.assertEqual(splits[0].train_end, 10)
        self.assertEqual(splits[1].train_start, 5)
        self.assertEqual(splits[1].train_end, 15)

    def test_generate_splits_test_range(self) -> None:
        cfg = self._cfg(n_splits=2, train_periods=10, test_periods=5)
        splits = self.opt.generate_splits(50, cfg)
        self.assertEqual(splits[0].test_start, 10)
        self.assertEqual(splits[0].test_end, 15)

    def test_generate_splits_gap(self) -> None:
        cfg = self._cfg(n_splits=2, train_periods=10, test_periods=5, gap_periods=3)
        splits = self.opt.generate_splits(50, cfg)
        self.assertEqual(splits[0].test_start, 13)

    def test_generate_splits_expanding_window(self) -> None:
        cfg = self._cfg(n_splits=3, train_periods=10, test_periods=5, expanding_window=True)
        splits = self.opt.generate_splits(50, cfg)
        self.assertEqual(splits[0].train_start, 0)
        self.assertEqual(splits[1].train_start, 0)
        self.assertEqual(splits[1].train_end, 15)
        self.assertEqual(splits[2].train_end, 20)

    def test_generate_splits_stops_when_data_exhausted(self) -> None:
        cfg = self._cfg(n_splits=100, train_periods=10, test_periods=5)
        splits = self.opt.generate_splits(25, cfg)
        self.assertLess(len(splits), 100)
        for s in splits:
            self.assertLessEqual(s.test_end, 25)

    def test_run_invokes_strategy_correct_number(self) -> None:
        calls: list[int] = []

        def strategy(train: list, test: list) -> dict[str, float]:
            calls.append(1)
            return {"metric": 1.0}

        cfg = self._cfg(n_splits=3)
        self.opt.run(self.data, strategy, cfg)
        self.assertEqual(sum(calls), 3)

    def test_run_correct_train_slice(self) -> None:
        seen: list[list] = []

        def strategy(train: list, test: list) -> dict[str, float]:
            seen.append(train)
            return {}

        cfg = self._cfg(n_splits=1, train_periods=5, test_periods=5)
        self.opt.run(list(range(20)), strategy, cfg)
        self.assertEqual(seen[0], [0, 1, 2, 3, 4])

    def test_run_result_contains_split_id(self) -> None:
        def strategy(train: list, test: list) -> dict[str, float]:
            return {"r": 0.0}

        results = self.opt.run(self.data, strategy, self._cfg(n_splits=2))
        self.assertIn("split_id", results[0])
        self.assertEqual(results[0]["split_id"], 0.0)

    def test_aggregate_mean(self) -> None:
        results = [{"sharpe": 1.0, "split_id": 0.0}, {"sharpe": 3.0, "split_id": 1.0}]
        agg = self.opt.aggregate_results(results)
        self.assertAlmostEqual(agg["sharpe_mean"], 2.0)

    def test_aggregate_std(self) -> None:
        results = [{"metric": 2.0}, {"metric": 4.0}]
        agg = self.opt.aggregate_results(results)
        self.assertAlmostEqual(agg["metric_std"], math.sqrt(2.0), places=9)

    def test_aggregate_empty(self) -> None:
        self.assertEqual(self.opt.aggregate_results([]), {})

    def test_overfitting_score_perfect(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        self.assertAlmostEqual(self.opt.overfitting_score(xs, ys), 1.0, places=9)

    def test_overfitting_score_negative(self) -> None:
        xs = [1.0, 2.0, 3.0]
        ys = [3.0, 2.0, 1.0]
        self.assertAlmostEqual(self.opt.overfitting_score(xs, ys), -1.0, places=9)

    def test_overfitting_score_insufficient_data(self) -> None:
        self.assertEqual(self.opt.overfitting_score([1.0], [2.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
