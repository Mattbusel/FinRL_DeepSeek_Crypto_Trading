"""Unit tests for src/strategy_optimizer.py."""

from __future__ import annotations

import math
import unittest

from src.strategy_optimizer import (
    Choice,
    GridSearchOptimizer,
    LogUniform,
    OptimizationResult,
    ParamResult,
    RandomSearchOptimizer,
    StrategyParams,
    Uniform,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sharpe_backtest(params: StrategyParams) -> dict:
    """Synthetic backtest: sharpe = lookback/100 - threshold*10."""
    return {
        "sharpe": params.lookback / 100.0 - params.threshold * 10,
        "win_rate": 0.5,
    }


def _constant_backtest(params: StrategyParams) -> dict:
    return {"sharpe": 1.0, "returns": 0.1}


def _raises_backtest(params: StrategyParams) -> dict:
    raise RuntimeError("simulated error")


# ---------------------------------------------------------------------------
# StrategyParams
# ---------------------------------------------------------------------------

class TestStrategyParams(unittest.TestCase):

    def test_defaults(self):
        p = StrategyParams()
        self.assertEqual(p.lookback, 20)
        self.assertAlmostEqual(p.threshold, 0.01)
        self.assertAlmostEqual(p.position_size, 0.1)
        self.assertAlmostEqual(p.stop_loss_pct, 0.05)
        self.assertAlmostEqual(p.take_profit_pct, 0.15)

    def test_to_dict_round_trip(self):
        p = StrategyParams(lookback=30, threshold=0.02, position_size=0.2,
                           stop_loss_pct=0.03, take_profit_pct=0.12)
        d = p.to_dict()
        p2 = StrategyParams.from_dict(d)
        self.assertEqual(p.lookback, p2.lookback)
        self.assertAlmostEqual(p.threshold, p2.threshold)

    def test_from_dict_defaults_for_missing_keys(self):
        p = StrategyParams.from_dict({})
        self.assertEqual(p.lookback, 20)

    def test_validate_passes_for_valid(self):
        p = StrategyParams(lookback=10, position_size=0.5)
        p.validate()  # no exception

    def test_validate_lookback_zero_raises(self):
        p = StrategyParams(lookback=0)
        with self.assertRaises(ValueError):
            p.validate()

    def test_validate_position_size_zero_raises(self):
        p = StrategyParams(position_size=0.0)
        with self.assertRaises(ValueError):
            p.validate()

    def test_validate_stop_loss_zero_raises(self):
        p = StrategyParams(stop_loss_pct=0.0)
        with self.assertRaises(ValueError):
            p.validate()

    def test_validate_take_profit_zero_raises(self):
        p = StrategyParams(take_profit_pct=0.0)
        with self.assertRaises(ValueError):
            p.validate()


# ---------------------------------------------------------------------------
# GridSearchOptimizer
# ---------------------------------------------------------------------------

class TestGridSearchOptimizer(unittest.TestCase):

    def _simple_grid(self):
        return {
            "lookback": [10, 20, 50],
            "threshold": [0.01, 0.02],
            "position_size": [0.1],
            "stop_loss_pct": [0.05],
            "take_profit_pct": [0.15],
        }

    def test_returns_optimization_result(self):
        opt = GridSearchOptimizer()
        result = opt.search(self._simple_grid(), _sharpe_backtest)
        self.assertIsInstance(result, OptimizationResult)

    def test_evaluates_all_combos(self):
        grid = self._simple_grid()
        n_combos = 3 * 2 * 1 * 1 * 1  # 6
        opt = GridSearchOptimizer()
        result = opt.search(grid, _sharpe_backtest)
        self.assertEqual(len(result.all_results), n_combos)

    def test_best_is_maximum(self):
        opt = GridSearchOptimizer()
        result = opt.search(self._simple_grid(), _sharpe_backtest)
        # lookback=50, threshold=0.01 → sharpe = 0.5 - 0.1 = 0.4
        self.assertAlmostEqual(result.best_score, max(
            r.score for r in result.all_results
        ))

    def test_best_params_lookback_50(self):
        opt = GridSearchOptimizer()
        result = opt.search(self._simple_grid(), _sharpe_backtest)
        self.assertEqual(result.best_params.lookback, 50)

    def test_metric_wins_rate(self):
        def fn(p: StrategyParams) -> dict:
            return {"sharpe": 0.5, "win_rate": p.lookback / 100.0}

        grid = {"lookback": [10, 50, 100], "threshold": [0.01],
                "position_size": [0.1], "stop_loss_pct": [0.05],
                "take_profit_pct": [0.15]}
        opt = GridSearchOptimizer()
        result = opt.search(grid, fn, metric="win_rate")
        self.assertEqual(result.metric, "win_rate")
        self.assertEqual(result.best_params.lookback, 100)

    def test_handles_exception_in_backtest(self):
        opt = GridSearchOptimizer()
        grid = {"lookback": [10], "threshold": [0.01],
                "position_size": [0.1], "stop_loss_pct": [0.05],
                "take_profit_pct": [0.15]}
        result = opt.search(grid, _raises_backtest)
        self.assertEqual(len(result.all_results), 1)
        self.assertEqual(result.best_score, float("-inf"))

    def test_elapsed_positive(self):
        opt = GridSearchOptimizer()
        result = opt.search(self._simple_grid(), _constant_backtest)
        self.assertGreaterEqual(result.elapsed_s, 0.0)

    def test_n_evaluated(self):
        opt = GridSearchOptimizer()
        result = opt.search(self._simple_grid(), _constant_backtest)
        self.assertEqual(result.n_evaluated(), 6)

    def test_top_k(self):
        opt = GridSearchOptimizer()
        result = opt.search(self._simple_grid(), _sharpe_backtest)
        top2 = result.top_k(2)
        self.assertEqual(len(top2), 2)
        self.assertGreaterEqual(top2[0].score, top2[1].score)

    def test_worst_score_less_than_best(self):
        opt = GridSearchOptimizer()
        result = opt.search(self._simple_grid(), _sharpe_backtest)
        self.assertLessEqual(result.worst_score(), result.best_score)

    def test_single_combo(self):
        grid = {"lookback": [10], "threshold": [0.01],
                "position_size": [0.1], "stop_loss_pct": [0.05],
                "take_profit_pct": [0.15]}
        opt = GridSearchOptimizer()
        result = opt.search(grid, _sharpe_backtest)
        self.assertEqual(len(result.all_results), 1)

    def test_param_result_has_metrics(self):
        opt = GridSearchOptimizer()
        result = opt.search(self._simple_grid(), _sharpe_backtest)
        for pr in result.all_results:
            self.assertIn("sharpe", pr.metrics)
            self.assertIn("win_rate", pr.metrics)


# ---------------------------------------------------------------------------
# RandomSearchOptimizer
# ---------------------------------------------------------------------------

class TestRandomSearchOptimizer(unittest.TestCase):

    def _dists(self):
        return {
            "lookback": Choice([10, 20, 50]),
            "threshold": Uniform(0.005, 0.05),
            "position_size": Uniform(0.05, 0.3),
            "stop_loss_pct": LogUniform(0.01, 0.1),
            "take_profit_pct": Uniform(0.1, 0.3),
        }

    def test_n_iter_results(self):
        opt = RandomSearchOptimizer(n_iter=20, seed=42)
        result = opt.search(self._dists(), _sharpe_backtest)
        self.assertEqual(len(result.all_results), 20)

    def test_reproducible_with_same_seed(self):
        opt1 = RandomSearchOptimizer(n_iter=10, seed=0)
        opt2 = RandomSearchOptimizer(n_iter=10, seed=0)
        r1 = opt1.search(self._dists(), _sharpe_backtest)
        r2 = opt2.search(self._dists(), _sharpe_backtest)
        self.assertAlmostEqual(r1.best_score, r2.best_score)

    def test_different_seed_may_differ(self):
        opt1 = RandomSearchOptimizer(n_iter=15, seed=1)
        opt2 = RandomSearchOptimizer(n_iter=15, seed=99)
        r1 = opt1.search(self._dists(), _sharpe_backtest)
        r2 = opt2.search(self._dists(), _sharpe_backtest)
        # Not guaranteed to differ, but very likely with 15 iterations
        self.assertIsInstance(r1.best_score, float)
        self.assertIsInstance(r2.best_score, float)

    def test_n_iter_zero_raises(self):
        with self.assertRaises(ValueError):
            RandomSearchOptimizer(n_iter=0)

    def test_handles_exception(self):
        opt = RandomSearchOptimizer(n_iter=5, seed=42)
        result = opt.search(self._dists(), _raises_backtest)
        self.assertEqual(len(result.all_results), 5)
        for r in result.all_results:
            self.assertEqual(r.score, float("-inf"))

    def test_elapsed_positive(self):
        opt = RandomSearchOptimizer(n_iter=5, seed=42)
        result = opt.search(self._dists(), _sharpe_backtest)
        self.assertGreaterEqual(result.elapsed_s, 0.0)

    def test_uniform_samples_in_range(self):
        d = Uniform(1.0, 2.0)
        import random as _random
        rng = _random.Random(0)
        for _ in range(50):
            v = d.sample(rng)
            self.assertGreaterEqual(v, 1.0)
            self.assertLessEqual(v, 2.0)

    def test_log_uniform_positive(self):
        d = LogUniform(0.001, 0.1)
        import random as _random
        rng = _random.Random(0)
        for _ in range(50):
            v = d.sample(rng)
            self.assertGreater(v, 0.0)
            self.assertLessEqual(v, 0.1)

    def test_log_uniform_invalid_raises(self):
        with self.assertRaises(ValueError):
            LogUniform(0.0, 1.0)
        with self.assertRaises(ValueError):
            LogUniform(1.0, 0.5)

    def test_choice_samples_from_options(self):
        d = Choice([10, 20, 30])
        import random as _random
        rng = _random.Random(0)
        for _ in range(30):
            v = d.sample(rng)
            self.assertIn(v, [10, 20, 30])

    def test_choice_empty_raises(self):
        with self.assertRaises(ValueError):
            Choice([])

    def test_top_k_results(self):
        opt = RandomSearchOptimizer(n_iter=20, seed=7)
        result = opt.search(self._dists(), _sharpe_backtest)
        top3 = result.top_k(3)
        self.assertEqual(len(top3), 3)
        scores = [r.score for r in top3]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
