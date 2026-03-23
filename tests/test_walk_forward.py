"""Tests for walk_forward.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from walk_forward import (
    WalkForwardConfig,
    WalkForwardEngine,
    WalkForwardResult,
    _MomentumStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_prices(n: int = 400, seed: int = 42, trend: float = 0.0001) -> np.ndarray:
    """Create a synthetic price series with a slight upward drift."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(trend, 0.01, size=n)
    return 100.0 * np.exp(np.cumsum(log_returns))


def make_df(n: int = 400) -> pd.DataFrame:
    return pd.DataFrame({"close": make_prices(n), "volume": np.ones(n) * 1000})


def small_config(**kwargs) -> WalkForwardConfig:
    base = dict(
        train_periods=60,
        test_periods=20,
        step=20,
        min_folds=3,
        n_jobs=1,
        initial_cash=10_000.0,
        seed=0,
    )
    base.update(kwargs)
    return WalkForwardConfig(**base)


# ---------------------------------------------------------------------------
# WalkForwardEngine
# ---------------------------------------------------------------------------

class TestWalkForwardEngine:
    def test_generates_correct_number_of_folds(self) -> None:
        engine = WalkForwardEngine(config=small_config())
        prices = make_prices(400)
        folds = engine._generate_folds(len(prices))
        # (400 - 60 - 20) / 20 + 1 folds
        assert len(folds) >= 3

    def test_run_returns_result_with_folds(self) -> None:
        engine = WalkForwardEngine(config=small_config())
        df = make_df(300)
        result = engine.run(df, price_col="close")
        assert isinstance(result, WalkForwardResult)
        assert len(result.folds) >= small_config().min_folds

    def test_fold_indices_are_sequential(self) -> None:
        engine = WalkForwardEngine(config=small_config())
        result = engine.run(make_df(300))
        indices = sorted(f.fold_index for f in result.folds)
        assert indices == list(range(len(result.folds)))

    def test_fold_windows_do_not_overlap_test_with_train(self) -> None:
        engine = WalkForwardEngine(config=small_config())
        result = engine.run(make_df(300))
        for fold in result.folds:
            assert fold.test_start == fold.train_end

    def test_equity_curves_have_correct_length(self) -> None:
        cfg = small_config(test_periods=20)
        engine = WalkForwardEngine(config=cfg)
        result = engine.run(make_df(300))
        for fold in result.folds:
            assert len(fold.equity_curve) == cfg.test_periods

    def test_total_return_within_sane_bounds(self) -> None:
        engine = WalkForwardEngine(config=small_config())
        result = engine.run(make_df(300))
        for fold in result.folds:
            assert -1.0 <= fold.total_return <= 10.0

    def test_sharpe_is_finite(self) -> None:
        engine = WalkForwardEngine(config=small_config())
        result = engine.run(make_df(300))
        for fold in result.folds:
            assert np.isfinite(fold.sharpe)

    def test_max_drawdown_is_non_positive(self) -> None:
        engine = WalkForwardEngine(config=small_config())
        result = engine.run(make_df(300))
        for fold in result.folds:
            assert fold.max_drawdown <= 0.0

    def test_win_rate_in_zero_one(self) -> None:
        engine = WalkForwardEngine(config=small_config())
        result = engine.run(make_df(300))
        for fold in result.folds:
            assert 0.0 <= fold.win_rate <= 1.0

    def test_raises_when_too_few_folds(self) -> None:
        cfg = small_config(train_periods=200, test_periods=50, min_folds=10)
        engine = WalkForwardEngine(config=cfg)
        with pytest.raises(ValueError, match="folds"):
            engine.run(make_df(300))

    def test_accepts_numpy_array(self) -> None:
        engine = WalkForwardEngine(config=small_config())
        prices = make_prices(300)
        result = engine.run(prices)
        assert len(result.folds) >= small_config().min_folds

    def test_reproducible_with_same_seed(self) -> None:
        prices = make_prices(300)
        engine1 = WalkForwardEngine(config=small_config(seed=7))
        engine2 = WalkForwardEngine(config=small_config(seed=7))
        r1 = engine1.run(prices)
        r2 = engine2.run(prices)
        for f1, f2 in zip(r1.folds, r2.folds):
            np.testing.assert_array_almost_equal(f1.equity_curve, f2.equity_curve)

    def test_custom_strategy_factory(self) -> None:
        class AlwaysLong:
            def fit(self, train):
                pass

            def predict(self, prices):
                return np.ones(len(prices))

        engine = WalkForwardEngine(
            config=small_config(),
            strategy_factory=lambda: AlwaysLong(),
        )
        result = engine.run(make_df(300))
        assert len(result.folds) >= 1


# ---------------------------------------------------------------------------
# WalkForwardResult aggregate methods
# ---------------------------------------------------------------------------

class TestWalkForwardResult:
    def _make_result(self) -> WalkForwardResult:
        engine = WalkForwardEngine(config=small_config())
        return engine.run(make_df(300))

    def test_mean_sharpe_is_finite(self) -> None:
        assert np.isfinite(self._make_result().mean_sharpe())

    def test_mean_return_is_float(self) -> None:
        assert isinstance(self._make_result().mean_return(), float)

    def test_mean_max_drawdown_non_positive(self) -> None:
        assert self._make_result().mean_max_drawdown() <= 0.0

    def test_mean_win_rate_in_zero_one(self) -> None:
        r = self._make_result().mean_win_rate()
        assert 0.0 <= r <= 1.0

    def test_stitched_curve_length(self) -> None:
        result = self._make_result()
        curve = result.stitched_equity_curve()
        total = sum(len(f.equity_curve) for f in result.folds)
        assert len(curve) == total

    def test_stitched_curve_starts_at_first_equity(self) -> None:
        result = self._make_result()
        curve = result.stitched_equity_curve()
        first_fold = sorted(result.folds, key=lambda f: f.fold_index)[0]
        np.testing.assert_almost_equal(curve[0], first_fold.equity_curve[0])

    def test_summary_contains_fold_count(self) -> None:
        result = self._make_result()
        summary = result.summary()
        assert str(len(result.folds)) in summary
        assert "Sharpe" in summary

    def test_empty_result_returns_nan(self) -> None:
        result = WalkForwardResult(folds=[], config=small_config(), price_index=None)
        assert np.isnan(result.mean_sharpe())
        assert np.isnan(result.mean_return())

    def test_plot_does_not_raise(self, tmp_path) -> None:
        result = self._make_result()
        out = tmp_path / "equity.png"
        result.plot_equity_curve(out)  # should not raise even if matplotlib is absent


# ---------------------------------------------------------------------------
# _MomentumStrategy
# ---------------------------------------------------------------------------

class TestMomentumStrategy:
    def test_returns_array_same_length(self) -> None:
        strat = _MomentumStrategy(fast=5, slow=20)
        prices = make_prices(100)
        signals = strat.predict(prices)
        assert len(signals) == len(prices)

    def test_signals_are_bounded(self) -> None:
        strat = _MomentumStrategy()
        prices = make_prices(200)
        signals = strat.predict(prices)
        assert np.all(np.abs(signals) <= 1.0)

    def test_short_series_returns_zeros(self) -> None:
        strat = _MomentumStrategy(fast=5, slow=20)
        signals = strat.predict(make_prices(10))
        assert np.all(signals == 0.0)

    def test_uptrend_gives_long_signals(self) -> None:
        strat = _MomentumStrategy(fast=5, slow=20)
        # Strongly increasing prices
        prices = np.linspace(100, 200, 100)
        signals = strat.predict(prices)
        # After the slow EMA warms up, most signals should be long
        assert np.mean(signals[30:]) > 0.5
