"""Tests for src/correlation_tracker.py"""
from __future__ import annotations

import math
import pytest

from src.correlation_tracker import CorrelationTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill(tracker, assets, returns_per_asset, ts_start=1.0):
    """Feed multiple assets with specified return lists."""
    for i, (asset, rets) in enumerate(zip(assets, returns_per_asset)):
        for j, r in enumerate(rets):
            tracker.update(asset, r, ts_start + j)


def _perfect_positive():
    """Two series with correlation +1."""
    xs = [float(i) / 10 for i in range(40)]
    return xs, xs[:]  # identical series


def _perfect_negative():
    """Two series with correlation -1."""
    xs = [float(i) / 10 for i in range(40)]
    ys = [-x for x in xs]
    return xs, ys


def _uncorrelated():
    """Two roughly uncorrelated series."""
    xs = [1.0 if i % 2 == 0 else -1.0 for i in range(40)]
    ys = [1.0 if (i + 1) % 2 == 0 else -1.0 for i in range(40)]
    return xs, ys


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_window(self):
        t = CorrelationTracker()
        assert t.window == 30

    def test_custom_window(self):
        t = CorrelationTracker(window=60)
        assert t.window == 60

    def test_too_small_window_raises(self):
        with pytest.raises(ValueError):
            CorrelationTracker(window=1)

    def test_predeclared_assets(self):
        t = CorrelationTracker(assets=["BTC", "ETH"])
        assert "BTC" in t.assets
        assert "ETH" in t.assets

    def test_no_assets_initially_empty(self):
        t = CorrelationTracker()
        assert t.assets == []


# ---------------------------------------------------------------------------
# update / observation_count
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_adds_asset(self):
        t = CorrelationTracker()
        t.update("BTC", 0.01, 1.0)
        assert "BTC" in t.assets

    def test_observation_count_increments(self):
        t = CorrelationTracker()
        t.update("BTC", 0.01, 1.0)
        t.update("BTC", 0.02, 2.0)
        assert t.observation_count("BTC") == 2

    def test_window_cap(self):
        t = CorrelationTracker(window=5)
        for i in range(10):
            t.update("BTC", float(i), float(i))
        assert t.observation_count("BTC") == 5

    def test_unknown_asset_count_zero(self):
        t = CorrelationTracker()
        assert t.observation_count("XRP") == 0

    def test_update_multiple_assets(self):
        t = CorrelationTracker()
        t.update("BTC", 0.01, 1.0)
        t.update("ETH", 0.02, 1.0)
        assert len(t.assets) == 2


# ---------------------------------------------------------------------------
# correlation()
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_unknown_asset_returns_none(self):
        t = CorrelationTracker()
        assert t.correlation("BTC", "ETH") is None

    def test_one_observation_returns_none(self):
        t = CorrelationTracker()
        t.update("BTC", 0.01, 1.0)
        t.update("ETH", 0.02, 1.0)
        assert t.correlation("BTC", "ETH") is None

    def test_perfect_positive_correlation(self):
        t = CorrelationTracker(window=50)
        xs, ys = _perfect_positive()
        for i, (x, y) in enumerate(zip(xs, ys)):
            t.update("A", x, float(i))
            t.update("B", y, float(i))
        corr = t.correlation("A", "B")
        assert corr is not None
        assert math.isclose(corr, 1.0, abs_tol=1e-9)

    def test_perfect_negative_correlation(self):
        t = CorrelationTracker(window=50)
        xs, ys = _perfect_negative()
        for i, (x, y) in enumerate(zip(xs, ys)):
            t.update("A", x, float(i))
            t.update("B", y, float(i))
        corr = t.correlation("A", "B")
        assert corr is not None
        assert math.isclose(corr, -1.0, abs_tol=1e-9)

    def test_correlation_symmetric(self):
        t = CorrelationTracker(window=50)
        xs, ys = _perfect_positive()
        for i, (x, y) in enumerate(zip(xs, ys)):
            t.update("A", x, float(i))
            t.update("B", y, float(i))
        assert t.correlation("A", "B") == t.correlation("B", "A")

    def test_constant_series_returns_none(self):
        t = CorrelationTracker(window=20)
        for i in range(20):
            t.update("A", 1.0, float(i))
            t.update("B", float(i) * 0.01, float(i))
        assert t.correlation("A", "B") is None

    def test_correlation_in_bounds(self):
        t = CorrelationTracker(window=50)
        xs, ys = _uncorrelated()
        for i, (x, y) in enumerate(zip(xs, ys)):
            t.update("A", x, float(i))
            t.update("B", y, float(i))
        corr = t.correlation("A", "B")
        if corr is not None:
            assert -1.0 <= corr <= 1.0


# ---------------------------------------------------------------------------
# full_matrix()
# ---------------------------------------------------------------------------

class TestFullMatrix:
    def test_single_asset_diagonal_one(self):
        t = CorrelationTracker(window=20)
        for i in range(20):
            t.update("BTC", float(i) * 0.01, float(i))
        m = t.full_matrix()
        try:
            # pandas DataFrame
            assert math.isclose(m.loc["BTC", "BTC"], 1.0)
        except AttributeError:
            assert math.isclose(m["BTC"]["BTC"], 1.0)

    def test_matrix_symmetric(self):
        t = CorrelationTracker(window=50)
        xs, ys = _perfect_positive()
        for i, (x, y) in enumerate(zip(xs, ys)):
            t.update("A", x, float(i))
            t.update("B", y, float(i))
        m = t.full_matrix()
        try:
            ab = m.loc["A", "B"]
            ba = m.loc["B", "A"]
        except AttributeError:
            ab = m["A"]["B"]
            ba = m["B"]["A"]
        assert ab == ba

    def test_empty_tracker_returns_empty(self):
        t = CorrelationTracker()
        m = t.full_matrix()
        try:
            assert len(m) == 0
        except TypeError:
            assert m == {}


# ---------------------------------------------------------------------------
# highly_correlated()
# ---------------------------------------------------------------------------

class TestHighlyCorrelated:
    def test_no_pairs_below_threshold(self):
        t = CorrelationTracker(window=50)
        xs, ys = _uncorrelated()
        for i, (x, y) in enumerate(zip(xs, ys)):
            t.update("A", x, float(i))
            t.update("B", y, float(i))
        pairs = t.highly_correlated(threshold=0.99)
        # Should not contain uncorrelated pair
        for a, b, corr in pairs:
            assert abs(corr) >= 0.99

    def test_perfect_corr_appears_in_high_threshold(self):
        t = CorrelationTracker(window=50)
        xs, ys = _perfect_positive()
        for i, (x, y) in enumerate(zip(xs, ys)):
            t.update("A", x, float(i))
            t.update("B", y, float(i))
        pairs = t.highly_correlated(threshold=0.8)
        assert len(pairs) >= 1
        assert math.isclose(pairs[0][2], 1.0, abs_tol=1e-9)

    def test_sorted_by_abs_corr_descending(self):
        t = CorrelationTracker(window=50)
        for i in range(50):
            t.update("A", float(i), float(i))
            t.update("B", float(i), float(i))
            t.update("C", float(-i), float(i))
        pairs = t.highly_correlated(threshold=0.5)
        abs_corrs = [abs(p[2]) for p in pairs]
        assert abs_corrs == sorted(abs_corrs, reverse=True)

    def test_tuple_structure(self):
        t = CorrelationTracker(window=50)
        xs, ys = _perfect_positive()
        for i, (x, y) in enumerate(zip(xs, ys)):
            t.update("A", x, float(i))
            t.update("B", y, float(i))
        pairs = t.highly_correlated(threshold=0.5)
        for pair in pairs:
            assert len(pair) == 3
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)
            assert isinstance(pair[2], float)


# ---------------------------------------------------------------------------
# diversification_score()
# ---------------------------------------------------------------------------

class TestDiversificationScore:
    def test_single_asset_returns_one(self):
        t = CorrelationTracker()
        for i in range(20):
            t.update("BTC", float(i) * 0.01, float(i))
        assert t.diversification_score() == 1.0

    def test_no_assets_returns_one(self):
        t = CorrelationTracker()
        assert t.diversification_score() == 1.0

    def test_perfectly_correlated_score_near_zero(self):
        t = CorrelationTracker(window=50)
        xs, ys = _perfect_positive()
        for i, (x, y) in enumerate(zip(xs, ys)):
            t.update("A", x, float(i))
            t.update("B", y, float(i))
        score = t.diversification_score()
        assert score is not None
        assert math.isclose(score, 0.0, abs_tol=1e-9)

    def test_score_in_bounds(self):
        t = CorrelationTracker(window=50)
        xs, ys = _uncorrelated()
        for i, (x, y) in enumerate(zip(xs, ys)):
            t.update("A", x, float(i))
            t.update("B", y, float(i))
        score = t.diversification_score()
        assert 0.0 <= score <= 1.0

    def test_score_improves_with_uncorrelated_asset(self):
        """Adding an uncorrelated asset should not decrease diversification."""
        t = CorrelationTracker(window=50)
        xs, ys = _perfect_positive()
        for i, (x, y) in enumerate(zip(xs, ys)):
            t.update("A", x, float(i))
            t.update("B", y, float(i))
        score_two = t.diversification_score()
        # Add uncorrelated series
        uncorr = [1.0 if i % 3 == 0 else -0.5 for i in range(40)]
        for i, u in enumerate(uncorr):
            t.update("C", u, float(i))
        score_three = t.diversification_score()
        assert score_three >= score_two - 0.05  # small tolerance for numerical effects


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_single_asset(self):
        t = CorrelationTracker()
        t.update("BTC", 0.01, 1.0)
        t.update("ETH", 0.02, 1.0)
        t.reset("BTC")
        assert t.observation_count("BTC") == 0
        assert t.observation_count("ETH") == 1

    def test_reset_all(self):
        t = CorrelationTracker()
        t.update("BTC", 0.01, 1.0)
        t.update("ETH", 0.02, 1.0)
        t.reset()
        assert t.assets == []

    def test_reset_unknown_asset_noop(self):
        t = CorrelationTracker()
        t.update("BTC", 0.01, 1.0)
        t.reset("XRP")  # should not raise
        assert t.observation_count("BTC") == 1
