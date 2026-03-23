"""Unit tests for src/attribution.py — 15+ test cases."""

from __future__ import annotations

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.attribution import (
    AttributionInput,
    AttributionReport,
    AssetAttribution,
    BhbAttribution,
    _normalise,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_input(**kwargs) -> AttributionInput:
    defaults = dict(
        period="2024-Q1",
        portfolio_weights={"BTC": 0.6, "ETH": 0.4},
        benchmark_weights={"BTC": 0.5, "ETH": 0.5},
        portfolio_returns={"BTC": 0.20, "ETH": 0.15},
        benchmark_returns={"BTC": 0.18, "ETH": 0.12},
    )
    defaults.update(kwargs)
    return AttributionInput(**defaults)


ENGINE = BhbAttribution()


# ---------------------------------------------------------------------------
# _normalise tests
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_already_normalised(self):
        w = {"A": 0.6, "B": 0.4}
        n = _normalise(w)
        assert n == pytest.approx(w)

    def test_scales_to_one(self):
        w = {"A": 2.0, "B": 3.0}
        n = _normalise(w)
        assert sum(n.values()) == pytest.approx(1.0)
        assert n["A"] == pytest.approx(0.4)

    def test_zero_weights(self):
        w = {"A": 0.0, "B": 0.0}
        n = _normalise(w)
        assert n == {"A": 0.0, "B": 0.0}


# ---------------------------------------------------------------------------
# AttributionInput validation
# ---------------------------------------------------------------------------

class TestAttributionInput:
    def test_valid_input(self):
        inp = make_input()
        assert inp.period == "2024-Q1"

    def test_empty_portfolio_weights_raises(self):
        with pytest.raises(ValueError, match="portfolio_weights"):
            AttributionInput(
                period="X",
                portfolio_weights={},
                benchmark_weights={"BTC": 1.0},
                portfolio_returns={},
                benchmark_returns={},
            )

    def test_empty_benchmark_weights_raises(self):
        with pytest.raises(ValueError, match="benchmark_weights"):
            AttributionInput(
                period="X",
                portfolio_weights={"BTC": 1.0},
                benchmark_weights={},
                portfolio_returns={},
                benchmark_returns={},
            )


# ---------------------------------------------------------------------------
# BHB Attribution formula tests
# ---------------------------------------------------------------------------

class TestBhbAttribution:
    def test_returns_attribution_report(self):
        report = ENGINE.compute(make_input())
        assert isinstance(report, AttributionReport)

    def test_period_preserved(self):
        report = ENGINE.compute(make_input(period="test-period"))
        assert report.period == "test-period"

    def test_by_asset_contains_all_assets(self):
        report = ENGINE.compute(make_input())
        symbols = {a.asset for a in report.by_asset}
        assert "BTC" in symbols
        assert "ETH" in symbols

    def test_total_active_return_equals_sum_of_effects(self):
        report = ENGINE.compute(make_input())
        expected = (
            report.allocation_effect
            + report.selection_effect
            + report.interaction_effect
        )
        assert report.total_active_return == pytest.approx(expected, rel=1e-9)

    def test_asset_total_equals_sum_of_components(self):
        report = ENGINE.compute(make_input())
        for aa in report.by_asset:
            expected = aa.allocation + aa.selection + aa.interaction
            assert aa.total == pytest.approx(expected, rel=1e-9)

    def test_sum_of_asset_allocation_equals_portfolio_allocation(self):
        report = ENGINE.compute(make_input())
        total_alloc = sum(a.allocation for a in report.by_asset)
        assert total_alloc == pytest.approx(report.allocation_effect, rel=1e-9)

    def test_sum_of_asset_selection_equals_portfolio_selection(self):
        report = ENGINE.compute(make_input())
        total_sel = sum(a.selection for a in report.by_asset)
        assert total_sel == pytest.approx(report.selection_effect, rel=1e-9)

    def test_sum_of_asset_interaction_equals_portfolio_interaction(self):
        report = ENGINE.compute(make_input())
        total_inter = sum(a.interaction for a in report.by_asset)
        assert total_inter == pytest.approx(report.interaction_effect, rel=1e-9)

    def test_identical_portfolio_and_benchmark_zero_active(self):
        inp = make_input(
            portfolio_weights={"BTC": 0.5, "ETH": 0.5},
            benchmark_weights={"BTC": 0.5, "ETH": 0.5},
            portfolio_returns={"BTC": 0.10, "ETH": 0.10},
            benchmark_returns={"BTC": 0.10, "ETH": 0.10},
        )
        report = ENGINE.compute(inp)
        assert report.total_active_return == pytest.approx(0.0, abs=1e-12)

    def test_allocation_formula_single_asset(self):
        # Only BTC with known numbers
        # w_p=0.7, w_b=0.5, r_b=0.10, R_b=0.10*0.5+0.20*0.5=0.15
        # allocation = (0.7-0.5)*(0.10-0.15) = 0.2*(-0.05) = -0.01
        inp = make_input(
            portfolio_weights={"BTC": 0.7, "ETH": 0.3},
            benchmark_weights={"BTC": 0.5, "ETH": 0.5},
            portfolio_returns={"BTC": 0.05, "ETH": 0.05},
            benchmark_returns={"BTC": 0.10, "ETH": 0.20},
        )
        report = ENGINE.compute(inp)
        btc = next(a for a in report.by_asset if a.asset == "BTC")
        R_b = 0.10 * 0.5 + 0.20 * 0.5
        expected_alloc = (0.7 - 0.5) * (0.10 - R_b)
        assert btc.allocation == pytest.approx(expected_alloc, rel=1e-9)

    def test_selection_formula_single_asset(self):
        inp = make_input(
            portfolio_weights={"BTC": 0.5, "ETH": 0.5},
            benchmark_weights={"BTC": 0.5, "ETH": 0.5},
            portfolio_returns={"BTC": 0.30, "ETH": 0.10},
            benchmark_returns={"BTC": 0.20, "ETH": 0.10},
        )
        report = ENGINE.compute(inp)
        btc = next(a for a in report.by_asset if a.asset == "BTC")
        # selection = w_b * (r_p - r_b) = 0.5 * (0.30 - 0.20) = 0.05
        assert btc.selection == pytest.approx(0.05, rel=1e-9)

    def test_interaction_formula_single_asset(self):
        inp = make_input(
            portfolio_weights={"BTC": 0.7, "ETH": 0.3},
            benchmark_weights={"BTC": 0.5, "ETH": 0.5},
            portfolio_returns={"BTC": 0.25, "ETH": 0.15},
            benchmark_returns={"BTC": 0.20, "ETH": 0.10},
        )
        report = ENGINE.compute(inp)
        btc = next(a for a in report.by_asset if a.asset == "BTC")
        expected = (0.7 - 0.5) * (0.25 - 0.20)
        assert btc.interaction == pytest.approx(expected, rel=1e-9)

    def test_asset_in_portfolio_not_in_benchmark(self):
        inp = make_input(
            portfolio_weights={"BTC": 0.6, "SOL": 0.4},
            benchmark_weights={"BTC": 1.0},
            portfolio_returns={"BTC": 0.10, "SOL": 0.20},
            benchmark_returns={"BTC": 0.10},
        )
        report = ENGINE.compute(inp)
        symbols = {a.asset for a in report.by_asset}
        assert "SOL" in symbols

    def test_portfolio_return_helper(self):
        inp = make_input(
            portfolio_weights={"BTC": 0.5, "ETH": 0.5},
            benchmark_weights={"BTC": 0.5, "ETH": 0.5},
            portfolio_returns={"BTC": 0.20, "ETH": 0.10},
            benchmark_returns={"BTC": 0.20, "ETH": 0.10},
        )
        rp = ENGINE.portfolio_return(inp)
        assert rp == pytest.approx(0.15, rel=1e-9)

    def test_benchmark_return_helper(self):
        inp = make_input(
            portfolio_weights={"BTC": 0.5, "ETH": 0.5},
            benchmark_weights={"BTC": 0.6, "ETH": 0.4},
            portfolio_returns={"BTC": 0.20, "ETH": 0.10},
            benchmark_returns={"BTC": 0.20, "ETH": 0.10},
        )
        rb = ENGINE.benchmark_return(inp)
        assert rb == pytest.approx(0.20 * 0.6 + 0.10 * 0.4, rel=1e-9)

    def test_three_asset_portfolio(self):
        inp = AttributionInput(
            period="2024-Q2",
            portfolio_weights={"BTC": 0.5, "ETH": 0.3, "BNB": 0.2},
            benchmark_weights={"BTC": 0.4, "ETH": 0.4, "BNB": 0.2},
            portfolio_returns={"BTC": 0.20, "ETH": 0.10, "BNB": 0.05},
            benchmark_returns={"BTC": 0.15, "ETH": 0.08, "BNB": 0.04},
        )
        report = ENGINE.compute(inp)
        assert len(report.by_asset) == 3
        # Verify total active = sum of all individual totals
        total_from_assets = sum(a.total for a in report.by_asset)
        assert report.total_active_return == pytest.approx(total_from_assets, rel=1e-9)
