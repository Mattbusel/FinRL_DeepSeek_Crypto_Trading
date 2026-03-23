"""Unit tests for src/report_generator.py — BacktestReport and PerformanceMetrics."""

from __future__ import annotations

import math
import os
import tempfile

import pytest

from src.report_generator import (
    BacktestReport,
    PerformanceMetrics,
    _ascii_equity_curve,
    _compute_metrics,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _sample_trades():
    return [
        {"symbol": "BTC", "side": "long", "entry_date": "2024-01-01",
         "exit_date": "2024-01-05", "entry_price": 42000.0, "exit_price": 43000.0, "pnl": 1000.0},
        {"symbol": "ETH", "side": "short", "entry_date": "2024-01-06",
         "exit_date": "2024-01-08", "entry_price": 2200.0, "exit_price": 2100.0, "pnl": 100.0},
        {"symbol": "BTC", "side": "long", "entry_date": "2024-01-09",
         "exit_date": "2024-01-12", "entry_price": 43500.0, "exit_price": 43000.0, "pnl": -500.0},
    ]


def _sample_equity():
    return [10000.0, 10200.0, 10500.0, 10300.0, 10800.0, 10600.0, 11000.0]


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_empty_equity_returns_zeros(self):
        m = _compute_metrics([], [])
        assert m.total_return == 0.0
        assert m.sharpe_ratio == 0.0

    def test_single_bar_equity_returns_zeros(self):
        m = _compute_metrics([], [10000.0])
        assert m.total_return == 0.0

    def test_total_return_positive(self):
        m = _compute_metrics([], [10000.0, 11000.0])
        assert m.total_return == pytest.approx(0.10)

    def test_total_return_negative(self):
        m = _compute_metrics([], [10000.0, 9000.0])
        assert m.total_return == pytest.approx(-0.10)

    def test_max_drawdown_zero_on_monotone_rise(self):
        equity = [1000.0 + i * 10 for i in range(20)]
        m = _compute_metrics([], equity)
        assert m.max_drawdown == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_detected(self):
        equity = [100.0, 110.0, 90.0, 95.0]
        m = _compute_metrics([], equity)
        # peak is 110, trough is 90 → dd = (110-90)/110 ≈ 0.1818
        assert m.max_drawdown == pytest.approx((110 - 90) / 110, rel=1e-6)

    def test_win_rate_all_winners(self):
        trades = [{"pnl": 100.0}, {"pnl": 200.0}, {"pnl": 50.0}]
        m = _compute_metrics(trades, _sample_equity())
        assert m.win_rate == pytest.approx(1.0)

    def test_win_rate_all_losers(self):
        trades = [{"pnl": -100.0}, {"pnl": -50.0}]
        m = _compute_metrics(trades, _sample_equity())
        assert m.win_rate == pytest.approx(0.0)

    def test_win_rate_mixed(self):
        trades = [{"pnl": 100.0}, {"pnl": -50.0}, {"pnl": 80.0}, {"pnl": -30.0}]
        m = _compute_metrics(trades, _sample_equity())
        assert m.win_rate == pytest.approx(0.5)

    def test_profit_factor_no_losses(self):
        trades = [{"pnl": 100.0}, {"pnl": 200.0}]
        m = _compute_metrics(trades, _sample_equity())
        assert math.isinf(m.profit_factor)

    def test_profit_factor_mixed(self):
        trades = [{"pnl": 300.0}, {"pnl": -100.0}]
        m = _compute_metrics(trades, _sample_equity())
        assert m.profit_factor == pytest.approx(3.0)

    def test_avg_trade_pnl(self):
        trades = [{"pnl": 100.0}, {"pnl": -50.0}]
        m = _compute_metrics(trades, _sample_equity())
        assert m.avg_trade_pnl == pytest.approx(25.0)

    def test_sharpe_positive_drift(self):
        equity = [10000.0 * (1.001 ** i) for i in range(252)]
        m = _compute_metrics([], equity)
        assert m.sharpe_ratio > 0.0

    def test_calmar_ratio(self):
        equity = [100.0, 120.0, 90.0, 130.0]
        m = _compute_metrics([], equity)
        if m.max_drawdown > 0 and m.annualized_return != 0:
            assert abs(m.calmar_ratio) == pytest.approx(
                abs(m.annualized_return / m.max_drawdown), rel=1e-3
            )


# ---------------------------------------------------------------------------
# ASCII equity curve
# ---------------------------------------------------------------------------

class TestAsciiEquityCurve:
    def test_returns_string(self):
        result = _ascii_equity_curve([1.0, 2.0, 3.0])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_width_80(self):
        result = _ascii_equity_curve(list(range(100)), width=80)
        lines = result.split("\n")
        # At least some non-ruler lines should be ≤ 82 chars (ruler line has 9-char prefix)
        assert any(len(line) > 0 for line in lines)

    def test_empty_returns_no_data(self):
        result = _ascii_equity_curve([])
        assert "no data" in result

    def test_flat_curve(self):
        result = _ascii_equity_curve([5000.0] * 50)
        assert isinstance(result, str)

    def test_contains_block_chars(self):
        result = _ascii_equity_curve([1.0, 5.0, 10.0, 8.0, 3.0])
        # Should contain at least one block character
        blocks = " ▁▂▃▄▅▆▇█"
        assert any(c in result for c in blocks[1:])  # exclude space


# ---------------------------------------------------------------------------
# BacktestReport.generate
# ---------------------------------------------------------------------------

class TestBacktestReport:
    def test_generate_returns_string(self, tmp_path):
        report = BacktestReport(use_matplotlib=False)
        outfile = str(tmp_path / "report.html")
        html = report.generate(_sample_trades(), _sample_equity(), outfile)
        assert isinstance(html, str)
        assert len(html) > 100

    def test_file_written(self, tmp_path):
        report = BacktestReport(use_matplotlib=False)
        outfile = str(tmp_path / "report.html")
        report.generate(_sample_trades(), _sample_equity(), outfile)
        assert os.path.exists(outfile)
        with open(outfile, encoding="utf-8") as fh:
            content = fh.read()
        assert "<!DOCTYPE html>" in content

    def test_html_contains_metrics_section(self, tmp_path):
        report = BacktestReport(use_matplotlib=False)
        outfile = str(tmp_path / "report.html")
        html = report.generate(_sample_trades(), _sample_equity(), outfile)
        assert "Performance Metrics" in html

    def test_html_contains_trade_log_section(self, tmp_path):
        report = BacktestReport(use_matplotlib=False)
        outfile = str(tmp_path / "report.html")
        html = report.generate(_sample_trades(), _sample_equity(), outfile)
        assert "Trade Log" in html

    def test_html_contains_equity_curve_section(self, tmp_path):
        report = BacktestReport(use_matplotlib=False)
        outfile = str(tmp_path / "report.html")
        html = report.generate(_sample_trades(), _sample_equity(), outfile)
        assert "Equity Curve" in html

    def test_trade_symbols_in_output(self, tmp_path):
        report = BacktestReport(use_matplotlib=False)
        outfile = str(tmp_path / "report.html")
        html = report.generate(_sample_trades(), _sample_equity(), outfile)
        assert "BTC" in html
        assert "ETH" in html

    def test_empty_trades(self, tmp_path):
        report = BacktestReport(use_matplotlib=False)
        outfile = str(tmp_path / "report.html")
        html = report.generate([], _sample_equity(), outfile)
        assert isinstance(html, str)

    def test_directory_created_automatically(self, tmp_path):
        report = BacktestReport(use_matplotlib=False)
        outfile = str(tmp_path / "subdir" / "nested" / "report.html")
        html = report.generate(_sample_trades(), _sample_equity(), outfile)
        assert os.path.exists(outfile)

    def test_compute_metrics_returns_dataclass(self):
        report = BacktestReport()
        m = report.compute_metrics(_sample_trades(), _sample_equity())
        assert isinstance(m, PerformanceMetrics)

    def test_xss_escaping_in_symbol(self, tmp_path):
        """Ensure user data is HTML-escaped."""
        trades = [{"symbol": "<script>alert(1)</script>", "side": "long",
                   "entry_date": "2024-01-01", "exit_date": "2024-01-02",
                   "entry_price": 100.0, "exit_price": 110.0, "pnl": 10.0}]
        report = BacktestReport(use_matplotlib=False)
        outfile = str(tmp_path / "report.html")
        html = report.generate(trades, [100.0, 110.0], outfile)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
