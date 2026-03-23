"""Backtest Report Generator.

Generates a self-contained HTML report (with optional matplotlib charts or
ASCII fallbacks) from backtest trades and an equity curve.

Typical usage::

    from src.report_generator import BacktestReport

    trades = [
        {"entry_date": "2024-01-02", "exit_date": "2024-01-05", "pnl": 320.0,
         "entry_price": 42000.0, "exit_price": 42800.0, "side": "long", "symbol": "BTC"},
    ]
    equity = [10000.0, 10120.0, 10320.0, 10280.0, 10420.0]

    report = BacktestReport()
    html = report.generate(trades, equity, output_path="/tmp/report.html")
"""

from __future__ import annotations

import html as _html
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional matplotlib
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    _HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# PerformanceMetrics
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    """Aggregated performance statistics for a backtest run."""
    total_return: float = 0.0          # e.g. 0.15 = 15 %
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0          # positive value, e.g. 0.12 = 12 %
    calmar_ratio: float = 0.0
    win_rate: float = 0.0              # fraction of winning trades
    profit_factor: float = 0.0        # gross_profit / gross_loss
    avg_trade_pnl: float = 0.0        # mean PnL per trade in currency units


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_metrics(trades: List[Dict], equity_curve: List[float]) -> PerformanceMetrics:
    """Compute :class:`PerformanceMetrics` from *trades* and *equity_curve*."""
    m = PerformanceMetrics()

    if not equity_curve or len(equity_curve) < 2:
        return m

    start = equity_curve[0]
    end = equity_curve[-1]

    # Total return
    m.total_return = (end - start) / start if start != 0 else 0.0

    # Annualised return — assume each bar is one trading day, 252 days / year
    n_bars = len(equity_curve) - 1
    years = n_bars / 252.0
    if years > 0 and start > 0 and end > 0:
        m.annualized_return = (end / start) ** (1.0 / years) - 1.0

    # Daily returns
    daily_returns = [
        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        for i in range(1, len(equity_curve))
        if equity_curve[i - 1] != 0
    ]

    if daily_returns:
        mean_r = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_r) ** 2 for r in daily_returns) / len(daily_returns)
        std_r = math.sqrt(variance) if variance > 0 else 0.0

        # Sharpe (annualised, risk-free = 0)
        m.sharpe_ratio = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0

        # Sortino
        downside_returns = [r for r in daily_returns if r < 0]
        if downside_returns:
            downside_variance = sum(r ** 2 for r in downside_returns) / len(daily_returns)
            downside_std = math.sqrt(downside_variance)
            m.sortino_ratio = (mean_r / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    m.max_drawdown = max_dd

    # Calmar
    if m.max_drawdown > 0:
        m.calmar_ratio = m.annualized_return / m.max_drawdown

    # Trade-level stats
    pnls = [t.get("pnl", 0.0) for t in trades]
    if pnls:
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        m.win_rate = len(wins) / len(pnls)
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        m.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        m.avg_trade_pnl = sum(pnls) / len(pnls)

    return m


def _ascii_equity_curve(equity_curve: List[float], width: int = 80, height: int = 16) -> str:
    """Render an 80-column ASCII block chart of the equity curve."""
    if not equity_curve:
        return "(no data)"

    blocks = " ▁▂▃▄▅▆▇█"
    n = len(equity_curve)
    # Down-sample to `width` columns
    sampled: List[float] = []
    for col in range(width):
        idx = int(col * n / width)
        sampled.append(equity_curve[min(idx, n - 1)])

    lo = min(sampled)
    hi = max(sampled)
    span = hi - lo if hi != lo else 1.0

    # Build a 2-D grid: rows (height) × cols (width)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for col, val in enumerate(sampled):
        # Normalise to [0, height*8) for block resolution
        norm = (val - lo) / span  # 0.0 – 1.0
        total_eighths = int(norm * height * 8)
        for row in range(height):
            eighths_in_row = total_eighths - row * 8
            if eighths_in_row >= 8:
                grid[height - 1 - row][col] = blocks[8]
            elif eighths_in_row > 0:
                grid[height - 1 - row][col] = blocks[eighths_in_row]

    lines = ["".join(row) for row in grid]
    ruler = f"{'':>8} {'-' * width}"
    caption = f"Min: {lo:,.2f}  Max: {hi:,.2f}  N={n}"
    return "\n".join([ruler] + lines + [ruler, caption])


def _matplotlib_equity_png_b64(equity_curve: List[float]) -> str:
    """Return a base-64-encoded PNG of the equity curve (requires matplotlib)."""
    import base64
    import io

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_curve, color="#2ecc71", linewidth=1.5)
    ax.fill_between(range(len(equity_curve)), equity_curve, alpha=0.15, color="#2ecc71")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _matplotlib_drawdown_png_b64(equity_curve: List[float]) -> str:
    """Return a base-64-encoded PNG of the drawdown series."""
    import base64
    import io

    peak = equity_curve[0] if equity_curve else 1.0
    dd_series = []
    for v in equity_curve:
        if v > peak:
            peak = v
        dd_series.append(-((peak - v) / peak) if peak > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(range(len(dd_series)), dd_series, 0, color="#e74c3c", alpha=0.4)
    ax.plot(dd_series, color="#e74c3c", linewidth=1.0)
    ax.set_title("Drawdown")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# HTML template helpers
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }
h1   { color: #00d4aa; border-bottom: 2px solid #00d4aa; padding-bottom: 8px; }
h2   { color: #7ec8e3; margin-top: 32px; }
table { border-collapse: collapse; width: 100%; margin-top: 12px; }
th   { background: #16213e; color: #00d4aa; padding: 8px 12px; text-align: left; }
td   { padding: 8px 12px; border-bottom: 1px solid #2a2a4a; }
tr:hover td { background: #22223b; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; margin-top: 16px; }
.metric-card { background: #16213e; border-radius: 8px; padding: 16px; }
.metric-label { font-size: 0.78rem; color: #7ec8e3; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 1.6rem; font-weight: bold; color: #00d4aa; margin-top: 4px; }
.positive { color: #2ecc71; }
.negative { color: #e74c3c; }
pre  { background: #16213e; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; color: #a8d8a8; }
img  { max-width: 100%; border-radius: 8px; margin-top: 12px; }
"""


def _metric_card(label: str, value: str, css_class: str = "") -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{_html.escape(label)}</div>'
        f'<div class="metric-value {css_class}">{_html.escape(value)}</div>'
        f"</div>"
    )


def _pct(v: float) -> str:
    return f"{v * 100:+.2f} %"


def _f2(v: float) -> str:
    if math.isinf(v):
        return "inf"
    return f"{v:.2f}"


def _build_html(
    trades: List[Dict],
    equity_curve: List[float],
    metrics: PerformanceMetrics,
    has_mpl: bool,
) -> str:
    # Charts
    equity_section = ""
    drawdown_section = ""

    if has_mpl and _HAS_MATPLOTLIB and equity_curve:
        eq_b64 = _matplotlib_equity_png_b64(equity_curve)
        equity_section = f'<img src="data:image/png;base64,{eq_b64}" alt="Equity curve">'
        dd_b64 = _matplotlib_drawdown_png_b64(equity_curve)
        drawdown_section = f'<img src="data:image/png;base64,{dd_b64}" alt="Drawdown">'
    elif equity_curve:
        ascii_chart = _ascii_equity_curve(equity_curve)
        equity_section = f"<pre>{_html.escape(ascii_chart)}</pre>"

    # Metrics cards
    mcard = _metric_card
    pos = "positive"
    neg = "negative"
    tr_class = pos if metrics.total_return >= 0 else neg
    ann_class = pos if metrics.annualized_return >= 0 else neg
    sharpe_class = pos if metrics.sharpe_ratio >= 1 else ""

    metric_cards = "\n".join([
        mcard("Total Return", _pct(metrics.total_return), tr_class),
        mcard("Annualised Return", _pct(metrics.annualized_return), ann_class),
        mcard("Sharpe Ratio", _f2(metrics.sharpe_ratio), sharpe_class),
        mcard("Sortino Ratio", _f2(metrics.sortino_ratio)),
        mcard("Max Drawdown", _pct(-metrics.max_drawdown), neg if metrics.max_drawdown > 0 else pos),
        mcard("Calmar Ratio", _f2(metrics.calmar_ratio)),
        mcard("Win Rate", _pct(metrics.win_rate)),
        mcard("Profit Factor", _f2(metrics.profit_factor)),
        mcard("Avg Trade PnL", f"${metrics.avg_trade_pnl:+,.2f}",
              pos if metrics.avg_trade_pnl >= 0 else neg),
    ])

    # Trade table
    trade_rows = ""
    for i, t in enumerate(trades, 1):
        pnl = t.get("pnl", 0.0)
        pnl_class = "positive" if pnl >= 0 else "negative"
        trade_rows += (
            f"<tr>"
            f"<td>{i}</td>"
            f"<td>{_html.escape(str(t.get('symbol', '')))}</td>"
            f"<td>{_html.escape(str(t.get('side', '')))}</td>"
            f"<td>{_html.escape(str(t.get('entry_date', '')))}</td>"
            f"<td>{_html.escape(str(t.get('exit_date', '')))}</td>"
            f"<td>{t.get('entry_price', 0.0):,.4f}</td>"
            f"<td>{t.get('exit_price', 0.0):,.4f}</td>"
            f'<td class="{pnl_class}">${pnl:+,.2f}</td>'
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Backtest Report</h1>

<h2>Equity Curve</h2>
{equity_section}

<h2>Performance Metrics</h2>
<div class="metric-grid">
{metric_cards}
</div>

<h2>Drawdown</h2>
{drawdown_section if drawdown_section else '<p>(see equity curve)</p>'}

<h2>Trade Log</h2>
<table>
<thead>
<tr>
  <th>#</th><th>Symbol</th><th>Side</th>
  <th>Entry Date</th><th>Exit Date</th>
  <th>Entry Price</th><th>Exit Price</th><th>PnL</th>
</tr>
</thead>
<tbody>
{trade_rows}
</tbody>
</table>

</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# BacktestReport
# ---------------------------------------------------------------------------

class BacktestReport:
    """Generates an HTML report from backtest results.

    Parameters:
        use_matplotlib: If ``True`` (default), attempt to use matplotlib for
                        charts.  Falls back to ASCII charts if matplotlib is
                        unavailable.
    """

    def __init__(self, use_matplotlib: bool = True) -> None:
        self._use_matplotlib = use_matplotlib

    def compute_metrics(
        self,
        trades: List[Dict],
        equity_curve: List[float],
    ) -> PerformanceMetrics:
        """Compute and return :class:`PerformanceMetrics` without generating HTML."""
        return _compute_metrics(trades, equity_curve)

    def generate(
        self,
        trades: List[Dict],
        equity_curve: List[float],
        output_path: str,
    ) -> str:
        """Generate the HTML report, save it to *output_path*, and return the HTML string.

        Args:
            trades:       List of trade dicts.  Expected keys:
                          ``symbol``, ``side``, ``entry_date``, ``exit_date``,
                          ``entry_price``, ``exit_price``, ``pnl``.
                          Missing keys are treated as empty / 0.
            equity_curve: Sequence of portfolio equity values (one per bar).
            output_path:  Path where the ``.html`` file will be written.

        Returns:
            The generated HTML as a string.
        """
        metrics = _compute_metrics(trades, equity_curve)
        has_mpl = self._use_matplotlib
        html_content = _build_html(trades, equity_curve, metrics, has_mpl)

        # Ensure directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html_content)

        return html_content
