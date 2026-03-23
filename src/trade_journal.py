"""Trade journaling and performance analysis for FinRL DeepSeek Crypto Trading.

Provides comprehensive trade record keeping, P&L attribution, and CSV export.
No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import csv
import io
import math
import os
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# TradeRecord
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """A single completed trade.

    Attributes
    ----------
    trade_id:
        Unique identifier for this trade.
    symbol:
        Trading pair / ticker (e.g. "BTC/USDT").
    entry_time:
        Entry timestamp as a string (e.g. "2024-01-15 09:30:00").
    exit_time:
        Exit timestamp as a string.
    entry_price:
        Price at which the position was opened.
    exit_price:
        Price at which the position was closed.
    quantity:
        Number of units traded.
    side:
        "LONG" or "SHORT".
    pnl:
        Realised profit/loss (positive = profit, negative = loss).
    fees:
        Total commissions / fees paid.
    tags:
        Optional list of user-defined labels (e.g. ["breakout", "momentum"]).
    """

    trade_id: str
    symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # "LONG" or "SHORT"
    pnl: float
    fees: float = 0.0
    tags: List[str] = field(default_factory=list)

    @property
    def net_pnl(self) -> float:
        """P&L after fees."""
        return self.pnl - self.fees

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0.0

    @property
    def is_loser(self) -> bool:
        return self.net_pnl < 0.0

    # Derive the YYYY-MM month key from entry_time.
    @property
    def month_key(self) -> str:
        """Return 'YYYY-MM' extracted from entry_time."""
        # Supports ISO formats like "2024-01-15 09:30:00" or "2024-01-15".
        return self.entry_time[:7]


# ---------------------------------------------------------------------------
# TradeJournal
# ---------------------------------------------------------------------------


class TradeJournal:
    """Store and analyse a collection of completed trades."""

    def __init__(self) -> None:
        self._trades: List[TradeRecord] = []

    # ------------------------------------------------------------------
    # Record management
    # ------------------------------------------------------------------

    def record_trade(self, trade: TradeRecord) -> None:
        """Append a completed trade to the journal."""
        self._trades.append(trade)

    def trades(self) -> List[TradeRecord]:
        """Return a copy of all recorded trades (oldest first)."""
        return list(self._trades)

    def clear(self) -> None:
        """Remove all trades from the journal."""
        self._trades.clear()

    def __len__(self) -> int:
        return len(self._trades)

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def win_rate(self) -> float:
        """Fraction of trades with net_pnl > 0.

        Returns
        -------
        float
            Value in [0.0, 1.0].  Returns 0.0 if no trades.
        """
        if not self._trades:
            return 0.0
        winners = sum(1 for t in self._trades if t.is_winner)
        return winners / len(self._trades)

    def profit_factor(self) -> float:
        """Gross profit divided by gross loss.

        Returns
        -------
        float
            > 1.0 means profitable overall.  Returns ``math.inf`` if there
            are no losing trades, and 0.0 if there are no winning trades.
        """
        gross_profit = sum(t.net_pnl for t in self._trades if t.is_winner)
        gross_loss   = sum(-t.net_pnl for t in self._trades if t.is_loser)
        if gross_loss == 0.0:
            return math.inf if gross_profit > 0.0 else 0.0
        return gross_profit / gross_loss

    def average_win(self) -> float:
        """Mean net P&L of winning trades.  Returns 0.0 if none."""
        wins = [t.net_pnl for t in self._trades if t.is_winner]
        return sum(wins) / len(wins) if wins else 0.0

    def average_loss(self) -> float:
        """Mean net P&L (negative) of losing trades.  Returns 0.0 if none."""
        losses = [t.net_pnl for t in self._trades if t.is_loser]
        return sum(losses) / len(losses) if losses else 0.0

    def total_pnl(self) -> float:
        """Sum of net_pnl across all trades."""
        return sum(t.net_pnl for t in self._trades)

    def total_fees(self) -> float:
        """Sum of fees across all trades."""
        return sum(t.fees for t in self._trades)

    # ------------------------------------------------------------------
    # Drawdown / streak metrics
    # ------------------------------------------------------------------

    def max_consecutive_losses(self) -> int:
        """Longest consecutive streak of losing trades.

        Returns
        -------
        int
            Zero if there are no trades or no losing trades.
        """
        max_streak = 0
        current = 0
        for t in self._trades:
            if t.is_loser:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def max_consecutive_wins(self) -> int:
        """Longest consecutive streak of winning trades."""
        max_streak = 0
        current = 0
        for t in self._trades:
            if t.is_winner:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown in cumulative net P&L.

        Returns
        -------
        float
            The drawdown as a positive number (e.g. 500.0 means a $500 drop).
        """
        if not self._trades:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in self._trades:
            cumulative += t.net_pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd

    # ------------------------------------------------------------------
    # Time-series aggregation
    # ------------------------------------------------------------------

    def monthly_pnl(self) -> Dict[str, float]:
        """Aggregate net P&L by calendar month.

        Returns
        -------
        Dict[str, float]
            Keys are "YYYY-MM" strings; values are summed net_pnl.
        """
        result: Dict[str, float] = {}
        for t in self._trades:
            key = t.month_key
            result[key] = result.get(key, 0.0) + t.net_pnl
        return dict(sorted(result.items()))

    def cumulative_pnl_series(self) -> List[float]:
        """Return the running cumulative net P&L after each trade."""
        cumulative = 0.0
        series = []
        for t in self._trades:
            cumulative += t.net_pnl
            series.append(cumulative)
        return series

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_symbol(self, symbol: str) -> "TradeJournal":
        """Return a new journal containing only trades for ``symbol``."""
        sub = TradeJournal()
        for t in self._trades:
            if t.symbol == symbol:
                sub.record_trade(t)
        return sub

    def filter_by_tag(self, tag: str) -> "TradeJournal":
        """Return a new journal containing only trades tagged with ``tag``."""
        sub = TradeJournal()
        for t in self._trades:
            if tag in t.tags:
                sub.record_trade(t)
        return sub

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_csv(self, path: str) -> None:
        """Write all trades to a CSV file at ``path``.

        Parameters
        ----------
        path:
            File system path for the output CSV.  Parent directories must exist.
        """
        fieldnames = [
            "trade_id", "symbol", "entry_time", "exit_time",
            "entry_price", "exit_price", "quantity", "side",
            "pnl", "fees", "net_pnl", "tags",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in self._trades:
                writer.writerow({
                    "trade_id":    t.trade_id,
                    "symbol":      t.symbol,
                    "entry_time":  t.entry_time,
                    "exit_time":   t.exit_time,
                    "entry_price": t.entry_price,
                    "exit_price":  t.exit_price,
                    "quantity":    t.quantity,
                    "side":        t.side,
                    "pnl":         t.pnl,
                    "fees":        t.fees,
                    "net_pnl":     t.net_pnl,
                    "tags":        "|".join(t.tags),
                })

    def export_csv_string(self) -> str:
        """Return CSV content as a string (useful for testing)."""
        buf = io.StringIO()
        fieldnames = [
            "trade_id", "symbol", "entry_time", "exit_time",
            "entry_price", "exit_price", "quantity", "side",
            "pnl", "fees", "net_pnl", "tags",
        ]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for t in self._trades:
            writer.writerow({
                "trade_id":    t.trade_id,
                "symbol":      t.symbol,
                "entry_time":  t.entry_time,
                "exit_time":   t.exit_time,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "quantity":    t.quantity,
                "side":        t.side,
                "pnl":         t.pnl,
                "fees":        t.fees,
                "net_pnl":     t.net_pnl,
                "tags":        "|".join(t.tags),
            })
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def _make_trade(
    trade_id: str = "T001",
    symbol: str = "BTC/USDT",
    entry_time: str = "2024-01-15 09:30:00",
    exit_time: str = "2024-01-15 10:00:00",
    entry_price: float = 40000.0,
    exit_price: float = 41000.0,
    quantity: float = 0.5,
    side: str = "LONG",
    pnl: float = 500.0,
    fees: float = 10.0,
    tags: Optional[List[str]] = None,
) -> TradeRecord:
    return TradeRecord(
        trade_id=trade_id,
        symbol=symbol,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        side=side,
        pnl=pnl,
        fees=fees,
        tags=tags or [],
    )


class TestTradeRecord(unittest.TestCase):
    def test_net_pnl(self):
        t = _make_trade(pnl=500.0, fees=10.0)
        self.assertAlmostEqual(t.net_pnl, 490.0)

    def test_is_winner(self):
        self.assertTrue(_make_trade(pnl=100.0, fees=5.0).is_winner)
        self.assertFalse(_make_trade(pnl=-100.0, fees=5.0).is_winner)

    def test_is_loser(self):
        self.assertTrue(_make_trade(pnl=-50.0, fees=5.0).is_loser)

    def test_month_key(self):
        t = _make_trade(entry_time="2024-03-22 10:00:00")
        self.assertEqual(t.month_key, "2024-03")


class TestTradeJournalBasic(unittest.TestCase):
    def setUp(self):
        self.journal = TradeJournal()

    def test_empty_journal_length(self):
        self.assertEqual(len(self.journal), 0)

    def test_record_trade_increases_length(self):
        self.journal.record_trade(_make_trade())
        self.assertEqual(len(self.journal), 1)

    def test_clear_empties_journal(self):
        self.journal.record_trade(_make_trade())
        self.journal.clear()
        self.assertEqual(len(self.journal), 0)


class TestWinRate(unittest.TestCase):
    def setUp(self):
        self.journal = TradeJournal()

    def test_empty_journal(self):
        self.assertAlmostEqual(self.journal.win_rate(), 0.0)

    def test_all_winners(self):
        for i in range(5):
            self.journal.record_trade(_make_trade(trade_id=str(i), pnl=100.0))
        self.assertAlmostEqual(self.journal.win_rate(), 1.0)

    def test_all_losers(self):
        for i in range(4):
            self.journal.record_trade(_make_trade(trade_id=str(i), pnl=-50.0, fees=0.0))
        self.assertAlmostEqual(self.journal.win_rate(), 0.0)

    def test_mixed(self):
        self.journal.record_trade(_make_trade("t1", pnl=100.0))
        self.journal.record_trade(_make_trade("t2", pnl=-50.0, fees=0.0))
        self.journal.record_trade(_make_trade("t3", pnl=200.0))
        self.assertAlmostEqual(self.journal.win_rate(), 2 / 3, places=6)


class TestProfitFactor(unittest.TestCase):
    def setUp(self):
        self.journal = TradeJournal()

    def test_no_losers_returns_inf(self):
        self.journal.record_trade(_make_trade(pnl=100.0, fees=0.0))
        self.assertEqual(self.journal.profit_factor(), math.inf)

    def test_no_winners_returns_zero(self):
        self.journal.record_trade(_make_trade(pnl=-100.0, fees=0.0))
        self.assertAlmostEqual(self.journal.profit_factor(), 0.0)

    def test_known_profit_factor(self):
        self.journal.record_trade(_make_trade("t1", pnl=300.0, fees=0.0))
        self.journal.record_trade(_make_trade("t2", pnl=-100.0, fees=0.0))
        self.assertAlmostEqual(self.journal.profit_factor(), 3.0)


class TestAverages(unittest.TestCase):
    def setUp(self):
        self.journal = TradeJournal()
        self.journal.record_trade(_make_trade("t1", pnl=200.0, fees=0.0))
        self.journal.record_trade(_make_trade("t2", pnl=100.0, fees=0.0))
        self.journal.record_trade(_make_trade("t3", pnl=-50.0, fees=0.0))
        self.journal.record_trade(_make_trade("t4", pnl=-150.0, fees=0.0))

    def test_average_win(self):
        self.assertAlmostEqual(self.journal.average_win(), 150.0)

    def test_average_loss(self):
        self.assertAlmostEqual(self.journal.average_loss(), -100.0)

    def test_empty_average_win(self):
        j = TradeJournal()
        self.assertAlmostEqual(j.average_win(), 0.0)

    def test_empty_average_loss(self):
        j = TradeJournal()
        self.assertAlmostEqual(j.average_loss(), 0.0)


class TestConsecutiveLosses(unittest.TestCase):
    def test_no_trades(self):
        self.assertEqual(TradeJournal().max_consecutive_losses(), 0)

    def test_all_winners(self):
        j = TradeJournal()
        for i in range(3):
            j.record_trade(_make_trade(str(i), pnl=100.0))
        self.assertEqual(j.max_consecutive_losses(), 0)

    def test_streak(self):
        j = TradeJournal()
        pnls = [100.0, -50.0, -50.0, -50.0, 100.0, -50.0]
        for i, pnl in enumerate(pnls):
            j.record_trade(_make_trade(str(i), pnl=pnl, fees=0.0))
        self.assertEqual(j.max_consecutive_losses(), 3)


class TestMonthlyPnl(unittest.TestCase):
    def test_groups_by_month(self):
        j = TradeJournal()
        j.record_trade(_make_trade("t1", entry_time="2024-01-10", pnl=100.0, fees=0.0))
        j.record_trade(_make_trade("t2", entry_time="2024-01-20", pnl=200.0, fees=0.0))
        j.record_trade(_make_trade("t3", entry_time="2024-02-05", pnl=-50.0, fees=0.0))
        monthly = j.monthly_pnl()
        self.assertAlmostEqual(monthly["2024-01"], 300.0)
        self.assertAlmostEqual(monthly["2024-02"], -50.0)

    def test_sorted_keys(self):
        j = TradeJournal()
        j.record_trade(_make_trade("t1", entry_time="2024-03-01", pnl=0.0, fees=0.0))
        j.record_trade(_make_trade("t2", entry_time="2024-01-01", pnl=0.0, fees=0.0))
        keys = list(j.monthly_pnl().keys())
        self.assertEqual(keys, sorted(keys))


class TestCsvExport(unittest.TestCase):
    def test_csv_has_header(self):
        j = TradeJournal()
        j.record_trade(_make_trade())
        csv_str = j.export_csv_string()
        self.assertIn("trade_id", csv_str)
        self.assertIn("symbol", csv_str)
        self.assertIn("net_pnl", csv_str)

    def test_csv_row_count(self):
        j = TradeJournal()
        for i in range(3):
            j.record_trade(_make_trade(str(i)))
        lines = j.export_csv_string().strip().split("\n")
        # 1 header + 3 rows
        self.assertEqual(len(lines), 4)

    def test_export_csv_file(self):
        import tempfile
        j = TradeJournal()
        j.record_trade(_make_trade("t1", symbol="ETH/USDT", pnl=300.0))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            j.export_csv(path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("ETH/USDT", content)
        finally:
            os.unlink(path)


class TestFilterMethods(unittest.TestCase):
    def setUp(self):
        self.journal = TradeJournal()
        self.journal.record_trade(_make_trade("t1", symbol="BTC/USDT", tags=["breakout"]))
        self.journal.record_trade(_make_trade("t2", symbol="ETH/USDT", tags=["momentum"]))
        self.journal.record_trade(_make_trade("t3", symbol="BTC/USDT", tags=["breakout", "trend"]))

    def test_filter_by_symbol(self):
        btc = self.journal.filter_by_symbol("BTC/USDT")
        self.assertEqual(len(btc), 2)

    def test_filter_by_tag(self):
        breakout = self.journal.filter_by_tag("breakout")
        self.assertEqual(len(breakout), 2)

    def test_filter_by_missing_symbol(self):
        sol = self.journal.filter_by_symbol("SOL/USDT")
        self.assertEqual(len(sol), 0)


if __name__ == "__main__":
    unittest.main()
