"""Rich trade logging and performance review for the LARSA trading system.

Every trade is recorded with full context: entry signals, active market regime,
model confidence, outcome, and post-trade analysis.  A weekly performance
review is auto-generated and can be exported as a self-contained HTML report.

Key components
--------------
* :class:`TradeRecord` -- immutable record of a single trade lifecycle.
* :class:`TradingJournal` -- persistent in-memory store with CSV backing.
* :class:`PostTradeAnalyzer` -- computes correctness and attribution.
* :class:`WeeklyReviewGenerator` -- aggregates stats and generates the report.
* :func:`export_html_report` -- renders the weekly review as HTML.

Example::

    from trading_journal import TradingJournal

    journal = TradingJournal(path="./output/journal.csv")
    journal.record_entry(
        symbol="BTC/USDT",
        action="buy",
        price=65000.0,
        signals={"transformer_mu": 0.003, "sentiment": 0.7},
        regime="BULL",
        confidence=0.82,
    )
    # ... later ...
    journal.record_exit(trade_id, exit_price=67000.0)
    report = journal.weekly_review()
    journal.export_html(report, path="./output/weekly_report.html")
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """A single trade's full lifecycle record.

    Attributes:
        trade_id: Unique hash-based identifier.
        symbol: Trading pair (e.g. ``"BTC/USDT"``).
        action: ``"buy"`` or ``"sell"``.
        entry_price: Price at which the trade was opened.
        exit_price: Price at which the trade was closed (``None`` if open).
        entry_timestamp: Unix seconds when the trade was opened.
        exit_timestamp: Unix seconds when the trade was closed.
        signals: Dict of signal name → value active at entry.
        regime: Market regime label at entry (e.g. ``"BULL"``).
        confidence: Model confidence at entry in [0, 1].
        pnl_pct: Percentage return of the trade (set on close).
        prediction_correct: True if direction of move matched prediction.
        notes: Analyst/algorithm commentary.
    """

    trade_id: str
    symbol: str
    action: str
    entry_price: float
    exit_price: Optional[float]
    entry_timestamp: int
    exit_timestamp: Optional[int]
    signals: Dict[str, float]
    regime: str
    confidence: float
    pnl_pct: Optional[float] = None
    prediction_correct: Optional[bool] = None
    notes: str = ""


@dataclass
class WeeklyReview:
    """Auto-generated weekly performance summary.

    Attributes:
        week_start: ISO date string of the week's Monday.
        week_end: ISO date string of the week's Sunday.
        total_trades: Total closed trades in the week.
        win_rate: Fraction of profitable trades.
        mean_pnl_pct: Mean return per trade.
        best_trade_pct: Best single-trade return.
        worst_trade_pct: Worst single-trade return.
        regime_breakdown: Dict mapping regime → trade count.
        signal_accuracy: Dict mapping signal → prediction accuracy.
        prediction_accuracy: Overall fraction of correct direction predictions.
        notes: Auto-generated observations.
    """

    week_start: str
    week_end: str
    total_trades: int
    win_rate: float
    mean_pnl_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    regime_breakdown: Dict[str, int]
    signal_accuracy: Dict[str, float]
    prediction_accuracy: float
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Post-trade analyser
# ---------------------------------------------------------------------------


class PostTradeAnalyzer:
    """Analyses closed trades to determine prediction correctness.

    Args:
        direction_threshold: Minimum absolute PnL (%) to call a trade
            directionally significant (default 0.1).
    """

    def __init__(self, direction_threshold: float = 0.1) -> None:
        self.direction_threshold = direction_threshold

    def analyse(self, record: TradeRecord) -> TradeRecord:
        """Annotate *record* with prediction correctness and notes.

        A trade is marked correct when:
        - ``action == "buy"`` and ``pnl_pct > threshold`` (long paid off), or
        - ``action == "sell"`` and ``pnl_pct < -threshold`` (short paid off).

        Args:
            record: A closed :class:`TradeRecord` (must have ``exit_price``).

        Returns:
            Updated record with ``prediction_correct`` and ``notes`` set.
        """
        if record.exit_price is None or record.pnl_pct is None:
            return record

        pnl = record.pnl_pct
        thr = self.direction_threshold

        if record.action == "buy":
            correct = pnl > thr
        elif record.action == "sell":
            correct = pnl < -thr
        else:
            correct = None

        notes_parts = []
        if correct is True:
            notes_parts.append(f"Direction correct (pnl={pnl:.2f}%).")
        elif correct is False:
            notes_parts.append(f"Direction incorrect (pnl={pnl:.2f}%).")
        else:
            notes_parts.append("Direction ambiguous (within threshold).")

        # Signal attribution
        dominant = self._dominant_signal(record.signals)
        if dominant:
            notes_parts.append(f"Dominant signal at entry: {dominant}.")

        if record.regime:
            notes_parts.append(f"Regime at entry: {record.regime}.")

        record.prediction_correct = correct
        record.notes = " ".join(notes_parts)
        return record

    @staticmethod
    def _dominant_signal(signals: Dict[str, float]) -> Optional[str]:
        if not signals:
            return None
        return max(signals, key=lambda k: abs(signals[k]))


# ---------------------------------------------------------------------------
# Trading journal
# ---------------------------------------------------------------------------


class TradingJournal:
    """Persistent trade journal with CSV backing and HTML export.

    Args:
        path: Path to the CSV file used for persistence.
            Created automatically if it does not exist.
        analyzer: :class:`PostTradeAnalyzer` instance; created with defaults
            when not supplied.
    """

    _CSV_FIELDS = [
        "trade_id", "symbol", "action", "entry_price", "exit_price",
        "entry_timestamp", "exit_timestamp", "signals", "regime",
        "confidence", "pnl_pct", "prediction_correct", "notes",
    ]

    def __init__(
        self,
        path: str = "./output/journal.csv",
        analyzer: Optional[PostTradeAnalyzer] = None,
    ) -> None:
        self.path = path
        self.analyzer = analyzer or PostTradeAnalyzer()
        self._records: Dict[str, TradeRecord] = {}
        self._ensure_file()
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_entry(
        self,
        symbol: str,
        action: str,
        price: float,
        signals: Optional[Dict[str, float]] = None,
        regime: str = "UNKNOWN",
        confidence: float = 0.5,
    ) -> str:
        """Open a new trade and return its unique trade_id.

        Args:
            symbol: Ticker pair, e.g. ``"BTC/USDT"``.
            action: ``"buy"`` or ``"sell"``.
            price: Entry price.
            signals: Dict of signal name → value.
            regime: Market regime label.
            confidence: Model confidence in [0, 1].

        Returns:
            Unique ``trade_id`` string.
        """
        ts = int(time.time())
        raw = f"{symbol}:{action}:{price}:{ts}"
        trade_id = hashlib.sha1(raw.encode()).hexdigest()[:12]

        record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            action=action,
            entry_price=price,
            exit_price=None,
            entry_timestamp=ts,
            exit_timestamp=None,
            signals=signals or {},
            regime=regime,
            confidence=confidence,
        )
        self._records[trade_id] = record
        self._append_row(record)
        return trade_id

    def record_exit(self, trade_id: str, exit_price: float) -> Optional[TradeRecord]:
        """Close an existing trade and run post-trade analysis.

        Args:
            trade_id: ID returned by :meth:`record_entry`.
            exit_price: Exit price.

        Returns:
            Updated :class:`TradeRecord`, or ``None`` when *trade_id* is
            unknown.
        """
        record = self._records.get(trade_id)
        if record is None:
            return None

        record.exit_price = exit_price
        record.exit_timestamp = int(time.time())

        if record.action == "buy":
            record.pnl_pct = (exit_price - record.entry_price) / record.entry_price * 100.0
        else:
            record.pnl_pct = (record.entry_price - exit_price) / record.entry_price * 100.0

        record = self.analyzer.analyse(record)
        self._records[trade_id] = record
        self._rewrite_csv()
        return record

    def weekly_review(self, week_offset: int = 0) -> WeeklyReview:
        """Generate a weekly performance review.

        Args:
            week_offset: 0 = current week, -1 = last week, etc.

        Returns:
            :class:`WeeklyReview` data class.
        """
        from datetime import timedelta

        now = datetime.now(tz=timezone.utc)
        # Monday of the target week
        week_start_dt = now - timedelta(days=now.weekday(), weeks=-week_offset)
        week_start_dt = week_start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        week_end_dt = week_start_dt + timedelta(days=6, hours=23, minutes=59, seconds=59)

        start_ts = int(week_start_dt.timestamp())
        end_ts = int(week_end_dt.timestamp())

        closed = [
            r for r in self._records.values()
            if r.exit_timestamp is not None
            and start_ts <= r.exit_timestamp <= end_ts
        ]

        pnls = [r.pnl_pct for r in closed if r.pnl_pct is not None]
        wins = [p for p in pnls if p > 0]
        win_rate = len(wins) / len(pnls) if pnls else 0.0
        mean_pnl = float(sum(pnls) / len(pnls)) if pnls else 0.0

        regime_breakdown: Dict[str, int] = {}
        for r in closed:
            regime_breakdown[r.regime] = regime_breakdown.get(r.regime, 0) + 1

        correct_calls = [r for r in closed if r.prediction_correct is True]
        pred_acc = len(correct_calls) / len(closed) if closed else 0.0

        # Signal accuracy: per-signal, was it positive when trade was correct?
        signal_counts: Dict[str, List[int]] = {}
        for r in closed:
            if r.prediction_correct is None:
                continue
            for sig, val in r.signals.items():
                if sig not in signal_counts:
                    signal_counts[sig] = []
                direction_match = (r.action == "buy" and val > 0) or (
                    r.action == "sell" and val < 0
                )
                signal_counts[sig].append(
                    1 if (direction_match and r.prediction_correct) else 0
                )
        signal_accuracy = {
            sig: sum(vals) / len(vals) for sig, vals in signal_counts.items() if vals
        }

        # Auto-generated observations
        notes: List[str] = []
        if win_rate >= 0.6:
            notes.append(f"Strong week: win rate {win_rate:.0%}.")
        elif win_rate < 0.4:
            notes.append(f"Difficult week: win rate only {win_rate:.0%}.")
        if pnls:
            notes.append(
                f"Best trade: {max(pnls):.2f}%, Worst: {min(pnls):.2f}%."
            )
        dominant_regime = max(regime_breakdown, key=regime_breakdown.get, default="N/A")
        notes.append(f"Dominant regime: {dominant_regime}.")

        return WeeklyReview(
            week_start=week_start_dt.date().isoformat(),
            week_end=week_end_dt.date().isoformat(),
            total_trades=len(closed),
            win_rate=win_rate,
            mean_pnl_pct=mean_pnl,
            best_trade_pct=max(pnls) if pnls else 0.0,
            worst_trade_pct=min(pnls) if pnls else 0.0,
            regime_breakdown=regime_breakdown,
            signal_accuracy=signal_accuracy,
            prediction_accuracy=pred_acc,
            notes=notes,
        )

    def export_html(self, review: WeeklyReview, path: str) -> str:
        """Export *review* as a self-contained HTML report.

        Args:
            review: :class:`WeeklyReview` to render.
            path: Output file path.

        Returns:
            Absolute path of the written file.
        """
        html = _render_html_report(review, list(self._records.values()))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return os.path.abspath(path)

    def all_records(self) -> List[TradeRecord]:
        """Return all trade records (open and closed)."""
        return list(self._records.values())

    # ------------------------------------------------------------------
    # CSV persistence
    # ------------------------------------------------------------------

    def _ensure_file(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._CSV_FIELDS)
                writer.writeheader()

    def _load(self) -> None:
        try:
            with open(self.path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    record = self._row_to_record(row)
                    self._records[record.trade_id] = record
        except (FileNotFoundError, csv.Error):
            pass

    def _append_row(self, record: TradeRecord) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._CSV_FIELDS)
            writer.writerow(self._record_to_row(record))

    def _rewrite_csv(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._CSV_FIELDS)
            writer.writeheader()
            for record in self._records.values():
                writer.writerow(self._record_to_row(record))

    @staticmethod
    def _record_to_row(r: TradeRecord) -> Dict[str, Any]:
        return {
            "trade_id": r.trade_id,
            "symbol": r.symbol,
            "action": r.action,
            "entry_price": r.entry_price,
            "exit_price": r.exit_price if r.exit_price is not None else "",
            "entry_timestamp": r.entry_timestamp,
            "exit_timestamp": r.exit_timestamp if r.exit_timestamp is not None else "",
            "signals": json.dumps(r.signals),
            "regime": r.regime,
            "confidence": r.confidence,
            "pnl_pct": r.pnl_pct if r.pnl_pct is not None else "",
            "prediction_correct": (
                "" if r.prediction_correct is None else str(r.prediction_correct)
            ),
            "notes": r.notes,
        }

    @staticmethod
    def _row_to_record(row: Dict[str, str]) -> TradeRecord:
        def _float(v: str) -> Optional[float]:
            return float(v) if v not in ("", None) else None

        def _int(v: str) -> Optional[int]:
            return int(v) if v not in ("", None) else None

        def _bool(v: str) -> Optional[bool]:
            if v == "True":
                return True
            if v == "False":
                return False
            return None

        return TradeRecord(
            trade_id=row["trade_id"],
            symbol=row["symbol"],
            action=row["action"],
            entry_price=float(row["entry_price"]),
            exit_price=_float(row["exit_price"]),
            entry_timestamp=int(row["entry_timestamp"]),
            exit_timestamp=_int(row["exit_timestamp"]),
            signals=json.loads(row["signals"] or "{}"),
            regime=row["regime"],
            confidence=float(row["confidence"]),
            pnl_pct=_float(row["pnl_pct"]),
            prediction_correct=_bool(row["prediction_correct"]),
            notes=row["notes"],
        )


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _render_html_report(review: WeeklyReview, records: List[TradeRecord]) -> str:
    """Render a weekly review + trade table as a self-contained HTML page."""

    def _pct(v: float) -> str:
        return f"{v:.2f}%"

    def _color(v: float) -> str:
        return "#2ecc71" if v >= 0 else "#e74c3c"

    signal_rows = "".join(
        f"<tr><td>{sig}</td><td>{acc:.0%}</td></tr>"
        for sig, acc in review.signal_accuracy.items()
    )
    regime_rows = "".join(
        f"<tr><td>{reg}</td><td>{cnt}</td></tr>"
        for reg, cnt in review.regime_breakdown.items()
    )

    # Closed trades in the week window (all records passed in for simplicity)
    closed_records = [r for r in records if r.exit_timestamp is not None]
    trade_rows_html = ""
    for r in sorted(closed_records, key=lambda x: x.exit_timestamp or 0, reverse=True)[:50]:
        pnl_str = _pct(r.pnl_pct) if r.pnl_pct is not None else "open"
        color = _color(r.pnl_pct or 0)
        correct_icon = (
            "Y" if r.prediction_correct
            else ("N" if r.prediction_correct is False else "-")
        )
        trade_rows_html += (
            f"<tr>"
            f"<td>{r.trade_id}</td>"
            f"<td>{r.symbol}</td>"
            f"<td>{r.action.upper()}</td>"
            f"<td>{r.regime}</td>"
            f"<td>{r.confidence:.0%}</td>"
            f"<td style='color:{color};font-weight:bold'>{pnl_str}</td>"
            f"<td>{correct_icon}</td>"
            f"<td style='font-size:0.8em'>{r.notes[:80]}</td>"
            f"</tr>\n"
        )

    notes_html = "".join(f"<li>{n}</li>" for n in review.notes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Weekly Trading Report {review.week_start}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background:#0d1117; color:#c9d1d9; margin:0; padding:20px; }}
  h1,h2 {{ color:#58a6ff; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
  th {{ background:#161b22; color:#58a6ff; padding:8px 12px; text-align:left; }}
  td {{ padding:6px 12px; border-bottom:1px solid #21262d; }}
  tr:hover td {{ background:#161b22; }}
  .kpi {{ display:inline-block; background:#161b22; border:1px solid #30363d;
           border-radius:8px; padding:16px 24px; margin:8px; min-width:140px; text-align:center; }}
  .kpi-label {{ font-size:0.75em; color:#8b949e; }}
  .kpi-value {{ font-size:1.6em; font-weight:bold; }}
  ul {{ color:#8b949e; }}
</style>
</head>
<body>
<h1>Weekly Trading Report</h1>
<p style="color:#8b949e">{review.week_start} &mdash; {review.week_end}</p>

<div>
  <div class="kpi">
    <div class="kpi-label">Trades</div>
    <div class="kpi-value">{review.total_trades}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Win Rate</div>
    <div class="kpi-value" style="color:{'#2ecc71' if review.win_rate>=0.5 else '#e74c3c'}">{review.win_rate:.0%}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Mean PnL</div>
    <div class="kpi-value" style="color:{_color(review.mean_pnl_pct)}">{_pct(review.mean_pnl_pct)}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Best</div>
    <div class="kpi-value" style="color:#2ecc71">{_pct(review.best_trade_pct)}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Worst</div>
    <div class="kpi-value" style="color:#e74c3c">{_pct(review.worst_trade_pct)}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Pred. Accuracy</div>
    <div class="kpi-value">{review.prediction_accuracy:.0%}</div>
  </div>
</div>

<h2>Observations</h2>
<ul>{notes_html}</ul>

<h2>Regime Breakdown</h2>
<table>
<tr><th>Regime</th><th>Trades</th></tr>
{regime_rows}
</table>

<h2>Signal Accuracy</h2>
<table>
<tr><th>Signal</th><th>Directional Accuracy</th></tr>
{signal_rows}
</table>

<h2>Trade Log</h2>
<table>
<tr>
  <th>ID</th><th>Symbol</th><th>Action</th><th>Regime</th>
  <th>Conf.</th><th>PnL</th><th>Correct</th><th>Notes</th>
</tr>
{trade_rows_html}
</table>
<p style="color:#484f58;font-size:0.8em">Generated by LARSA TradingJournal</p>
</body>
</html>"""
