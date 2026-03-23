"""Real-time risk monitoring for the LARSA crypto trading system.

Provides:
- ``RiskLimits`` — configurable risk thresholds.
- ``RiskViolation`` — a single risk rule breach.
- ``RiskManager`` — checks positions and equity curves against limits.

All VaR / CVaR calculations use historical simulation (non-parametric).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RiskLimits:
    """Configurable risk constraint thresholds."""

    max_position_size_pct: float = 0.10  # 10% of portfolio per position
    max_drawdown_pct: float = 0.20        # 20% max drawdown
    max_daily_loss_pct: float = 0.05      # 5% max daily loss
    max_correlation: float = 0.80         # Pearson correlation limit
    min_sharpe_window: int = 20           # Rolling window for Sharpe checks (days)


@dataclass
class RiskViolation:
    """Represents a single risk rule breach."""

    rule: str
    severity: str            # "warn" or "critical"
    current_value: float
    limit_value: float
    message: str


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------


class RiskManager:
    """Real-time portfolio risk monitor.

    Args:
        limits: ``RiskLimits`` instance with the thresholds to enforce.
    """

    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self.limits = limits or RiskLimits()

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_position_size(
        self,
        symbol: str,
        value: float,
        portfolio_value: float,
    ) -> Optional[RiskViolation]:
        """Check whether a single position is too large relative to portfolio.

        Args:
            symbol: Ticker symbol.
            value: Absolute market value of the position.
            portfolio_value: Total portfolio market value.

        Returns:
            ``RiskViolation`` if limit exceeded, else ``None``.
        """
        if portfolio_value <= 0.0:
            return None
        pct = abs(value) / portfolio_value
        limit = self.limits.max_position_size_pct
        if pct > limit:
            severity = "critical" if pct > limit * 1.5 else "warn"
            return RiskViolation(
                rule="max_position_size",
                severity=severity,
                current_value=pct,
                limit_value=limit,
                message=(
                    f"{symbol} position is {pct:.1%} of portfolio "
                    f"(limit: {limit:.1%})"
                ),
            )
        return None

    def check_drawdown(
        self,
        equity_curve: list[float],
    ) -> Optional[RiskViolation]:
        """Check the current drawdown against the maximum allowed.

        Args:
            equity_curve: Ordered list of portfolio equity values.

        Returns:
            ``RiskViolation`` if limit exceeded, else ``None``.
        """
        if len(equity_curve) < 2:
            return None
        arr = np.asarray(equity_curve, dtype=float)
        peak = np.maximum.accumulate(arr)
        current_dd = float((peak[-1] - arr[-1]) / max(peak[-1], 1e-12))
        limit = self.limits.max_drawdown_pct
        if current_dd > limit:
            severity = "critical" if current_dd > limit * 1.5 else "warn"
            return RiskViolation(
                rule="max_drawdown",
                severity=severity,
                current_value=current_dd,
                limit_value=limit,
                message=(
                    f"Drawdown is {current_dd:.1%} (limit: {limit:.1%})"
                ),
            )
        return None

    def check_daily_loss(
        self,
        today_pnl: float,
        portfolio_value: float,
    ) -> Optional[RiskViolation]:
        """Check today's P&L against the maximum daily loss threshold.

        Args:
            today_pnl: Signed P&L for today (negative = loss).
            portfolio_value: Portfolio value at start of day.

        Returns:
            ``RiskViolation`` if limit exceeded, else ``None``.
        """
        if portfolio_value <= 0.0:
            return None
        loss_pct = -today_pnl / portfolio_value  # positive = loss
        limit = self.limits.max_daily_loss_pct
        if loss_pct > limit:
            severity = "critical" if loss_pct > limit * 1.5 else "warn"
            return RiskViolation(
                rule="max_daily_loss",
                severity=severity,
                current_value=loss_pct,
                limit_value=limit,
                message=(
                    f"Daily loss is {loss_pct:.1%} of portfolio "
                    f"(limit: {limit:.1%})"
                ),
            )
        return None

    # ------------------------------------------------------------------
    # Quantile risk measures
    # ------------------------------------------------------------------

    @staticmethod
    def compute_var(
        returns: list[float],
        confidence: float = 0.95,
    ) -> float:
        """Historical Value at Risk (VaR).

        Args:
            returns: List of period returns (e.g. daily P&L / portfolio value).
            confidence: Confidence level (e.g. 0.95 for 95% VaR).

        Returns:
            VaR as a positive number representing the loss quantile.
            E.g. 0.03 means "you will lose at most 3% with 95% confidence".
        """
        if not returns:
            return 0.0
        arr = np.asarray(returns, dtype=float)
        var = float(np.percentile(arr, (1.0 - confidence) * 100.0))
        return -min(var, 0.0)  # return as a positive loss

    @staticmethod
    def compute_cvar(
        returns: list[float],
        confidence: float = 0.95,
    ) -> float:
        """Conditional Value at Risk (CVaR / Expected Shortfall).

        The average loss in the worst (1 - confidence) fraction of scenarios.

        Args:
            returns: List of period returns.
            confidence: Confidence level.

        Returns:
            CVaR as a positive number (the expected loss beyond VaR).
        """
        if not returns:
            return 0.0
        arr = np.asarray(returns, dtype=float)
        var_threshold = np.percentile(arr, (1.0 - confidence) * 100.0)
        tail = arr[arr <= var_threshold]
        if len(tail) == 0:
            return 0.0
        return float(-tail.mean())

    # ------------------------------------------------------------------
    # Correlation check
    # ------------------------------------------------------------------

    @staticmethod
    def correlation_check(
        returns_a: list[float],
        returns_b: list[float],
    ) -> float:
        """Compute Pearson correlation between two return series.

        Args:
            returns_a: Return series for asset A.
            returns_b: Return series for asset B.

        Returns:
            Pearson correlation coefficient in [-1, 1].
        """
        a = np.asarray(returns_a, dtype=float)
        b = np.asarray(returns_b, dtype=float)
        n = min(len(a), len(b))
        if n < 2:
            return 0.0
        a, b = a[:n], b[:n]
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    # ------------------------------------------------------------------
    # Comprehensive risk report
    # ------------------------------------------------------------------

    def risk_report(
        self,
        positions: dict,
        equity_curve: list[float],
        daily_pnls: list[float],
    ) -> dict:
        """Produce a comprehensive risk metrics report.

        Args:
            positions: Dict of {symbol: {"value": float}} open positions.
            equity_curve: Historical portfolio equity values.
            daily_pnls: Historical daily P&L values.

        Returns:
            Dict of risk metrics and any active violations.
        """
        portfolio_value = equity_curve[-1] if equity_curve else 0.0
        violations: list[dict] = []

        # Position size checks
        for symbol, pos_data in positions.items():
            value = pos_data.get("value", 0.0) if isinstance(pos_data, dict) else float(pos_data)
            v = self.check_position_size(symbol, value, portfolio_value)
            if v:
                violations.append({
                    "rule": v.rule,
                    "severity": v.severity,
                    "message": v.message,
                })

        # Drawdown check
        dd_v = self.check_drawdown(equity_curve)
        if dd_v:
            violations.append({
                "rule": dd_v.rule,
                "severity": dd_v.severity,
                "message": dd_v.message,
            })

        # Daily loss check
        if daily_pnls and portfolio_value > 0:
            dl_v = self.check_daily_loss(daily_pnls[-1], portfolio_value)
            if dl_v:
                violations.append({
                    "rule": dl_v.rule,
                    "severity": dl_v.severity,
                    "message": dl_v.message,
                })

        # VaR / CVaR
        returns_series = [
            (equity_curve[i] - equity_curve[i - 1]) / max(equity_curve[i - 1], 1e-12)
            for i in range(1, len(equity_curve))
        ]
        var_95 = self.compute_var(returns_series, 0.95)
        cvar_95 = self.compute_cvar(returns_series, 0.95)

        # Max drawdown
        if len(equity_curve) >= 2:
            arr = np.asarray(equity_curve, dtype=float)
            peak = np.maximum.accumulate(arr)
            dd_series = (peak - arr) / np.maximum(peak, 1e-12)
            max_dd = float(dd_series.max())
        else:
            max_dd = 0.0

        # Rolling Sharpe (last min_sharpe_window periods)
        win = self.limits.min_sharpe_window
        recent_returns = returns_series[-win:] if len(returns_series) >= win else returns_series
        if len(recent_returns) > 1:
            r_arr = np.asarray(recent_returns)
            rolling_sharpe = float(r_arr.mean() / max(r_arr.std(ddof=1), 1e-12) * np.sqrt(252))
        else:
            rolling_sharpe = 0.0

        return {
            "portfolio_value": portfolio_value,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "max_drawdown": max_dd,
            "rolling_sharpe": rolling_sharpe,
            "num_positions": len(positions),
            "violations": violations,
            "violation_count": len(violations),
            "critical_violations": sum(1 for v in violations if v["severity"] == "critical"),
        }
