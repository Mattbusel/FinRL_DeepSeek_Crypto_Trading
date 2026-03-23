"""Statistical mean reversion trading strategy."""

from dataclasses import dataclass
from typing import List, Dict
import math


@dataclass
class MeanReversionSignal:
    zscore: float
    position: str  # "long", "short", "exit", "stop", "flat"
    entry_threshold: float
    exit_threshold: float


class MeanReversionStrategy:
    def __init__(
        self,
        lookback: int = 20,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_loss_zscore: float = 3.5,
    ):
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore

    def compute_zscore(self, prices: List[float]) -> float:
        """(last_price - mean) / std of the lookback window."""
        if len(prices) < self.lookback:
            window = prices
        else:
            window = prices[-self.lookback:]

        if len(window) < 2:
            return 0.0

        n = len(window)
        mean = sum(window) / n
        variance = sum((p - mean) ** 2 for p in window) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 1e-10
        return (window[-1] - mean) / std

    def generate_signal(self, prices: List[float]) -> MeanReversionSignal:
        z = self.compute_zscore(prices)
        if abs(z) > self.stop_loss_zscore:
            position = "stop"
        elif abs(z) < self.exit_zscore:
            position = "exit"
        elif z < -self.entry_zscore:
            position = "long"
        elif z > self.entry_zscore:
            position = "short"
        else:
            position = "flat"

        return MeanReversionSignal(
            zscore=z,
            position=position,
            entry_threshold=self.entry_zscore,
            exit_threshold=self.exit_zscore,
        )

    def half_life(self, prices: List[float]) -> float:
        """AR(1) regression: half_life = -ln(2)/ln(phi)."""
        if len(prices) < 3:
            return float("nan")

        # Build y = p[t] - p[t-1], x = p[t-1]
        y = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        x = prices[:-1]

        n = len(x)
        if n < 2:
            return float("nan")

        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        den = sum((xi - mean_x) ** 2 for xi in x)
        if abs(den) < 1e-12:
            return float("nan")

        # OLS coefficient on p[t-1]: y = phi * x + c
        phi = num / den
        if phi >= 0 or abs(phi) < 1e-12:
            return float("nan")

        return -math.log(2) / math.log(1 + phi)

    def adf_stat(self, prices: List[float]) -> float:
        """Simplified ADF: regress Δp_t on p_{t-1}, return t-stat of coefficient."""
        if len(prices) < 3:
            return 0.0

        delta = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        lagged = prices[:-1]
        n = len(delta)

        if n < 2:
            return 0.0

        mean_lag = sum(lagged) / n
        mean_del = sum(delta) / n

        num = sum((lagged[i] - mean_lag) * (delta[i] - mean_del) for i in range(n))
        den = sum((l - mean_lag) ** 2 for l in lagged)

        if abs(den) < 1e-12:
            return 0.0

        beta = num / den

        # Residuals
        alpha = mean_del - beta * mean_lag
        residuals = [delta[i] - (alpha + beta * lagged[i]) for i in range(n)]
        sse = sum(r ** 2 for r in residuals)
        se_beta = math.sqrt(sse / max(n - 2, 1) / den) if den > 0 else 1e-10
        return beta / se_beta

    def cointegration_spread(
        self, series_a: List[float], series_b: List[float]
    ) -> List[float]:
        """OLS hedge ratio; returns spread = a - beta*b."""
        n = min(len(series_a), len(series_b))
        if n < 2:
            return []

        a = series_a[:n]
        b = series_b[:n]

        mean_b = sum(b) / n
        mean_a = sum(a) / n
        num = sum((b[i] - mean_b) * (a[i] - mean_a) for i in range(n))
        den = sum((bi - mean_b) ** 2 for bi in b)
        beta = num / den if abs(den) > 1e-12 else 0.0

        return [a[i] - beta * b[i] for i in range(n)]

    def backtest(
        self, prices: List[float], initial_capital: float = 100_000.0
    ) -> Dict:
        """Simple backtest returning {pnl, sharpe, max_drawdown, trades}."""
        if len(prices) < self.lookback + 1:
            return {"pnl": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "trades": 0}

        capital = initial_capital
        position = 0.0  # shares held (positive=long, negative=short)
        entry_price = 0.0
        trades = 0
        daily_returns: List[float] = []
        peak = capital
        max_drawdown = 0.0

        for i in range(self.lookback, len(prices)):
            window = prices[: i + 1]
            signal = self.generate_signal(window)
            price = prices[i]

            if signal.position == "stop" and position != 0.0:
                # Close position
                pnl = position * (price - entry_price)
                capital += pnl
                daily_returns.append(pnl / max(abs(capital), 1))
                position = 0.0
                trades += 1
            elif signal.position == "exit" and position != 0.0:
                pnl = position * (price - entry_price)
                capital += pnl
                daily_returns.append(pnl / max(abs(capital), 1))
                position = 0.0
                trades += 1
            elif signal.position == "long" and position <= 0.0:
                if position < 0.0:
                    pnl = position * (price - entry_price)
                    capital += pnl
                    trades += 1
                unit_size = capital * 0.1 / max(price, 1e-9)
                position = unit_size
                entry_price = price
            elif signal.position == "short" and position >= 0.0:
                if position > 0.0:
                    pnl = position * (price - entry_price)
                    capital += pnl
                    trades += 1
                unit_size = capital * 0.1 / max(price, 1e-9)
                position = -unit_size
                entry_price = price
            else:
                daily_returns.append(0.0)

            # Track drawdown
            if capital > peak:
                peak = capital
            dd = (peak - capital) / max(peak, 1e-9)
            if dd > max_drawdown:
                max_drawdown = dd

        # Close any open position
        if position != 0.0:
            price = prices[-1]
            pnl = position * (price - entry_price)
            capital += pnl

        total_pnl = capital - initial_capital

        # Sharpe
        n_ret = len(daily_returns)
        if n_ret > 1:
            mean_ret = sum(daily_returns) / n_ret
            var_ret = sum((r - mean_ret) ** 2 for r in daily_returns) / (n_ret - 1)
            std_ret = math.sqrt(var_ret) if var_ret > 0 else 1e-10
            sharpe = mean_ret / std_ret * math.sqrt(252)
        else:
            sharpe = 0.0

        return {
            "pnl": total_pnl,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "trades": trades,
        }
