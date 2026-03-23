"""Position sizing methods: Kelly, fixed fraction, vol-targeting, ATR-based."""

import math
from typing import List


# ---------------------------------------------------------------------------
# Standalone sizing functions
# ---------------------------------------------------------------------------


def kelly_fraction(
    win_prob: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.5,
) -> float:
    """
    Kelly criterion position fraction (win_prob / avg_loss - (1-win_prob) / avg_win),
    scaled by `fraction` (default 0.5 = half-Kelly).

    Parameters
    ----------
    win_prob  : probability of a winning trade (0–1)
    avg_win   : average profit per winning trade (positive)
    avg_loss  : average loss per losing trade (positive magnitude)
    fraction  : scaling factor (1.0 = full Kelly, 0.5 = half-Kelly)

    Returns
    -------
    float : fraction of capital to risk (clamped to [0, 1])
    """
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0
    q = 1.0 - win_prob
    kelly = win_prob / avg_loss - q / avg_win
    return max(0.0, min(1.0, kelly * fraction))


def fixed_fraction(capital: float, fraction: float, price: float) -> int:
    """
    Fixed-fraction position sizing: risk a constant % of capital.

    Returns number of shares (integer, floored).
    """
    if price <= 0:
        return 0
    dollar_risk = capital * max(0.0, min(1.0, fraction))
    return int(dollar_risk / price)


def volatility_target(
    capital: float,
    target_vol: float,
    asset_vol: float,
    price: float,
) -> int:
    """
    Volatility-targeting position sizing.

    Sizes the position so its annual volatility contribution equals target_vol
    as a fraction of capital.

    Returns number of shares.
    """
    if asset_vol <= 0 or price <= 0:
        return 0
    dollar_position = capital * (target_vol / asset_vol)
    return int(dollar_position / price)


def atr_sizing(
    capital: float,
    atr: float,
    risk_pct: float,
    price: float,
) -> int:
    """
    ATR-based position sizing with a stop-loss set at 1 ATR below entry.

    risk_pct : fraction of capital to risk per trade (e.g. 0.01 = 1%)
    atr      : average true range (same units as price)

    Returns number of shares.
    """
    if atr <= 0 or price <= 0:
        return 0
    dollar_risk = capital * risk_pct
    risk_per_share = atr  # stop is placed 1 ATR away
    if risk_per_share <= 0:
        return 0
    return int(dollar_risk / risk_per_share)


# ---------------------------------------------------------------------------
# PositionSizer class
# ---------------------------------------------------------------------------


class PositionSizer:
    """
    Unified position sizer supporting multiple sizing methods.

    Methods: 'volatility_target', 'kelly', 'fixed_fraction', 'atr'
    """

    _VALID_METHODS = {"volatility_target", "kelly", "fixed_fraction", "atr"}

    def __init__(
        self,
        capital: float,
        method: str = "volatility_target",
        **kwargs,
    ) -> None:
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {self._VALID_METHODS}."
            )
        self.capital = capital
        self.method = method
        self.kwargs = kwargs

    def size(
        self,
        price: float,
        volatility: float,
        signal_strength: float = 1.0,
    ) -> int:
        """
        Compute a base position size and scale by signal strength.

        Parameters
        ----------
        price           : current asset price
        volatility      : annualized volatility of the asset
        signal_strength : conviction scalar in [0, 1] (default 1.0 = full size)

        Returns
        -------
        int : number of shares
        """
        if self.method == "volatility_target":
            target_vol = self.kwargs.get("target_vol", 0.10)
            base = volatility_target(self.capital, target_vol, volatility, price)
        elif self.method == "kelly":
            win_prob = self.kwargs.get("win_prob", 0.55)
            avg_win = self.kwargs.get("avg_win", 1.0)
            avg_loss = self.kwargs.get("avg_loss", 1.0)
            frac = kelly_fraction(win_prob, avg_win, avg_loss)
            base = fixed_fraction(self.capital, frac, price)
        elif self.method == "fixed_fraction":
            fraction = self.kwargs.get("fraction", 0.01)
            base = fixed_fraction(self.capital, fraction, price)
        elif self.method == "atr":
            atr = self.kwargs.get("atr", volatility * price)
            risk_pct = self.kwargs.get("risk_pct", 0.01)
            base = atr_sizing(self.capital, atr, risk_pct, price)
        else:
            base = 0

        return self.scale_by_conviction(base, signal_strength)

    def max_position(
        self,
        capital: float,
        price: float,
        concentration_limit: float = 0.10,
    ) -> int:
        """
        Maximum position size given a concentration limit.

        concentration_limit : max fraction of capital in a single position.
        """
        if price <= 0:
            return 0
        max_dollars = capital * concentration_limit
        return int(max_dollars / price)

    def scale_by_conviction(
        self,
        base_size: int,
        signal_strength: float,
        min_scale: float = 0.25,
        max_scale: float = 2.0,
    ) -> int:
        """
        Scale a base size by signal conviction.

        signal_strength ∈ [0, 1] is mapped linearly to [min_scale, max_scale].
        """
        signal_strength = max(0.0, min(1.0, signal_strength))
        scale = min_scale + signal_strength * (max_scale - min_scale)
        return max(0, int(base_size * scale))

    def portfolio_heat(
        self,
        positions: List[int],
        volatilities: List[float],
        prices: List[float],
    ) -> float:
        """
        Total portfolio risk as a fraction of capital.

        Computes the sum of (position * price * vol) / capital for each asset.
        """
        if not positions or self.capital <= 0:
            return 0.0
        n = min(len(positions), len(volatilities), len(prices))
        total_risk = sum(
            abs(positions[i]) * prices[i] * volatilities[i]
            for i in range(n)
        )
        return total_risk / self.capital

    def rebalance_orders(
        self,
        current: List[int],
        target: List[int],
        prices: List[float],
        min_trade: float = 100.0,
    ) -> List[int]:
        """
        Generate rebalancing orders (shares to trade) from current to target.

        Only generates orders where the trade value exceeds `min_trade`.

        Returns a list of share deltas (positive = buy, negative = sell).
        """
        n = min(len(current), len(target), len(prices))
        orders = []
        for i in range(n):
            delta = target[i] - current[i]
            trade_value = abs(delta) * prices[i] if prices[i] > 0 else 0.0
            if trade_value >= min_trade:
                orders.append(delta)
            else:
                orders.append(0)
        return orders
