"""Portfolio hedging: delta/gamma hedging, options overlay, hedge ratios."""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bsm_greeks(S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True) -> dict:
    if T <= 0 or sigma <= 0:
        return {'delta': 1.0 if (is_call and S > K) else 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    nd1 = norm_cdf(d1)
    nd2 = norm_cdf(d2)
    phi_d1 = math.exp(-0.5*d1**2) / math.sqrt(2*math.pi)

    delta = nd1 if is_call else nd1 - 1
    gamma = phi_d1 / (S * sigma * math.sqrt(T))
    vega = S * math.sqrt(T) * phi_d1 / 100  # per 1% vol change
    theta = (-S * phi_d1 * sigma / (2 * math.sqrt(T))
             - r * K * math.exp(-r*T) * (nd2 if is_call else (1-nd2))) / 365
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}


@dataclass
class EquityPosition:
    ticker: str
    shares: float
    entry_price: float
    current_price: float
    beta: float = 1.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def delta(self) -> float:
        return self.shares  # equity delta = shares


@dataclass
class OptionPosition:
    ticker: str
    strike: float
    expiry_days: int
    is_call: bool
    contracts: float   # positive = long, negative = short
    implied_vol: float
    underlying_price: float
    risk_free_rate: float = 0.05

    def T(self) -> float:
        return max(self.expiry_days / 365, 1e-6)

    def greeks(self) -> dict:
        return bsm_greeks(self.underlying_price, self.strike, self.T(),
                          self.risk_free_rate, self.implied_vol, self.is_call)

    @property
    def position_delta(self) -> float:
        return self.contracts * 100 * self.greeks()['delta']

    @property
    def position_gamma(self) -> float:
        return self.contracts * 100 * self.greeks()['gamma']

    @property
    def position_vega(self) -> float:
        return self.contracts * 100 * self.greeks()['vega']


class PortfolioHedger:
    """Compute hedge ratios and overlay strategies for a mixed portfolio."""

    def __init__(self):
        self.equity_positions: List[EquityPosition] = []
        self.option_positions: List[OptionPosition] = []

    def add_equity(self, pos: EquityPosition):
        self.equity_positions.append(pos)

    def add_option(self, pos: OptionPosition):
        self.option_positions.append(pos)

    def portfolio_greeks(self) -> Dict[str, float]:
        """Aggregate delta, gamma, vega across all positions."""
        delta = sum(p.shares for p in self.equity_positions)
        gamma = 0.0
        vega = 0.0
        for op in self.option_positions:
            delta += op.position_delta
            gamma += op.position_gamma
            vega += op.position_vega
        return {'delta': delta, 'gamma': gamma, 'vega': vega}

    def delta_hedge_ratio(self) -> float:
        """Shares to short/buy to neutralize portfolio delta."""
        g = self.portfolio_greeks()
        return -g['delta']

    def gamma_hedge_option(self, S: float, K: float, T_days: int, sigma: float,
                            r: float = 0.05) -> float:
        """Number of ATM call contracts needed to neutralize gamma."""
        g = self.portfolio_greeks()
        if abs(g['gamma']) < 1e-10:
            return 0.0
        option_gamma = bsm_greeks(S, K, T_days/365, r, sigma)['gamma'] * 100
        return -g['gamma'] / option_gamma if abs(option_gamma) > 1e-10 else 0.0

    def beta_adjusted_hedge(self, spy_price: float) -> float:
        """SPY shares to short for market-neutral beta hedge."""
        total_beta_exposure = sum(p.beta * p.market_value for p in self.equity_positions)
        return -total_beta_exposure / spy_price

    def var_95_estimate(self, portfolio_vol: float, time_horizon_days: int = 1) -> float:
        """Parametric 95% VaR."""
        total_value = sum(p.market_value for p in self.equity_positions)
        daily_vol = portfolio_vol / math.sqrt(252)
        return total_value * daily_vol * 1.645 * math.sqrt(time_horizon_days)

    def hedge_effectiveness(self, pre_hedge_vol: float, post_hedge_vol: float) -> float:
        """Reduction in variance as fraction of original."""
        return max(1.0 - (post_hedge_vol / pre_hedge_vol)**2, 0.0)

    def summary(self) -> str:
        g = self.portfolio_greeks()
        total_mv = sum(p.market_value for p in self.equity_positions)
        lines = [
            f"Portfolio Market Value: ${total_mv:,.0f}",
            f"Net Delta: {g['delta']:.1f} shares",
            f"Net Gamma: {g['gamma']:.4f}",
            f"Net Vega: {g['vega']:.2f} per 1% vol",
            f"Delta Hedge: {'Buy' if g['delta'] < 0 else 'Short'} {abs(g['delta']):.0f} shares",
        ]
        return "\n".join(lines)
