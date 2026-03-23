"""Extreme value theory and tail risk measures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TailRiskMetrics:
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    gpd_xi: float
    gpd_sigma: float
    expected_shortfall: float


# ---------------------------------------------------------------------------
# EVT Analyser
# ---------------------------------------------------------------------------

class EVTAnalyzer:
    """
    Extreme Value Theory analyser using the Peaks-Over-Threshold (POT) method
    with a Generalized Pareto Distribution (GPD) fit.
    """

    def __init__(self, threshold_quantile: float = 0.9) -> None:
        self.threshold_quantile = threshold_quantile

    # ------------------------------------------------------------------
    # GPD helpers
    # ------------------------------------------------------------------

    def gpd_cdf(self, x: float, xi: float, sigma: float, u: float) -> float:
        """
        GPD CDF evaluated at x (excess over threshold u):
          G(x; xi, sigma) = 1 - (1 + xi*(x-u)/sigma)^{-1/xi}   xi != 0
          G(x; sigma)     = 1 - exp(-(x-u)/sigma)               xi == 0
        """
        y = x - u
        if y < 0:
            return 0.0
        if abs(xi) < 1e-10:
            return 1.0 - math.exp(-y / sigma)
        base = 1.0 + xi * y / sigma
        if base <= 0:
            return 1.0
        return 1.0 - base ** (-1.0 / xi)

    def _gpd_log_likelihood(self, exceedances: List[float], xi: float, sigma: float) -> float:
        """Log-likelihood of GPD(xi, sigma) for a list of exceedances."""
        if sigma <= 0:
            return -math.inf
        n = len(exceedances)
        ll = -n * math.log(sigma)
        for x in exceedances:
            if abs(xi) < 1e-10:
                ll -= x / sigma
            else:
                val = 1.0 + xi * x / sigma
                if val <= 0:
                    return -math.inf
                ll -= (1.0 / xi + 1.0) * math.log(val)
        return ll

    def fit_gpd(self, losses: List[float]) -> Tuple[float, float]:
        """
        Fit GPD parameters (xi, sigma) to the exceedances above the threshold
        quantile via grid search over xi in [-0.5, 2.0], then refine sigma
        analytically for each xi.

        Returns (xi, sigma).
        """
        if not losses:
            return 0.0, 1.0

        losses_sorted = sorted(losses)
        n = len(losses_sorted)
        thresh_idx = int(self.threshold_quantile * n)
        u = losses_sorted[min(thresh_idx, n - 1)]
        exceedances = [x - u for x in losses_sorted if x > u]

        if len(exceedances) < 2:
            # Fall back to exponential (xi=0, sigma=mean exceedance)
            mean_exc = sum(losses_sorted) / len(losses_sorted)
            return 0.0, max(mean_exc, 1e-6)

        best_xi = 0.0
        best_sigma = max(sum(exceedances) / len(exceedances), 1e-6)
        best_ll = -math.inf

        xi_candidates = [-0.5 + i * 0.1 for i in range(26)]  # -0.5 to 2.0 in 0.1 steps
        for xi in xi_candidates:
            # For given xi, MLE sigma has closed form (when xi != 0):
            # sigma_hat = (1 + xi) * mean_exceedance (simplified PWM estimator)
            mean_exc = sum(exceedances) / len(exceedances)
            if abs(xi) < 1e-10:
                sigma = mean_exc
            else:
                sigma = max(mean_exc * (1.0 + xi), 1e-6)
            ll = self._gpd_log_likelihood(exceedances, xi, sigma)
            if ll > best_ll:
                best_ll = ll
                best_xi = xi
                best_sigma = sigma

        return best_xi, best_sigma

    # ------------------------------------------------------------------
    # VaR and CVaR via EVT
    # ------------------------------------------------------------------

    def var_evt(self, losses: List[float], p: float) -> float:
        """EVT-based VaR at confidence level p using fitted GPD."""
        if not losses:
            return 0.0
        losses_sorted = sorted(losses)
        n = len(losses_sorted)
        thresh_idx = int(self.threshold_quantile * n)
        u = losses_sorted[min(thresh_idx, n - 1)]
        xi, sigma = self.fit_gpd(losses)
        nu = n - thresh_idx  # number of exceedances

        if nu <= 0:
            return u

        # EVT VaR formula:
        # VaR_p = u + (sigma/xi) * ((n/nu * (1-p))^(-xi) - 1)   xi != 0
        # VaR_p = u - sigma * ln(n/nu * (1-p))                   xi == 0
        factor = (n / max(nu, 1)) * (1.0 - p)
        if factor <= 0:
            return losses_sorted[-1]

        if abs(xi) < 1e-10:
            return u - sigma * math.log(factor)
        else:
            return u + (sigma / xi) * (factor ** (-xi) - 1.0)

    def cvar_evt(self, losses: List[float], p: float) -> float:
        """
        Analytical CVaR (Expected Shortfall) from GPD parameters.

        CVaR_p = VaR_p / (1 - xi) + (sigma - xi * u) / (1 - xi)
        """
        if not losses:
            return 0.0
        xi, sigma = self.fit_gpd(losses)
        var = self.var_evt(losses, p)
        losses_sorted = sorted(losses)
        n = len(losses_sorted)
        u = losses_sorted[min(int(self.threshold_quantile * n), n - 1)]
        if abs(xi) >= 1.0:
            # CVaR undefined; fall back to empirical
            tail = [x for x in losses if x >= var]
            return sum(tail) / len(tail) if tail else var
        return var / (1.0 - xi) + (sigma - xi * u) / (1.0 - xi)

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def analyze(self, returns: List[float]) -> TailRiskMetrics:
        """
        Full tail risk analysis.

        Losses are defined as the negated returns (losses = -r for each r).
        """
        losses = [-r for r in returns]
        xi, sigma = self.fit_gpd(losses)
        var_95 = self.var_evt(losses, 0.95)
        var_99 = self.var_evt(losses, 0.99)
        cvar_95 = self.cvar_evt(losses, 0.95)
        cvar_99 = self.cvar_evt(losses, 0.99)
        # Expected shortfall = CVaR at 99%
        expected_shortfall = cvar_99

        return TailRiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            gpd_xi=xi,
            gpd_sigma=sigma,
            expected_shortfall=expected_shortfall,
        )

    def stress_test(self, returns: List[float], multiplier: float) -> TailRiskMetrics:
        """Scale tail losses by `multiplier` and re-analyse."""
        if not returns:
            return self.analyze(returns)
        # Find the threshold to identify tail observations
        losses = [-r for r in returns]
        losses_sorted = sorted(losses)
        n = len(losses_sorted)
        thresh_idx = int(self.threshold_quantile * n)
        u = losses_sorted[min(thresh_idx, n - 1)]
        # Scale exceedances by multiplier
        stressed_returns = []
        for r, loss in zip(returns, losses):
            if loss > u:
                stressed_returns.append(-(loss * multiplier))
            else:
                stressed_returns.append(r)
        return self.analyze(stressed_returns)
