"""Fama-French 3-factor model and factor risk attribution."""

from __future__ import annotations

import math
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# OLS via Gaussian elimination
# ---------------------------------------------------------------------------

def ols(y: List[float], X: List[List[float]]) -> List[float]:
    """
    Ordinary least squares: solve (X^T X) β = X^T y via Gaussian elimination.

    Parameters
    ----------
    y : response vector (n,)
    X : design matrix (n × k) — each inner list is one observation's features

    Returns
    -------
    β : coefficient vector (k,)
    """
    n = len(y)
    if n == 0 or not X:
        return []
    k = len(X[0])

    # Build X^T X (k×k) and X^T y (k,)
    XtX = [[0.0] * k for _ in range(k)]
    Xty = [0.0] * k

    for i in range(n):
        xi = X[i]
        yi = y[i]
        for r in range(k):
            Xty[r] += xi[r] * yi
            for c in range(k):
                XtX[r][c] += xi[r] * xi[c]

    # Augmented matrix [XtX | Xty]
    aug = [XtX[r][:] + [Xty[r]] for r in range(k)]

    # Forward elimination with partial pivoting
    for col in range(k):
        # Find pivot
        pivot_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, k):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                pivot_row = row
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-15:
            continue  # singular, leave as-is
        for row in range(col + 1, k):
            factor = aug[row][col] / pivot
            for j in range(col, k + 1):
                aug[row][j] -= factor * aug[col][j]

    # Back substitution
    beta = [0.0] * k
    for row in range(k - 1, -1, -1):
        s = aug[row][k]
        for j in range(row + 1, k):
            s -= aug[row][j] * beta[j]
        denom = aug[row][row]
        beta[row] = s / denom if abs(denom) > 1e-15 else 0.0

    return beta


# ---------------------------------------------------------------------------
# Fama-French 3-Factor Model
# ---------------------------------------------------------------------------

class FamaFrench3Factor:
    """Fama-French 3-factor model: Rp - Rf = α + β_m·MKT + β_s·SMB + β_h·HML + ε."""

    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate
        self.alpha_: float = 0.0
        self.beta_mkt_: float = 0.0
        self.beta_smb_: float = 0.0
        self.beta_hml_: float = 0.0
        self._fitted = False
        self._residuals: List[float] = []

    def fit(
        self,
        excess_returns: List[float],
        mkt_rf: List[float],
        smb: List[float],
        hml: List[float],
    ) -> Dict[str, object]:
        """
        OLS regression of excess returns on Fama-French factors.

        Returns dict with alpha, betas, r_squared, t_stats.
        """
        n = len(excess_returns)
        if n < 4:
            raise ValueError("Need at least 4 observations")

        # Design matrix: [1, mkt_rf, smb, hml]
        X = [[1.0, mkt_rf[i], smb[i], hml[i]] for i in range(n)]
        beta = ols(excess_returns, X)

        self.alpha_ = beta[0]
        self.beta_mkt_ = beta[1]
        self.beta_smb_ = beta[2]
        self.beta_hml_ = beta[3]
        self._fitted = True

        # Residuals
        y_pred = [
            beta[0] + beta[1] * mkt_rf[i] + beta[2] * smb[i] + beta[3] * hml[i]
            for i in range(n)
        ]
        self._residuals = [excess_returns[i] - y_pred[i] for i in range(n)]

        # R-squared
        r2 = self.r_squared_helper(excess_returns, y_pred)

        # T-statistics (simplified: t = β / se, se = σ_ε / sqrt(Σxi²))
        sse = sum(r ** 2 for r in self._residuals)
        sigma2 = sse / max(n - 4, 1)

        t_stats: Dict[str, float] = {}
        names = ['alpha', 'beta_mkt', 'beta_smb', 'beta_hml']
        for idx, name in enumerate(names):
            col_ss = sum(X[i][idx] ** 2 for i in range(n))
            se = math.sqrt(sigma2 / col_ss) if col_ss > 1e-15 else float('inf')
            t_stats[name] = beta[idx] / se if se > 1e-15 else 0.0

        return {
            'alpha': self.alpha_,
            'beta_mkt': self.beta_mkt_,
            'beta_smb': self.beta_smb_,
            'beta_hml': self.beta_hml_,
            'r_squared': r2,
            't_stats': t_stats,
        }

    @staticmethod
    def r_squared_helper(y: List[float], y_pred: List[float]) -> float:
        """Coefficient of determination R²."""
        n = len(y)
        if n == 0:
            return 0.0
        y_mean = sum(y) / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    def predict(self, mkt_rf: float, smb: float, hml: float) -> float:
        """Expected excess return given factor realizations."""
        return self.alpha_ + self.beta_mkt_ * mkt_rf + self.beta_smb_ * smb + self.beta_hml_ * hml

    def information_ratio(self, alpha: float, residuals: List[float]) -> float:
        """IR = alpha / tracking_error (annualized)."""
        te = self.tracking_error_helper(residuals)
        return alpha / te if te > 1e-15 else 0.0

    @staticmethod
    def tracking_error_helper(active_returns: List[float]) -> float:
        """Annualized tracking error = std(active_returns) * sqrt(252)."""
        n = len(active_returns)
        if n < 2:
            return 0.0
        mean = sum(active_returns) / n
        var = sum((r - mean) ** 2 for r in active_returns) / (n - 1)
        return math.sqrt(var * 252)

    def factor_exposure(self) -> Dict[str, float]:
        """Return fitted betas."""
        return {
            'alpha': self.alpha_,
            'beta_mkt': self.beta_mkt_,
            'beta_smb': self.beta_smb_,
            'beta_hml': self.beta_hml_,
        }


# ---------------------------------------------------------------------------
# General Factor Risk Model
# ---------------------------------------------------------------------------

class FactorRiskModel:
    """Multi-factor risk model with OLS-based attribution."""

    def __init__(self, factors: List[str]):
        self.factors = factors
        self._betas: List[float] = []

    def compute_factor_returns(
        self,
        portfolio_returns: List[float],
        factor_returns: List[List[float]],
    ) -> Dict[str, object]:
        """
        OLS attribution of portfolio returns to factors.

        Parameters
        ----------
        portfolio_returns : (n,)
        factor_returns    : (k × n) — factor_returns[k][t]

        Returns
        -------
        Dict with betas, r_squared, residuals
        """
        n = len(portfolio_returns)
        k = len(factor_returns)
        if n == 0 or k == 0:
            return {'betas': [], 'r_squared': 0.0, 'residuals': []}

        # Design matrix with intercept: [1, f1, f2, ..., fk]
        X = [[1.0] + [factor_returns[fi][t] for fi in range(k)] for t in range(n)]
        beta = ols(portfolio_returns, X)
        self._betas = beta[1:] if len(beta) > 1 else []

        y_pred = [
            sum(beta[j] * X[t][j] for j in range(len(beta)))
            for t in range(n)
        ]
        residuals = [portfolio_returns[t] - y_pred[t] for t in range(n)]
        r2 = FamaFrench3Factor.r_squared_helper(portfolio_returns, y_pred)

        result: Dict[str, object] = {
            'intercept': beta[0] if beta else 0.0,
            'betas': {self.factors[i]: self._betas[i] for i in range(min(len(self.factors), len(self._betas)))},
            'r_squared': r2,
            'residuals': residuals,
        }
        return result

    def systematic_risk(self, betas: List[float], factor_cov: List[List[float]]) -> float:
        """Systematic variance: β^T Σ_F β."""
        k = len(betas)
        if k == 0 or not factor_cov:
            return 0.0
        # β^T Σ β = Σ_i Σ_j β_i Σ_ij β_j
        result = 0.0
        for i in range(k):
            for j in range(k):
                if i < len(factor_cov) and j < len(factor_cov[i]):
                    result += betas[i] * factor_cov[i][j] * betas[j]
        return result

    def idiosyncratic_risk(self, total_var: float, systematic_var: float) -> float:
        """Idiosyncratic variance = total - systematic."""
        return max(total_var - systematic_var, 0.0)

    def r_squared(self, y: List[float], y_pred: List[float]) -> float:
        """R-squared between actuals and predictions."""
        return FamaFrench3Factor.r_squared_helper(y, y_pred)

    def active_return(self, portfolio_ret: float, benchmark_ret: float) -> float:
        """Active return = portfolio - benchmark."""
        return portfolio_ret - benchmark_ret

    def tracking_error(self, active_returns: List[float]) -> float:
        """Annualized tracking error = std(active_returns) * sqrt(252)."""
        return FamaFrench3Factor.tracking_error_helper(active_returns)
