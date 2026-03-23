"""Multi-factor risk model."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import math


@dataclass
class Factor:
    name: str
    returns: List[float]
    loading: float = 1.0


@dataclass
class FactorExposure:
    asset: str
    exposures: Dict[str, float]
    idiosyncratic_var: float = 0.0


class FactorModel:
    def __init__(self, factors: List[Factor]):
        self.factors = factors

    def _ols(self, y: List[float], xs: List[List[float]]) -> List[float]:
        """Simple OLS via normal equations (no external deps)."""
        n = len(y)
        k = len(xs)
        if n < k + 1:
            return [0.0] * k

        # Build X matrix (n x k) and y vector
        # Use each xs[j] as a column
        # beta = (X'X)^{-1} X'y  — use k=1 shortcut or iterate
        # For simplicity, do univariate OLS per factor (ignoring collinearity)
        betas = []
        for j in range(k):
            x = xs[j]
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            den = sum((xi - mean_x) ** 2 for xi in x)
            beta = num / den if abs(den) > 1e-12 else 0.0
            betas.append(beta)
        return betas

    def fit(self, asset_returns: Dict[str, List[float]]) -> Dict[str, FactorExposure]:
        """OLS regression of each asset on factors."""
        exposures: Dict[str, FactorExposure] = {}

        factor_xs = [f.returns for f in self.factors]
        factor_names = [f.name for f in self.factors]

        for asset, ret in asset_returns.items():
            n = len(ret)
            # Align lengths
            min_len = min(n, *(len(f) for f in factor_xs)) if factor_xs else n
            y = ret[:min_len]
            xs = [x[:min_len] for x in factor_xs]

            betas = self._ols(y, xs) if xs else []

            # Compute residuals
            fitted = [
                sum(betas[j] * xs[j][i] for j in range(len(betas)))
                for i in range(min_len)
            ]
            residuals = [y[i] - fitted[i] for i in range(min_len)]
            idio_var = (
                sum(r ** 2 for r in residuals) / max(min_len - len(betas) - 1, 1)
                if residuals
                else 0.0
            )

            exposures[asset] = FactorExposure(
                asset=asset,
                exposures=dict(zip(factor_names, betas)),
                idiosyncratic_var=idio_var,
            )

        return exposures

    def factor_covariance_matrix(self) -> List[List[float]]:
        """Sample covariance of factor returns."""
        k = len(self.factors)
        if k == 0:
            return []

        returns_list = [f.returns for f in self.factors]
        min_len = min(len(r) for r in returns_list)
        trimmed = [r[:min_len] for r in returns_list]
        n = min_len

        if n < 2:
            return [[0.0] * k for _ in range(k)]

        means = [sum(r) / n for r in trimmed]

        cov = [[0.0] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                cov[i][j] = sum(
                    (trimmed[i][t] - means[i]) * (trimmed[j][t] - means[j])
                    for t in range(n)
                ) / (n - 1)
        return cov

    def asset_variance(self, exposure: FactorExposure) -> float:
        """Total variance = factor variance + idiosyncratic variance."""
        cov = self.factor_covariance_matrix()
        factor_names = [f.name for f in self.factors]
        beta = [exposure.exposures.get(name, 0.0) for name in factor_names]
        k = len(beta)

        factor_var = 0.0
        for i in range(k):
            for j in range(k):
                factor_var += beta[i] * beta[j] * (cov[i][j] if cov else 0.0)

        return factor_var + exposure.idiosyncratic_var

    def portfolio_factor_exposure(
        self,
        weights: Dict[str, float],
        exposures: Dict[str, FactorExposure],
    ) -> Dict[str, float]:
        """Weighted sum of factor exposures across portfolio."""
        factor_names = [f.name for f in self.factors]
        port_exp: Dict[str, float] = {name: 0.0 for name in factor_names}

        for asset, w in weights.items():
            if asset in exposures:
                for fname, bval in exposures[asset].exposures.items():
                    port_exp[fname] = port_exp.get(fname, 0.0) + w * bval

        return port_exp

    def risk_attribution(
        self,
        weights: Dict[str, float],
        exposures: Dict[str, FactorExposure],
    ) -> Dict[str, float]:
        """Factor contribution to portfolio variance."""
        cov = self.factor_covariance_matrix()
        factor_names = [f.name for f in self.factors]
        port_exp = self.portfolio_factor_exposure(weights, exposures)
        beta = [port_exp.get(name, 0.0) for name in factor_names]
        k = len(beta)

        attribution: Dict[str, float] = {}
        total_factor_var = 0.0
        for i in range(k):
            for j in range(k):
                total_factor_var += beta[i] * beta[j] * (cov[i][j] if cov else 0.0)

        for i, name in enumerate(factor_names):
            contrib = sum(beta[i] * beta[j] * (cov[i][j] if cov else 0.0) for j in range(k))
            attribution[name] = contrib

        return attribution


def create_macro_factors(returns_data: Dict) -> List[Factor]:
    """Create 4 placeholder macro factors: momentum, value, size, quality."""
    def _get(key: str) -> List[float]:
        data = returns_data.get(key, [])
        if isinstance(data, list):
            return [float(v) for v in data]
        return []

    momentum = Factor(name="momentum", returns=_get("momentum") or [0.0], loading=1.0)
    value = Factor(name="value", returns=_get("value") or [0.0], loading=1.0)
    size = Factor(name="size", returns=_get("size") or [0.0], loading=1.0)
    quality = Factor(name="quality", returns=_get("quality") or [0.0], loading=1.0)
    return [momentum, value, size, quality]
