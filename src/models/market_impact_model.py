"""Market impact models: Almgren-Chriss, square-root, linear."""
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class LinearImpactModel:
    """Linear market impact: Delta_P = eta * x / V where x=trade size, V=ADV."""
    eta: float = 0.1      # permanent impact coefficient
    temp_eta: float = 0.1  # temporary impact coefficient
    adv: float = 1_000_000  # average daily volume

    def permanent_impact(self, trade_size: float) -> float:
        """Price impact that persists after trade."""
        return self.eta * trade_size / self.adv

    def temporary_impact(self, trade_rate: float) -> float:
        """Transient impact during trade: proportional to rate."""
        return self.temp_eta * trade_rate / self.adv

    def implementation_shortfall(self, trade_schedule: List[float]) -> float:
        total = sum(trade_schedule)
        cost = 0.0
        for x in trade_schedule:
            cost += x * (self.permanent_impact(total) + self.temporary_impact(x))
        return cost


@dataclass
class SquareRootImpactModel:
    """Square-root market impact: Delta_P = sigma * sqrt(x/V)."""
    sigma: float = 0.02    # daily volatility
    adv: float = 1_000_000
    alpha: float = 0.5     # typically 0.5 (square root)
    k_temp: float = 0.1    # temporary impact scaling

    def price_impact(self, trade_size: float) -> float:
        """Price impact in units of daily sigma."""
        return self.sigma * (abs(trade_size) / self.adv) ** self.alpha * math.copysign(1, trade_size)

    def cost_of_execution(self, total_shares: float, T_days: float, n_slices: int) -> float:
        """Total execution cost for uniform slicing."""
        slice_size = total_shares / n_slices
        # Permanent: 0.5 * total * impact(total)
        perm_cost = 0.5 * total_shares * abs(self.price_impact(total_shares))
        # Temporary: n * slice * temp_impact(slice)
        temp_impact = self.k_temp * self.sigma * (abs(slice_size) / self.adv) ** self.alpha
        temp_cost = n_slices * abs(slice_size) * temp_impact
        return perm_cost + temp_cost

    def optimal_execution_rate(self, total_shares: float, T_days: float, risk_aversion: float) -> float:
        """Almgren-Chriss optimal rate (uniform schedule for zero risk aversion)."""
        if risk_aversion == 0:
            return total_shares / T_days
        kappa = math.sqrt(risk_aversion * self.sigma**2 / self.k_temp)
        return total_shares * kappa / math.sinh(kappa * T_days)


@dataclass
class AlmgrenChrissModel:
    """Full Almgren-Chriss execution optimization model."""
    sigma: float           # asset volatility
    eta: float             # temporary impact (linear)
    gamma: float           # permanent impact (linear)
    rho: float             # risk aversion parameter (annualized)

    def kappa(self) -> float:
        """Characteristic time scale."""
        return math.sqrt(self.rho * self.sigma**2 / self.eta)

    def optimal_trajectory(self, X: float, T: float, n: int) -> List[float]:
        """Optimal liquidation trajectory: x_j shares remaining at time j*tau."""
        kappa = self.kappa()
        tau = T / n
        trajectory = []
        for j in range(n + 1):
            t = j * tau
            # x(t) = X * sinh(kappa*(T-t)) / sinh(kappa*T)
            denom = math.sinh(kappa * T) if abs(kappa * T) > 1e-10 else T
            numerator = math.sinh(kappa * (T - t)) if abs(kappa * (T - t)) > 1e-10 else (T - t)
            x_t = X * numerator / denom if abs(denom) > 1e-10 else X * (1 - t/T)
            trajectory.append(max(x_t, 0.0))
        return trajectory

    def optimal_trades(self, X: float, T: float, n: int) -> List[float]:
        """Trade sizes at each period (n_j = x_j - x_{j+1})."""
        traj = self.optimal_trajectory(X, T, n)
        return [traj[i] - traj[i+1] for i in range(n)]

    def expected_cost(self, X: float, T: float) -> float:
        """E[Implementation Shortfall] = 0.5*gamma*X^2 + eta*X^2/T * (kappa*T/sinh(kappa*T))"""
        kappa = self.kappa()
        perm = 0.5 * self.gamma * X**2
        temp_ratio = kappa * T / math.sinh(kappa * T) if abs(math.sinh(kappa * T)) > 1e-10 else 1.0
        temp = self.eta * X**2 / T * temp_ratio
        return perm + temp

    def variance_of_cost(self, X: float, T: float) -> float:
        """Var[Cost] = sigma^2 * integral of x(t)^2 dt (simplified)."""
        kappa = self.kappa()
        # Approximate: sum of squared remaining positions
        n = 50
        traj = self.optimal_trajectory(X, T, n)
        dt = T / n
        return self.sigma**2 * sum(x**2 for x in traj) * dt

    def efficient_frontier(self, X: float, T: float, n_points: int = 20) -> List[Tuple[float, float]]:
        """(E[cost], Std[cost]) pairs for varying risk aversion."""
        frontier = []
        for i in range(n_points):
            rho_test = 1e-6 * (10 ** (i * 6 / n_points))
            model = AlmgrenChrissModel(self.sigma, self.eta, self.gamma, rho_test)
            e_cost = model.expected_cost(X, T)
            std_cost = math.sqrt(max(model.variance_of_cost(X, T), 0))
            frontier.append((e_cost, std_cost))
        return frontier
