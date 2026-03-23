"""Futures trading RL environment with contract roll simulation."""

import math
import random
from typing import List, Optional, Tuple


class FuturesEnv:
    """
    A simple futures trading RL environment with carry and roll mechanics.

    State: [S/S0, T_remaining, position, pnl_normalized, basis]
    Action: float in [-1, 1] representing the target position.
    """

    # Annualized carry / cost parameters
    _RISK_FREE_RATE: float = 0.05   # cost of carry proxy (r)
    _STORAGE_COST: float = 0.01     # commodity storage cost (y_s)
    _CONVENIENCE_YIELD: float = 0.02  # convenience yield (y)

    def __init__(
        self,
        S0: float = 100.0,
        cost_of_carry: float = 0.05,
        T_months: int = 3,
        n_steps: int = 63,
        transaction_cost: float = 0.001,
        allow_roll: bool = True,
        sigma: float = 0.20,
    ) -> None:
        self.S0 = S0
        self.cost_of_carry = cost_of_carry
        self.T_months = T_months
        self.T_years = T_months / 12.0
        self.n_steps = n_steps
        self.transaction_cost = transaction_cost
        self.allow_roll = allow_roll
        self.sigma = sigma
        self.dt = self.T_years / n_steps

        # State variables (initialised in reset)
        self._S: float = S0
        self._F: float = 0.0
        self._position: float = 0.0
        self._pnl: float = 0.0
        self._step: int = 0
        self._roll_count: int = 0

    def reset(self) -> List[float]:
        """Reset environment to initial state. Returns initial state vector."""
        self._S = self.S0
        self._F = self._theoretical_futures_price(self._S, self.T_years)
        self._position = 0.0
        self._pnl = 0.0
        self._step = 0
        self._roll_count = 0
        return self._get_state()

    def step(
        self,
        action: float,
        seed: int = 0,
    ) -> Tuple[List[float], float, bool]:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : float
            Target position in [-1, 1].  Positive = long, negative = short.
        seed : int
            RNG seed for this step (for reproducibility in testing).

        Returns
        -------
        state : List[float]
        reward : float
        done : bool
        """
        rng = random.Random(seed + self._step)
        action = max(-1.0, min(1.0, float(action)))

        # --- Simulate next spot price (GBM) ---
        z = rng.gauss(0.0, 1.0)
        drift = (self.cost_of_carry - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * math.sqrt(self.dt) * z
        self._S = self._S * math.exp(drift + diffusion)

        # --- Time remaining ---
        self._step += 1
        t_remaining = self.T_years - self._step * self.dt
        t_remaining = max(t_remaining, 0.0)

        # --- Theoretical futures price ---
        new_F = self._theoretical_futures_price(self._S, t_remaining)
        basis = self._basis(self._S, new_F)

        # --- P&L from existing position based on futures price change ---
        old_F = self._F
        futures_pnl = self._position * (new_F - old_F)
        self._F = new_F

        # --- Rebalance to target position ---
        trade = action - self._position
        tc_cost = abs(trade) * self._S * self.transaction_cost
        self._position = action

        # --- Handle contract roll at expiry ---
        rolled = False
        if t_remaining <= 0 and self.allow_roll:
            self._roll_position()
            rolled = True
            self._roll_count += 1

        step_pnl = futures_pnl - tc_cost
        self._pnl += step_pnl

        reward = step_pnl / self.S0  # normalised reward

        done = (t_remaining <= 0 and not self.allow_roll) or (
            self._step >= self.n_steps
        )

        return self._get_state(), reward, done

    def _theoretical_futures_price(self, S: float, T: float) -> float:
        """Cost-of-carry model: F = S * exp((r + y_s - y) * T)."""
        net_carry = self._RISK_FREE_RATE + self._STORAGE_COST - self._CONVENIENCE_YIELD
        return S * math.exp(net_carry * T)

    def _basis(self, S: float, F: float) -> float:
        """Basis = F - S (contango > 0, backwardation < 0)."""
        return F - S

    def _roll_position(self) -> None:
        """
        Simulate a contract roll: close current contract, open next one.
        Re-initialises time to full T_years and recalculates futures price.
        Transaction cost applied to roll.
        """
        roll_cost = abs(self._position) * self._S * self.transaction_cost * 2.0
        self._pnl -= roll_cost
        # Reset time horizon for the new contract
        self._step = 0
        self._F = self._theoretical_futures_price(self._S, self.T_years)

    def _get_state(self) -> List[float]:
        """Build the 5-element state vector."""
        t_remaining = max(self.T_years - self._step * self.dt, 0.0)
        basis = self._basis(self._S, self._F)
        return [
            self._S / self.S0,                   # normalised spot
            t_remaining / self.T_years,           # fraction of time remaining
            self._position,                       # current position (-1 to 1)
            self._pnl / self.S0,                  # normalised cumulative PnL
            basis / self.S0,                      # normalised basis
        ]

    def state_dim(self) -> int:
        """Dimension of the state space."""
        return 5

    def action_dim(self) -> int:
        """Dimension of the action space (scalar continuous action)."""
        return 1
