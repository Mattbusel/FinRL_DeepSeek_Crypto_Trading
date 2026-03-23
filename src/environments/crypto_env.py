"""
Crypto trading RL environment with GBM + jump diffusion price simulation.
Pure Python implementation — no gym/numpy dependency.
"""
import math
import random


class CryptoEnv:
    """
    Crypto perpetual futures trading environment.

    State: [price_norm, position, pnl_norm, funding_accrued, volatility, volume_norm]
    Actions: 0 = flat, 1 = long, 2 = short
    """

    def __init__(
        self,
        symbol: str = "BTC",
        initial_price: float = 50000.0,
        n_steps: int = 24 * 7,
        funding_rate: float = 0.0001,
        leverage: float = 1.0,
        transaction_cost: float = 0.001,
        liquidation_threshold: float = 0.15,
    ):
        self.symbol = symbol
        self.initial_price = initial_price
        self.n_steps = n_steps
        self.funding_rate = funding_rate  # Per 8-hour period
        self.leverage = leverage
        self.transaction_cost = transaction_cost
        self.liquidation_threshold = liquidation_threshold

        # Runtime state
        self._price = initial_price
        self._position = 0  # -1, 0, 1
        self._entry_price = 0.0
        self._pnl = 0.0
        self._funding_accrued = 0.0
        self._step_count = 0
        self._done = False
        self._volatility = 0.02  # Rolling annualized vol estimate
        self._volume = 1.0
        self._rng = random.Random(42)
        self._initial_equity = 1.0

    def reset(self) -> list:
        """Reset environment and return initial state."""
        self._price = self.initial_price
        self._position = 0
        self._entry_price = 0.0
        self._pnl = 0.0
        self._funding_accrued = 0.0
        self._step_count = 0
        self._done = False
        self._volatility = 0.02
        self._volume = 1.0
        self._rng = random.Random(42)
        self._initial_equity = 1.0
        return self._get_state()

    def step(self, action: int, seed: int = 0) -> tuple:
        """
        Execute action in environment.
        action: 0=flat, 1=long, 2=short
        Returns: (state, reward, done)
        """
        if self._done:
            return self._get_state(), 0.0, True

        prev_pnl = self._pnl
        old_position = self._position
        new_price = self._simulate_price(seed + self._step_count)

        # Apply transaction cost on position change
        cost = 0.0
        new_position = action - 1  # Map {0,1,2} -> {-1,0,1}
        if new_position != self._position:
            cost = abs(new_position - self._position) * self.transaction_cost
            if new_position != 0:
                self._entry_price = new_price
            self._position = new_position

        # Update PnL
        if self._position != 0 and self._entry_price > 0:
            raw_return = (new_price - self._entry_price) / self._entry_price
            self._pnl = self._position * raw_return * self.leverage - cost
        else:
            self._pnl -= cost

        # Apply funding
        self._apply_funding()

        # Update price and metrics
        price_return = abs(new_price - self._price) / self._price
        self._volatility = 0.95 * self._volatility + 0.05 * price_return * math.sqrt(365 * 24)
        self._price = new_price
        self._volume = max(0.1, self._rng.gauss(1.0, 0.3))
        self._step_count += 1

        # Compute reward: change in PnL
        reward = self._pnl - prev_pnl

        # Check liquidation
        liquidated = self._check_liquidation()
        if liquidated:
            reward = -self.liquidation_threshold * self.leverage
            self._done = True

        # Check episode end
        if self._step_count >= self.n_steps:
            self._done = True

        return self._get_state(), reward, self._done

    def _get_state(self) -> list:
        """Build normalized state vector."""
        price_norm = (self._price - self.initial_price) / self.initial_price
        pnl_norm = max(-1.0, min(1.0, self._pnl))
        funding_norm = max(-1.0, min(1.0, self._funding_accrued * 100))
        vol_norm = min(1.0, self._volatility / 0.5)
        vol_norm_clipped = max(0.0, vol_norm)
        return [
            price_norm,
            float(self._position),  # -1, 0, or 1
            pnl_norm,
            funding_norm,
            vol_norm_clipped,
            max(0.0, min(1.0, self._volume)),
        ]

    def _apply_funding(self) -> None:
        """Apply periodic funding rate to open positions (every 8 hours = 480 steps)."""
        if self._step_count > 0 and self._step_count % 480 == 0:
            if self._position != 0:
                funding = self._position * self.funding_rate
                self._funding_accrued += funding
                self._pnl -= funding

    def _check_liquidation(self) -> bool:
        """Return True if position loss exceeds liquidation threshold."""
        if self._position == 0 or self._entry_price <= 0:
            return False
        raw_return = (self._price - self._entry_price) / self._entry_price
        loss = -self._position * raw_return * self.leverage
        return loss > self.liquidation_threshold

    def _simulate_price(self, seed: int) -> float:
        """
        GBM + jump diffusion price process.
        dS = mu*S*dt + sigma*S*dW + J*dN
        where J ~ Lognormal jump, dN ~ Poisson(lambda_j * dt)
        """
        rng = random.Random(seed)

        # GBM parameters
        mu = 0.0  # Zero drift for neutrality
        sigma = 0.02  # Hourly volatility
        dt = 1.0 / (365 * 24)  # 1 hour in years

        # GBM step
        z = rng.gauss(0.0, 1.0)
        gbm_return = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z

        # Jump component (Poisson with lambda=0.02 jumps per hour)
        lambda_j = 0.02
        jump_prob = lambda_j * dt
        jump_return = 0.0
        if rng.random() < jump_prob:
            jump_size = rng.gauss(0.0, 0.03)  # Jump magnitude ~3%
            jump_return = jump_size

        # New price
        total_return = gbm_return + jump_return
        new_price = self._price * math.exp(total_return)

        # Clamp to reasonable range
        new_price = max(self.initial_price * 0.01, min(self.initial_price * 100.0, new_price))
        return new_price

    def state_dim(self) -> int:
        """Return state dimension."""
        return 6

    def action_dim(self) -> int:
        """Return number of discrete actions."""
        return 3
