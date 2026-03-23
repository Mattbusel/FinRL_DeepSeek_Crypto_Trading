"""Simulate execution latency and market impact."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class LatencyProfile:
    mean_ms: float
    std_ms: float
    p99_ms: float
    exchange: str


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class LatencySimulator:
    """Simulate order execution latency and fill prices."""

    # Average daily volume used by the square-root slippage model.
    ADV: float = 1_000_000.0
    # Assumed annualized volatility for slippage model (20%).
    SIGMA: float = 0.20

    def __init__(self, profile: LatencyProfile, seed: int = 42) -> None:
        self.profile = profile
        self._rng = random.Random(seed)

        # Fit log-normal parameters from mean/std using moment matching:
        #   mean = exp(mu + sigma^2/2)
        #   std^2 = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
        mean = profile.mean_ms
        std = profile.std_ms
        if std <= 0 or mean <= 0:
            self._ln_mu = math.log(max(mean, 1e-9))
            self._ln_sigma = 1e-9
        else:
            # sigma_ln^2 = ln(1 + (std/mean)^2)
            sigma_ln_sq = math.log(1.0 + (std / mean) ** 2)
            self._ln_sigma = math.sqrt(sigma_ln_sq)
            self._ln_mu = math.log(mean) - sigma_ln_sq / 2.0

    def sample_latency(self) -> float:
        """Draw a latency sample from the log-normal distribution (ms)."""
        z = self._rng.gauss(0.0, 1.0)
        return math.exp(self._ln_mu + self._ln_sigma * z)

    def slippage_model(self, size: float, price: float, side: str) -> float:
        """
        Square-root market impact model.

        slippage = sigma * sqrt(size / ADV) * price

        Positive for buys (paid above mid), negative for sells.
        """
        impact = self.SIGMA * math.sqrt(abs(size) / self.ADV) * price
        return impact if side.upper() in ("BUY", "B") else -impact

    def simulate_fill(self, order_size: float, mid_price: float, side: str) -> dict:
        """
        Simulate filling one order.

        Returns a dict with fill_price, latency_ms, and slippage_bps.
        """
        latency_ms = self.sample_latency()
        slippage = self.slippage_model(order_size, mid_price, side)
        fill_price = mid_price + slippage
        slippage_bps = (slippage / mid_price) * 10_000 if mid_price != 0 else 0.0
        return {
            "fill_price": fill_price,
            "latency_ms": latency_ms,
            "slippage_bps": slippage_bps,
        }

    def batch_simulate(self, orders: List[dict]) -> List[dict]:
        """
        Simulate a list of orders.

        Each order dict must have keys: size, price, side.
        """
        return [
            self.simulate_fill(o["size"], o["price"], o["side"])
            for o in orders
        ]

    def latency_percentiles(self, n_samples: int = 10_000) -> dict:
        """Estimate p50/p75/p95/p99 latency by sampling."""
        samples = sorted(self.sample_latency() for _ in range(n_samples))
        n = len(samples)

        def percentile(p: float) -> float:
            idx = int(math.ceil(p / 100.0 * n)) - 1
            return samples[max(0, min(idx, n - 1))]

        return {
            "p50": percentile(50),
            "p75": percentile(75),
            "p95": percentile(95),
            "p99": percentile(99),
        }
