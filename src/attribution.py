"""Brinson-Hood-Beebower (BHB) Performance Attribution for crypto portfolios.

Decomposes active portfolio return into:

- **Allocation effect** — did we overweight/underweight the right sectors?
- **Selection effect** — did we pick better assets within each position?
- **Interaction effect** — joint effect of over/under-weighting and selection.

Formulas (per asset)::

    allocation  = (w_p - w_b) * (r_b - R_b)
    selection   = w_b  * (r_p - r_b)
    interaction = (w_p - w_b) * (r_p - r_b)
    total       = allocation + selection + interaction

where:
    w_p = portfolio weight
    w_b = benchmark weight
    r_p = portfolio return for the asset
    r_b = benchmark return for the asset
    R_b = benchmark total return (weighted average)

Example::

    from src.attribution import AttributionInput, BhbAttribution

    inp = AttributionInput(
        period="2024-Q1",
        portfolio_weights={"BTC": 0.6, "ETH": 0.4},
        benchmark_weights={"BTC": 0.5, "ETH": 0.5},
        portfolio_returns={"BTC": 0.20, "ETH": 0.15},
        benchmark_returns={"BTC": 0.18, "ETH": 0.12},
    )
    report = BhbAttribution().compute(inp)
    print(report.total_active_return)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Input / Output data structures
# ---------------------------------------------------------------------------


@dataclass
class AttributionInput:
    """Inputs required for a single-period BHB attribution analysis.

    Attributes:
        period: Human-readable period label (e.g. ``"2024-Q1"``).
        portfolio_weights: Portfolio weights keyed by asset symbol.
            Need not sum to 1; they are normalised internally.
        benchmark_weights: Benchmark weights keyed by asset symbol.
        portfolio_returns: Realised returns per asset for the portfolio.
        benchmark_returns: Realised returns per asset for the benchmark.
    """

    period: str
    portfolio_weights: Dict[str, float]
    benchmark_weights: Dict[str, float]
    portfolio_returns: Dict[str, float]
    benchmark_returns: Dict[str, float]

    def __post_init__(self) -> None:
        if not self.portfolio_weights:
            raise ValueError("portfolio_weights must not be empty")
        if not self.benchmark_weights:
            raise ValueError("benchmark_weights must not be empty")


@dataclass
class AssetAttribution:
    """Per-asset BHB attribution components.

    Attributes:
        asset: Asset symbol.
        allocation: Allocation effect for this asset.
        selection: Selection effect for this asset.
        interaction: Interaction effect for this asset.
        total: Sum of the three effects.
    """

    asset: str
    allocation: float
    selection: float
    interaction: float
    total: float


@dataclass
class AttributionReport:
    """Aggregate BHB attribution report for a full portfolio.

    Attributes:
        period: Period label copied from :class:`AttributionInput`.
        allocation_effect: Portfolio-level sum of per-asset allocation effects.
        selection_effect: Portfolio-level sum of per-asset selection effects.
        interaction_effect: Portfolio-level sum of per-asset interaction effects.
        total_active_return: Algebraic sum of all three effects.
        by_asset: Detailed per-asset attribution breakdown.
    """

    period: str
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_active_return: float
    by_asset: List[AssetAttribution] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BHB Attribution engine
# ---------------------------------------------------------------------------


def _normalise(weights: Dict[str, float]) -> Dict[str, float]:
    """Return a copy of *weights* scaled to sum to 1.0."""
    total = sum(weights.values())
    if total == 0:
        return {k: 0.0 for k in weights}
    return {k: v / total for k, v in weights.items()}


class BhbAttribution:
    """Stateless Brinson-Hood-Beebower attribution calculator."""

    def compute(self, inp: AttributionInput) -> AttributionReport:
        """Compute BHB attribution for a single period.

        Args:
            inp: :class:`AttributionInput` containing weights and returns.

        Returns:
            :class:`AttributionReport` with aggregate and per-asset effects.
        """
        w_p = _normalise(inp.portfolio_weights)
        w_b = _normalise(inp.benchmark_weights)

        # All assets that appear in either portfolio or benchmark
        all_assets = set(w_p) | set(w_b)

        # Benchmark total return R_b = sum(w_b_i * r_b_i)
        R_b = sum(
            w_b.get(a, 0.0) * inp.benchmark_returns.get(a, 0.0)
            for a in all_assets
        )

        by_asset: List[AssetAttribution] = []
        total_alloc = 0.0
        total_sel = 0.0
        total_inter = 0.0

        for asset in sorted(all_assets):
            wp_i = w_p.get(asset, 0.0)
            wb_i = w_b.get(asset, 0.0)
            rp_i = inp.portfolio_returns.get(asset, 0.0)
            rb_i = inp.benchmark_returns.get(asset, 0.0)

            allocation  = (wp_i - wb_i) * (rb_i - R_b)
            selection   = wb_i * (rp_i - rb_i)
            interaction = (wp_i - wb_i) * (rp_i - rb_i)
            total       = allocation + selection + interaction

            by_asset.append(
                AssetAttribution(
                    asset=asset,
                    allocation=allocation,
                    selection=selection,
                    interaction=interaction,
                    total=total,
                )
            )

            total_alloc += allocation
            total_sel   += selection
            total_inter += interaction

        total_active = total_alloc + total_sel + total_inter

        return AttributionReport(
            period=inp.period,
            allocation_effect=total_alloc,
            selection_effect=total_sel,
            interaction_effect=total_inter,
            total_active_return=total_active,
            by_asset=by_asset,
        )

    def portfolio_return(self, inp: AttributionInput) -> float:
        """Compute the weighted portfolio return for the period.

        Args:
            inp: :class:`AttributionInput`.

        Returns:
            Weighted-average portfolio return.
        """
        w_p = _normalise(inp.portfolio_weights)
        return sum(
            w_p.get(a, 0.0) * inp.portfolio_returns.get(a, 0.0)
            for a in inp.portfolio_weights
        )

    def benchmark_return(self, inp: AttributionInput) -> float:
        """Compute the weighted benchmark return for the period.

        Args:
            inp: :class:`AttributionInput`.

        Returns:
            Weighted-average benchmark return.
        """
        w_b = _normalise(inp.benchmark_weights)
        return sum(
            w_b.get(a, 0.0) * inp.benchmark_returns.get(a, 0.0)
            for a in inp.benchmark_weights
        )
