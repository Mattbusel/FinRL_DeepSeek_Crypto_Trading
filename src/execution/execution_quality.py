"""
Execution quality metrics and Transaction Cost Analysis (TCA).

Provides implementation shortfall, arrival slippage, and venue comparison
analytics for evaluating trade execution quality.
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ExecutionRecord:
    symbol: str
    side: str          # "buy" or "sell"
    quantity: float
    decision_price: float
    fill_price: float
    timestamp: float
    venue: str


@dataclass
class TCAMetrics:
    implementation_shortfall_bps: float
    arrival_slippage_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    spread_cost_bps: float


class ExecutionQualityAnalyzer:
    """Analyzes execution quality and performs TCA."""

    def __init__(self):
        self._records: List[ExecutionRecord] = []

    def add_record(self, record: ExecutionRecord):
        """Add an execution record for analysis."""
        self._records.append(record)

    def implementation_shortfall(
        self, record: ExecutionRecord, benchmark_price: float
    ) -> float:
        """
        Implementation shortfall in basis points.

        For buys: (fill - decision) / decision * 10000
        For sells: (decision - fill) / decision * 10000
        """
        if benchmark_price <= 0 or record.decision_price <= 0:
            return 0.0
        if record.side.lower() == "buy":
            return (record.fill_price - record.decision_price) / record.decision_price * 10000.0
        else:
            return (record.decision_price - record.fill_price) / record.decision_price * 10000.0

    def arrival_slippage(
        self, record: ExecutionRecord, arrival_price: float
    ) -> float:
        """
        Arrival slippage in basis points.

        For buys: (fill - arrival) / arrival * 10000
        For sells: (arrival - fill) / arrival * 10000
        """
        if arrival_price <= 0:
            return 0.0
        if record.side.lower() == "buy":
            return (record.fill_price - arrival_price) / arrival_price * 10000.0
        else:
            return (arrival_price - record.fill_price) / arrival_price * 10000.0

    def compute_tca(
        self,
        record: ExecutionRecord,
        benchmark: float,
        arrival: float,
        bid: float,
        ask: float,
    ) -> TCAMetrics:
        """Compute comprehensive TCA metrics for a single execution."""
        is_bps = self.implementation_shortfall(record, benchmark)
        arrival_slip = self.arrival_slippage(record, arrival)

        # Market impact: difference between fill and arrival
        if arrival > 0:
            if record.side.lower() == "buy":
                market_impact = (record.fill_price - arrival) / arrival * 10000.0
            else:
                market_impact = (arrival - record.fill_price) / arrival * 10000.0
        else:
            market_impact = 0.0

        # Timing cost: IS - market_impact (captures delay from decision to arrival)
        timing_cost = is_bps - market_impact

        # Spread cost: half-spread * 10000
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else record.fill_price
        spread = ask - bid if (ask > bid) else 0.0
        spread_cost = (spread / 2.0) / mid * 10000.0 if mid > 0 else 0.0

        return TCAMetrics(
            implementation_shortfall_bps=is_bps,
            arrival_slippage_bps=arrival_slip,
            market_impact_bps=market_impact,
            timing_cost_bps=timing_cost,
            spread_cost_bps=spread_cost,
        )

    def venue_comparison(self) -> Dict[str, Dict]:
        """Average IS and average slippage per venue."""
        venue_data: Dict[str, List[float]] = {}
        for record in self._records:
            v = record.venue
            if v not in venue_data:
                venue_data[v] = []
            is_val = self.implementation_shortfall(record, record.decision_price)
            venue_data[v].append(is_val)

        result = {}
        for venue, is_values in venue_data.items():
            avg_is = sum(is_values) / len(is_values) if is_values else 0.0
            result[venue] = {
                "avg_implementation_shortfall_bps": avg_is,
                "count": len(is_values),
            }
        return result

    def quality_score(self, record: ExecutionRecord) -> float:
        """
        Normalized quality score 0-100 based on IS percentile.

        Lower IS is better (higher score). Score = (1 - percentile) * 100.
        """
        if not self._records:
            return 50.0
        all_is = [
            self.implementation_shortfall(r, r.decision_price)
            for r in self._records
        ]
        this_is = self.implementation_shortfall(record, record.decision_price)
        n_worse = sum(1 for x in all_is if x > this_is)
        percentile = (len(all_is) - n_worse) / len(all_is)
        return (1.0 - percentile) * 100.0

    def summary_stats(self) -> Dict:
        """Mean, std, and p95 of implementation shortfall across all records."""
        if not self._records:
            return {"mean": 0.0, "std": 0.0, "p95": 0.0, "count": 0}

        all_is = sorted(
            self.implementation_shortfall(r, r.decision_price)
            for r in self._records
        )
        n = len(all_is)
        mean_is = sum(all_is) / n
        std_is = statistics.pstdev(all_is) if n > 1 else 0.0
        p95_idx = min(int(0.95 * n), n - 1)
        p95_is = all_is[p95_idx]

        return {
            "mean": mean_is,
            "std": std_is,
            "p95": p95_is,
            "count": n,
        }
