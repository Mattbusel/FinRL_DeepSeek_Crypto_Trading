"""SHAP-based explainability and LARSA signal dashboard for trading decisions.

This module provides two complementary explainability systems:

1. :class:`TradeExplainer` -- SHAP-based attribution for individual DRL agent
   decisions (gradient or KernelSHAP).

2. :class:`LarsaExplainer` -- High-level per-signal explanation that combines
   DeepSeek sentiment drivers, Alpha101 factor contributions, ensemble votes,
   and regime context into an HTML report and JSON audit trail.

Example (LarsaExplainer)::

    from explainability import LarsaExplainer

    explainer = LarsaExplainer()
    explanation = explainer.explain_signal(
        signal_data={"direction": "buy", "confidence": 0.82, "regime": "bull"},
        news_items=["BTC ETF approved", "Fed pauses rate hikes"],
        factor_values={"alpha_001": 0.45, "alpha_012": -0.12},
        agent_outputs=[
            {"agent_type": "D3QN", "agent_id": 0, "action": "buy", "q_value": 1.23},
            {"agent_type": "DDQN", "agent_id": 1, "action": "buy", "q_value": 0.98},
            {"agent_type": "TwinD3QN", "agent_id": 2, "action": "hold", "q_value": 0.55},
        ],
    )
    html = explainer.generate_html_report(explanation)
    with open("trade_report.html", "w") as f:
        f.write(html)

    audit = explainer.generate_json_audit(explanation)
    with open("audit.jsonl", "a") as f:
        f.write(audit + "\\n")
"""

Wraps any LARSA DRL agent with SHAP analysis to surface the top features
driving each trade decision.  Supports both gradient-based explanations
(via Q-value gradients for DQN agents) and model-agnostic KernelSHAP as a
fallback.

Dependencies::

    pip install shap

Example::

    from explainability import TradeExplainer
    explainer = TradeExplainer(agent=my_agent, feature_names=FEATURE_NAMES)
    report = explainer.explain_decision(state=current_state, action=1)
    print(report.to_text())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from exceptions import ModelError
from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FeatureImportance:
    """SHAP attribution for a single feature.

    Attributes:
        feature_name: Human-readable name of the feature.
        shap_value: Raw SHAP value (positive = bullish, negative = bearish).
        direction: ``"bullish"`` if SHAP value >= 0, else ``"bearish"``.
        rank: Importance rank (1 = most influential).
    """

    feature_name: str
    shap_value: float
    direction: str
    rank: int

    @property
    def abs_shap(self) -> float:
        """Absolute SHAP value used for ranking."""
        return abs(self.shap_value)


@dataclass
class ExplainabilityReport:
    """Human-readable explanation of a single trade decision.

    Attributes:
        action: Agent action integer (0 = hold, 1 = buy, 2 = sell).
        action_label: String label corresponding to *action*.
        top_features: Top-5 :class:`FeatureImportance` objects sorted by
            absolute SHAP value descending.
        baseline_value: Mean Q-value over the background distribution.
        predicted_value: Q-value for the chosen action on the explained state.
        symbol: Asset symbol if provided, else empty string.
    """

    action: int
    action_label: str
    top_features: list[FeatureImportance]
    baseline_value: float = 0.0
    predicted_value: float = 0.0
    symbol: str = ""

    # action integer -> label
    _LABELS: dict[int, str] = field(
        default_factory=lambda: {0: "HOLD", 1: "BUY", 2: "SELL"},
        repr=False,
        compare=False,
    )

    def to_text(self) -> str:
        """Format the report as a human-readable explanation string.

        Returns:
            Multi-line explanation, e.g.::

                Traded BUY because:
                  sentiment_score  +0.34  (bullish)
                  momentum         +0.21  (bullish)
                  volatility       -0.15  (bearish)
        """
        header = f"Traded {self.action_label}"
        if self.symbol:
            header = f"Traded {self.action_label} {self.symbol}"
        lines = [f"{header} because:"]
        for fi in self.top_features:
            sign = "+" if fi.shap_value >= 0 else ""
            lines.append(
                f"  {fi.feature_name:<25} {sign}{fi.shap_value:.4f}  ({fi.direction})"
            )
        lines.append(
            f"  [baseline Q={self.baseline_value:.4f}, "
            f"predicted Q={self.predicted_value:.4f}]"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TradeExplainer
# ---------------------------------------------------------------------------

_DEFAULT_FEATURE_NAMES = [
    "position",
    "btc_price",
    "signal_0",
    "signal_1",
    "signal_2",
    "signal_3",
    "signal_4",
    "signal_5",
    "signal_6",
    "signal_7",
    "cash_ratio",
    "btc_ratio",
]


class TradeExplainer:
    """Wraps a DRL agent decision with SHAP-based feature attribution.

    Uses gradient-based attribution (``torch.autograd.grad`` on Q-values) for
    DQN-compatible agents when PyTorch is available, and falls back to
    KernelSHAP otherwise.  A background dataset of ``n_background`` random
    states is kept internally for the KernelSHAP baseline.

    Args:
        agent: A LARSA DRL agent instance.  Must have an ``act`` attribute
            that behaves like a :class:`torch.nn.Module`.
        feature_names: List of feature names for the state vector.  Defaults
            to a 12-element generic list matching the standard LARSA state dim.
        n_background: Number of zero-state background samples for KernelSHAP
            (default 50).
        top_k: Number of top features to return in each report (default 5).
        symbol: Asset symbol attached to reports (default empty).

    Example::

        explainer = TradeExplainer(agent=agent, feature_names=MY_FEATURES)
        report = explainer.explain_decision(state=state_tensor, action=1)
        print(report.to_text())
    """

    _ACTION_LABELS: dict[int, str] = {0: "HOLD", 1: "BUY", 2: "SELL"}

    def __init__(
        self,
        agent: Any,
        feature_names: list[str] | None = None,
        n_background: int = 50,
        top_k: int = 5,
        symbol: str = "",
    ) -> None:
        self.agent = agent
        self.feature_names = feature_names or _DEFAULT_FEATURE_NAMES
        self.n_background = n_background
        self.top_k = top_k
        self.symbol = symbol
        self._state_dim = len(self.feature_names)

        log.info(
            "trade_explainer.init",
            state_dim=self._state_dim,
            top_k=top_k,
            n_background=n_background,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain_decision(
        self,
        state: Any,
        action: int,
    ) -> ExplainabilityReport:
        """Explain why the agent chose *action* given *state*.

        Tries gradient-based attribution first; falls back to KernelSHAP if
        PyTorch is unavailable or the network is non-differentiable.

        Args:
            state: Current state, either a ``torch.Tensor`` of shape
                ``(1, state_dim)`` or ``(state_dim,)``, or a 1-D
                ``numpy.ndarray``.
            action: The action integer the agent chose.

        Returns:
            :class:`ExplainabilityReport` with the top-*k* features.

        Raises:
            ModelError: If attribution fails entirely.
        """
        state_np = self._to_numpy(state)

        try:
            shap_values, baseline_q, predicted_q = self._gradient_attribution(
                state_np, action
            )
            log.debug("trade_explainer.method", method="gradient")
        except Exception as exc:
            log.warning(
                "trade_explainer.gradient_failed",
                error=str(exc),
                fallback="kernel_shap",
            )
            try:
                shap_values, baseline_q, predicted_q = self._kernel_shap_attribution(
                    state_np, action
                )
                log.debug("trade_explainer.method", method="kernel_shap")
            except Exception as exc2:
                raise ModelError(
                    "Both gradient and KernelSHAP attribution failed.", cause=exc2
                ) from exc2

        top_features = self._rank_features(shap_values)
        report = ExplainabilityReport(
            action=action,
            action_label=self._ACTION_LABELS.get(action, str(action)),
            top_features=top_features,
            baseline_value=float(baseline_q),
            predicted_value=float(predicted_q),
            symbol=self.symbol,
        )
        log.info(
            "trade_explainer.explained",
            action=self._ACTION_LABELS.get(action, str(action)),
            top_feature=top_features[0].feature_name if top_features else "n/a",
        )
        return report

    # ------------------------------------------------------------------
    # Attribution methods
    # ------------------------------------------------------------------

    def _gradient_attribution(
        self,
        state_np: npt.NDArray[np.float64],
        action: int,
    ) -> tuple[npt.NDArray[np.float64], float, float]:
        """Compute attribution via input gradients of the Q-value.

        ``attribution[i] = state[i] * dQ/dstate[i]`` (input * gradient,
        the simplest integrated-gradients approximation with a zero baseline).

        Args:
            state_np: 1-D numpy state array.
            action: Chosen action index.

        Returns:
            Tuple of (shap_values, baseline_q, predicted_q).

        Raises:
            Exception: If PyTorch or the model forward pass fails.
        """
        import torch

        act_net = self.agent.act
        state_t = torch.tensor(
            state_np, dtype=torch.float32, requires_grad=True
        ).unsqueeze(0)

        q_vals = act_net(state_t)  # (1, action_dim)
        q_for_action = q_vals[0, action]
        q_for_action.backward()

        grads = state_t.grad.detach().cpu().numpy().squeeze()  # (state_dim,)
        attributions = state_np * grads  # element-wise

        baseline_q = float(q_vals.detach().cpu().numpy().mean())
        predicted_q = float(q_for_action.detach().cpu().item())

        return attributions, baseline_q, predicted_q

    def _kernel_shap_attribution(
        self,
        state_np: npt.NDArray[np.float64],
        action: int,
    ) -> tuple[npt.NDArray[np.float64], float, float]:
        """Compute attribution via KernelSHAP from the ``shap`` library.

        Uses a background of zero-valued states and a function that returns
        the Q-value for *action* for each row in the input matrix.

        Args:
            state_np: 1-D numpy state array.
            action: Chosen action index.

        Returns:
            Tuple of (shap_values_1d, baseline_q, predicted_q).

        Raises:
            ModelError: If ``shap`` is not installed.
        """
        try:
            import shap  # type: ignore[import]
        except ImportError as exc:
            raise ModelError(
                "shap is required for KernelSHAP attribution.  "
                "Run: pip install shap",
                cause=exc,
            ) from exc

        background = np.zeros((self.n_background, len(state_np)))

        def _predict(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            results = []
            for row in X:
                q = self._get_q_value(row)
                results.append(float(q[action]) if hasattr(q, "__len__") else float(q))
            return np.array(results)

        explainer = shap.KernelExplainer(_predict, background)
        shap_vals = explainer.shap_values(
            state_np.reshape(1, -1), nsamples=100, silent=True
        )
        # shap_vals is (1, state_dim) when output is scalar
        sv = np.asarray(shap_vals).squeeze()

        baseline_q = float(explainer.expected_value)
        predicted_q = float(_predict(state_np.reshape(1, -1))[0])

        return sv, baseline_q, predicted_q

    def _get_q_value(self, state_np: npt.NDArray[np.float64]) -> Any:
        """Run a forward pass on the agent's act network for a single state.

        Args:
            state_np: 1-D numpy state array.

        Returns:
            Q-values as a numpy array.
        """
        import torch

        with torch.no_grad():
            state_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
            q_vals = self.agent.act(state_t)
            return q_vals.cpu().numpy().squeeze()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(state: Any) -> npt.NDArray[np.float64]:
        """Convert *state* to a 1-D float64 numpy array.

        Args:
            state: Torch tensor, numpy array, or Python sequence.

        Returns:
            1-D numpy float64 array.
        """
        try:
            import torch

            if isinstance(state, torch.Tensor):
                return state.detach().cpu().numpy().squeeze().astype(np.float64)
        except ImportError:
            pass
        arr = np.asarray(state, dtype=np.float64)
        return arr.squeeze()

    def _rank_features(
        self, shap_values: npt.NDArray[np.float64]
    ) -> list[FeatureImportance]:
        """Sort features by absolute SHAP value and return the top *k*.

        Args:
            shap_values: 1-D array of SHAP values, one per feature.

        Returns:
            List of up to *k* :class:`FeatureImportance` objects.
        """
        sv = np.asarray(shap_values, dtype=np.float64).squeeze()
        if sv.ndim == 0:
            sv = sv.reshape(1)

        n = min(len(sv), len(self.feature_names))
        indices = np.argsort(np.abs(sv[:n]))[::-1][: self.top_k]

        features = []
        for rank, idx in enumerate(indices, start=1):
            val = float(sv[idx])
            features.append(
                FeatureImportance(
                    feature_name=self.feature_names[idx],
                    shap_value=val,
                    direction="bullish" if val >= 0 else "bearish",
                    rank=rank,
                )
            )
        return features


# ---------------------------------------------------------------------------
# LarsaExplainer — high-level per-signal explanation dashboard
# ---------------------------------------------------------------------------

import json as _json
import uuid as _uuid
from dataclasses import dataclass as _dataclass, field as _field
from typing import List as _List, Dict as _Dict, Any as _Any, Optional as _Optional
from datetime import datetime as _datetime


@_dataclass
class FactorContribution:
    """Contribution of a single Alpha101 factor to the LARSA signal.

    Attributes:
        factor_name: Short identifier (e.g. ``"alpha_001"``).
        factor_value: Raw computed factor value.
        contribution_pct: Percentage contribution to the final signal
            (positive = pro-signal, negative = counter-signal).
        direction: ``"positive"``, ``"negative"``, or ``"neutral"``.
        description: Human-readable plain-English meaning of this factor.
    """

    factor_name: str
    factor_value: float
    contribution_pct: float
    direction: str
    description: str


@_dataclass
class AgentVote:
    """How a single RL agent in the ensemble voted.

    Attributes:
        agent_type: Architecture label (``"D3QN"``, ``"DDQN"``, ``"TwinD3QN"``).
        agent_id: Zero-based index within the ensemble.
        action: The action the agent chose (``"buy"``, ``"sell"``, ``"hold"``).
        q_value: Q-value for the chosen action.
        confidence: Normalised confidence in [0, 1].
        reasoning: Short generated sentence explaining the vote.
    """

    agent_type: str
    agent_id: int
    action: str
    q_value: float
    confidence: float
    reasoning: str


@_dataclass
class TradeExplanation:
    """Complete explanation for a single LARSA trade signal.

    Attributes:
        signal_id: Unique UUID for this explanation.
        timestamp: ISO-8601 UTC timestamp.
        symbol: Trading pair (e.g. ``"BTC/USDT"``).
        final_decision: The ensemble's final decision (``"buy"`` / ``"sell"`` / ``"hold"``).
        confidence: Overall ensemble confidence in [0, 1].
        sentiment_score: DeepSeek sentiment score (1 = very negative, 5 = very positive).
        key_news_headlines: Top news headlines that influenced the sentiment.
        sentiment_keywords: Market-moving keywords extracted from headlines.
        top_factors: List of :class:`FactorContribution` objects sorted by |contribution|.
        regime_label: Detected market regime (``"bull"``, ``"bear"``, ``"sideways"``,
            ``"volatile"``).
        regime_confidence: Regime detector confidence in [0, 1].
        agent_votes: List of :class:`AgentVote` objects from the ensemble.
        consensus: ``"unanimous"``, ``"majority"``, or ``"split"``.
        current_drawdown_pct: Current portfolio drawdown as a percentage.
        days_since_last_trade: Calendar days since the last executed trade.
        similar_signals: List of dicts describing historical signals with similar
            characteristics (pattern matching).
    """

    signal_id: str
    timestamp: str
    symbol: str
    final_decision: str
    confidence: float
    sentiment_score: float
    key_news_headlines: _List[str]
    sentiment_keywords: _List[str]
    top_factors: _List[FactorContribution]
    regime_label: str
    regime_confidence: float
    agent_votes: _List[AgentVote]
    consensus: str
    current_drawdown_pct: float
    days_since_last_trade: int
    similar_signals: _List[dict]


class LarsaExplainer:
    """Generate per-trade HTML reports and JSON audit trails for LARSA signals.

    For every trade signal, this explainer combines:
    - DeepSeek sentiment: which news items drove the sentiment score.
    - Alpha101 factors: which factors had the most influence and why.
    - Ensemble votes: how each RL agent voted and its Q-value.
    - Regime context: what regime was detected and with what confidence.
    - Historical matches: similar past signal patterns and their outcomes.

    Args:
        model_dir: Directory where saved models live (default ``"./models"``).
            Used only for future extensions that load factor weights.

    Example::

        explainer = LarsaExplainer()
        exp = explainer.explain_signal(
            signal_data={"direction": "buy", "confidence": 0.82, "regime": "bull",
                         "sentiment_score": 4.0, "regime_confidence": 0.78,
                         "current_drawdown_pct": -1.2, "days_since_last_trade": 3},
            news_items=["BTC ETF approved", "Institutional buying surge"],
            factor_values={"alpha_001": 0.45, "alpha_012": -0.12, "alpha_028": 0.30},
            agent_outputs=[
                {"agent_type": "D3QN", "agent_id": 0, "action": "buy", "q_value": 1.23},
            ],
        )
        html = explainer.generate_html_report(exp)
    """

    # Alpha101 factor descriptions (subset of most common)
    _FACTOR_DESCRIPTIONS: _Dict[str, str] = {
        "alpha_001": "Returns-based price reversal signal: high recent return predicts mean-reversion",
        "alpha_002": "Log-price delta correlation with volume delta: measures price-volume divergence",
        "alpha_003": "Open-to-close return correlation across assets: inter-asset momentum link",
        "alpha_004": "Low-price rank over recent window: captures short-term oversold conditions",
        "alpha_005": "Open vs VWAP spread normalised by daily range: intraday price location",
        "alpha_006": "Open-to-open price autocorrelation: persistence in opening price gaps",
        "alpha_007": "Volume-weighted absolute daily return: strength of directional moves",
        "alpha_008": "Sum of open-return products minus lagged variant: momentum cross-term",
        "alpha_009": "Conditional delta: if close trend positive, use close delta, else sum",
        "alpha_010": "Max of rank(delta) or rank(correlation): combined momentum signal",
        "alpha_011": "VWAP-close spread times volume: measures buying/selling pressure",
        "alpha_012": "Volume sign times close-open delta: direction-weighted volume flow",
        "alpha_013": "Negative covariance rank of returns and volume: smart-money signal",
        "alpha_014": "Correlation of returns with volume rank: institutional footprint",
        "alpha_015": "Sum of max-return correlations: cross-asset momentum fingerprint",
        "alpha_016": "Negative covariance of high, volume rank: selling pressure gauge",
        "alpha_017": "Combined rank of close delta and volume correlation: trend + volume",
        "alpha_018": "Correlation of close and open over lookback: gap persistence signal",
        "alpha_019": "Conditional sign-sum return: trend-following with sign filter",
        "alpha_020": "Open vs high/low/close averages: intraday positioning signal",
        "alpha_021": "Conditional mean comparison over rolling windows: regime transition",
        "alpha_022": "Delta of cross-asset correlation: changing correlation dynamics",
        "alpha_023": "Delta of high price over half-window: short-term high breakout",
        "alpha_024": "Long/short on close delta relative to historical mean",
        "alpha_025": "ADV-scaled return rank with close ratio: volume-adjusted momentum",
        "alpha_026": "Max correlation of time-series: autocorrelation peak detector",
        "alpha_027": "Conditional buy signal when volume-return rank exceeds threshold",
        "alpha_028": "Correlation of ADV and close-high/close-low: liquidity-adjusted range",
        "alpha_029": "Sum of close ranks weighted by returns: value-weighted momentum",
        "alpha_030": "Direction-weighted signed return difference: reversal signal",
        "sentiment_score": "DeepSeek LLM sentiment: news tone from very negative (1) to very positive (5)",
        "risk_score": "DeepSeek LLM risk assessment: systemic risk level from low (5) to high (1)",
    }

    # Market-moving keywords with their typical signal direction
    _BULLISH_KEYWORDS = [
        "approved", "etf", "institutional", "adoption", "upgrade", "rally",
        "surge", "breakout", "bullish", "partnership", "launch", "record",
        "inflow", "buy", "accumulation", "halving", "upgrade", "fed pause",
    ]
    _BEARISH_KEYWORDS = [
        "ban", "hack", "regulation", "crackdown", "sell-off", "crash", "fraud",
        "investigation", "bearish", "outflow", "liquidation", "recession",
        "rate hike", "bankruptcy", "exploit", "sanctions", "warning",
    ]

    def __init__(self, model_dir: str = "./models") -> None:
        self.model_dir = model_dir
        log.info("larsa_explainer.init", model_dir=model_dir)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain_signal(
        self,
        signal_data: dict,
        news_items: list,
        factor_values: dict,
        agent_outputs: list,
    ) -> TradeExplanation:
        """Generate a complete explanation for a LARSA trade signal.

        Args:
            signal_data: Dict with keys: ``direction``, ``confidence``,
                ``regime``, ``sentiment_score`` (optional, default 3.0),
                ``regime_confidence`` (optional, default 0.5),
                ``current_drawdown_pct`` (optional, default 0.0),
                ``days_since_last_trade`` (optional, default 0),
                ``symbol`` (optional, default ``"BTC/USDT"``).
            news_items: List of news headline strings.
            factor_values: Dict mapping Alpha101 factor names to float values.
            agent_outputs: List of dicts with keys ``agent_type``, ``agent_id``,
                ``action``, ``q_value``.  ``confidence`` is optional.

        Returns:
            A fully populated :class:`TradeExplanation`.
        """
        signal_id = str(_uuid.uuid4())
        timestamp = _datetime.utcnow().isoformat()

        symbol = signal_data.get("symbol", "BTC/USDT")
        direction = signal_data.get("direction", "hold")
        confidence = float(signal_data.get("confidence", 0.5))
        regime_label = signal_data.get("regime", "sideways")
        regime_confidence = float(signal_data.get("regime_confidence", 0.5))
        sentiment_score = float(signal_data.get("sentiment_score", 3.0))
        current_drawdown_pct = float(signal_data.get("current_drawdown_pct", 0.0))
        days_since_last_trade = int(signal_data.get("days_since_last_trade", 0))

        # Sentiment component
        keywords = self._identify_keywords(news_items)

        # Factor component
        top_factors = self._build_factor_contributions(factor_values)

        # Agent votes
        agent_votes = self._build_agent_votes(agent_outputs)
        consensus = self._format_consensus(agent_votes)

        # Historical similarity (synthetic unless history is wired in)
        similar_signals = self._find_similar_signals(
            direction, confidence, regime_label, sentiment_score
        )

        explanation = TradeExplanation(
            signal_id=signal_id,
            timestamp=timestamp,
            symbol=symbol,
            final_decision=direction,
            confidence=confidence,
            sentiment_score=sentiment_score,
            key_news_headlines=list(news_items)[:5],
            sentiment_keywords=keywords,
            top_factors=top_factors,
            regime_label=regime_label,
            regime_confidence=regime_confidence,
            agent_votes=agent_votes,
            consensus=consensus,
            current_drawdown_pct=current_drawdown_pct,
            days_since_last_trade=days_since_last_trade,
            similar_signals=similar_signals,
        )

        log.info(
            "larsa_explainer.explained",
            signal_id=signal_id,
            direction=direction,
            confidence=confidence,
            consensus=consensus,
            n_factors=len(top_factors),
            n_votes=len(agent_votes),
        )
        return explanation

    def generate_html_report(self, explanation: TradeExplanation) -> str:
        """Generate a self-contained HTML report for *explanation*.

        The report includes:
        - Summary banner (decision, confidence, symbol, timestamp)
        - Sentiment section (score, news headlines, keywords)
        - Factor attribution table (top Alpha101 factors)
        - Ensemble vote breakdown (per-agent votes + Q-values)
        - Regime context (label, confidence)
        - Risk context (drawdown, days since last trade)
        - Similar historical signals

        Args:
            explanation: A :class:`TradeExplanation` from :meth:`explain_signal`.

        Returns:
            A complete HTML string with embedded CSS.  No external dependencies.
        """
        direction_color = {
            "buy": "#27ae60",
            "sell": "#e74c3c",
            "hold": "#f39c12",
        }.get(explanation.final_decision, "#7f8c8d")

        sentiment_label = {
            1: "Very Negative",
            2: "Negative",
            3: "Neutral",
            4: "Positive",
            5: "Very Positive",
        }.get(int(round(explanation.sentiment_score)), "Unknown")

        sentiment_color = "#e74c3c" if explanation.sentiment_score < 2.5 else (
            "#f39c12" if explanation.sentiment_score < 3.5 else "#27ae60"
        )

        # Build factor rows
        factor_rows = ""
        for fc in explanation.top_factors:
            bar_color = "#27ae60" if fc.direction == "positive" else (
                "#e74c3c" if fc.direction == "negative" else "#7f8c8d"
            )
            bar_width = min(100, abs(fc.contribution_pct))
            factor_rows += f"""
            <tr>
                <td><code>{fc.factor_name}</code></td>
                <td>{fc.factor_value:+.4f}</td>
                <td>
                    <div style="background:{bar_color};width:{bar_width:.0f}%;height:14px;
                                border-radius:3px;display:inline-block;"></div>
                    <span style="margin-left:6px;">{fc.contribution_pct:+.1f}%</span>
                </td>
                <td><span style="color:{bar_color};">{fc.direction}</span></td>
                <td style="font-size:0.85em;color:#555;">{fc.description}</td>
            </tr>"""

        # Build agent vote rows
        vote_rows = ""
        for vote in explanation.agent_votes:
            vote_color = {
                "buy": "#27ae60", "sell": "#e74c3c", "hold": "#f39c12"
            }.get(vote.action, "#7f8c8d")
            vote_rows += f"""
            <tr>
                <td><code>{vote.agent_type}</code> #{vote.agent_id}</td>
                <td><strong style="color:{vote_color};">{vote.action.upper()}</strong></td>
                <td>{vote.q_value:+.4f}</td>
                <td>{vote.confidence:.1%}</td>
                <td style="font-size:0.85em;color:#555;">{vote.reasoning}</td>
            </tr>"""

        # Build headlines list
        headlines_html = "".join(
            f"<li>{h}</li>" for h in explanation.key_news_headlines
        )

        # Build keywords pills
        keywords_html = "".join(
            f'<span style="background:#2c3e50;color:#ecf0f1;padding:3px 8px;'
            f'border-radius:12px;margin:2px;display:inline-block;">{kw}</span>'
            for kw in explanation.sentiment_keywords
        )

        # Build similar signals table
        similar_rows = ""
        for s in explanation.similar_signals:
            outcome_color = "#27ae60" if float(s.get("outcome_return_pct", 0)) > 0 else "#e74c3c"
            similar_rows += f"""
            <tr>
                <td>{s.get('date', 'N/A')}</td>
                <td>{s.get('direction', 'N/A')}</td>
                <td>{s.get('regime', 'N/A')}</td>
                <td>{float(s.get('confidence', 0)):.0%}</td>
                <td style="color:{outcome_color};">
                    {float(s.get('outcome_return_pct', 0)):+.2f}%
                </td>
            </tr>"""

        consensus_color = {
            "unanimous": "#27ae60",
            "majority": "#f39c12",
            "split": "#e74c3c",
        }.get(explanation.consensus, "#7f8c8d")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LARSA Trade Explanation – {explanation.signal_id}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #2c3e50; }}
  .container {{ max-width: 1100px; margin: 32px auto; padding: 0 16px; }}
  .banner {{ background: {direction_color}; color: white; border-radius: 10px; padding: 28px 32px;
             margin-bottom: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
  .banner h1 {{ font-size: 2.2em; letter-spacing: 1px; }}
  .banner .meta {{ opacity: 0.85; margin-top: 8px; font-size: 0.95em; }}
  .banner .confidence {{ font-size: 1.5em; font-weight: bold; margin-top: 10px; }}
  .card {{ background: white; border-radius: 10px; padding: 24px 28px; margin-bottom: 20px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
  .card h2 {{ font-size: 1.2em; color: #2c3e50; border-bottom: 2px solid #ecf0f1;
              padding-bottom: 10px; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th {{ background: #ecf0f1; padding: 10px 12px; text-align: left; font-weight: 600; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #f0f2f5; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  .pill {{ display: inline-block; background: #3498db; color: white; padding: 3px 10px;
           border-radius: 12px; font-size: 0.82em; margin: 2px; }}
  .metric-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .metric {{ background: #f8f9fa; border-radius: 8px; padding: 14px 18px; flex: 1; min-width: 140px; }}
  .metric .label {{ font-size: 0.78em; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }}
  .metric .value {{ font-size: 1.6em; font-weight: bold; margin-top: 4px; }}
  ul {{ padding-left: 20px; }}
  li {{ margin-bottom: 6px; line-height: 1.5; }}
  code {{ background: #f0f2f5; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }}
  .footer {{ text-align: center; color: #95a5a6; font-size: 0.8em; margin-top: 32px; padding-bottom: 24px; }}
</style>
</head>
<body>
<div class="container">

  <!-- Banner -->
  <div class="banner">
    <h1>{explanation.final_decision.upper()} {explanation.symbol}</h1>
    <div class="meta">Signal ID: {explanation.signal_id} &nbsp;|&nbsp; {explanation.timestamp} UTC</div>
    <div class="confidence">Confidence: {explanation.confidence:.1%}
      &nbsp;&nbsp; Consensus: <span style="font-size:0.85em;
        background:rgba(255,255,255,0.2);border-radius:6px;padding:2px 10px;">
        {explanation.consensus.upper()}
      </span>
    </div>
  </div>

  <!-- Key Metrics -->
  <div class="card">
    <h2>Key Metrics</h2>
    <div class="metric-row">
      <div class="metric">
        <div class="label">Sentiment</div>
        <div class="value" style="color:{sentiment_color};">
          {explanation.sentiment_score:.1f}/5
        </div>
        <div style="font-size:0.82em;color:#7f8c8d;">{sentiment_label}</div>
      </div>
      <div class="metric">
        <div class="label">Regime</div>
        <div class="value">{explanation.regime_label.title()}</div>
        <div style="font-size:0.82em;color:#7f8c8d;">
          {explanation.regime_confidence:.0%} confidence
        </div>
      </div>
      <div class="metric">
        <div class="label">Drawdown</div>
        <div class="value" style="color:{'#e74c3c' if explanation.current_drawdown_pct < -1 else '#27ae60'};">
          {explanation.current_drawdown_pct:+.2f}%
        </div>
      </div>
      <div class="metric">
        <div class="label">Days Since Trade</div>
        <div class="value">{explanation.days_since_last_trade}</div>
      </div>
      <div class="metric">
        <div class="label">Ensemble</div>
        <div class="value" style="color:{consensus_color};">
          {explanation.consensus.title()}
        </div>
        <div style="font-size:0.82em;color:#7f8c8d;">{len(explanation.agent_votes)} agents</div>
      </div>
    </div>
  </div>

  <!-- Sentiment -->
  <div class="card">
    <h2>DeepSeek Sentiment Analysis</h2>
    <p style="margin-bottom:12px;">
      <strong>Score: <span style="color:{sentiment_color};">
        {explanation.sentiment_score:.1f} / 5.0</span></strong>
      &nbsp;&nbsp;({sentiment_label})
    </p>
    <p style="margin-bottom:10px;"><strong>Key News Headlines:</strong></p>
    <ul>{headlines_html}</ul>
    <p style="margin-top:14px;margin-bottom:8px;"><strong>Market-Moving Keywords:</strong></p>
    <div>{keywords_html}</div>
  </div>

  <!-- Alpha101 Factors -->
  <div class="card">
    <h2>Alpha101 Factor Attribution</h2>
    <table>
      <thead>
        <tr>
          <th>Factor</th>
          <th>Value</th>
          <th>Contribution</th>
          <th>Direction</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        {factor_rows}
      </tbody>
    </table>
  </div>

  <!-- Ensemble Votes -->
  <div class="card">
    <h2>Ensemble Agent Votes</h2>
    <table>
      <thead>
        <tr>
          <th>Agent</th>
          <th>Vote</th>
          <th>Q-Value</th>
          <th>Confidence</th>
          <th>Reasoning</th>
        </tr>
      </thead>
      <tbody>
        {vote_rows}
      </tbody>
    </table>
  </div>

  <!-- Similar Historical Signals -->
  <div class="card">
    <h2>Similar Historical Signals</h2>
    {"<p style='color:#7f8c8d;'>No similar signals found in history.</p>" if not explanation.similar_signals else f"""
    <table>
      <thead>
        <tr><th>Date</th><th>Direction</th><th>Regime</th><th>Confidence</th><th>Outcome</th></tr>
      </thead>
      <tbody>{similar_rows}</tbody>
    </table>"""}
  </div>

  <div class="footer">
    Generated by LARSA Explainability Dashboard &nbsp;|&nbsp;
    Signal {explanation.signal_id} &nbsp;|&nbsp;
    LARSA is experimental. This is not financial advice.
  </div>

</div>
</body>
</html>"""
        return html

    def generate_json_audit(self, explanation: TradeExplanation) -> str:
        """Generate a single-line JSON audit trail entry.

        Suitable for appending to a ``.jsonl`` file for compliance records.

        Args:
            explanation: A :class:`TradeExplanation` from :meth:`explain_signal`.

        Returns:
            A compact JSON string (no trailing newline).
        """
        audit = {
            "signal_id": explanation.signal_id,
            "timestamp": explanation.timestamp,
            "symbol": explanation.symbol,
            "final_decision": explanation.final_decision,
            "confidence": explanation.confidence,
            "sentiment_score": explanation.sentiment_score,
            "key_news_headlines": explanation.key_news_headlines,
            "sentiment_keywords": explanation.sentiment_keywords,
            "top_factors": [
                {
                    "factor_name": fc.factor_name,
                    "factor_value": fc.factor_value,
                    "contribution_pct": fc.contribution_pct,
                    "direction": fc.direction,
                    "description": fc.description,
                }
                for fc in explanation.top_factors
            ],
            "regime_label": explanation.regime_label,
            "regime_confidence": explanation.regime_confidence,
            "agent_votes": [
                {
                    "agent_type": v.agent_type,
                    "agent_id": v.agent_id,
                    "action": v.action,
                    "q_value": v.q_value,
                    "confidence": v.confidence,
                    "reasoning": v.reasoning,
                }
                for v in explanation.agent_votes
            ],
            "consensus": explanation.consensus,
            "current_drawdown_pct": explanation.current_drawdown_pct,
            "days_since_last_trade": explanation.days_since_last_trade,
            "similar_signals": explanation.similar_signals,
        }
        return _json.dumps(audit, separators=(",", ":"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _describe_factor(self, name: str, value: float) -> str:
        """Map an Alpha101 factor name to a human-readable description.

        Args:
            name: Factor identifier (e.g. ``"alpha_001"``).
            value: Computed factor value (used to contextualise the description).

        Returns:
            A plain-English description string.
        """
        base = self._FACTOR_DESCRIPTIONS.get(
            name,
            f"Custom factor '{name}': quantitative signal derived from price/volume data",
        )
        if abs(value) > 0.5:
            intensity = "strongly"
        elif abs(value) > 0.2:
            intensity = "moderately"
        else:
            intensity = "weakly"
        direction_word = "bullish" if value > 0 else "bearish"
        return f"{base} — currently {intensity} {direction_word} (value={value:+.3f})"

    def _identify_keywords(self, headlines: _List[str]) -> _List[str]:
        """Extract market-moving keywords from a list of news headlines.

        Args:
            headlines: List of news headline strings.

        Returns:
            Deduplicated list of found keywords, bullish ones first.
        """
        found_bullish = []
        found_bearish = []
        combined = " ".join(headlines).lower()
        for kw in self._BULLISH_KEYWORDS:
            if kw in combined and kw not in found_bullish:
                found_bullish.append(kw)
        for kw in self._BEARISH_KEYWORDS:
            if kw in combined and kw not in found_bearish:
                found_bearish.append(kw)
        return found_bullish + found_bearish

    def _build_factor_contributions(
        self, factor_values: _Dict[str, float]
    ) -> _List[FactorContribution]:
        """Build ranked :class:`FactorContribution` objects from raw factor values.

        Contribution percentage is approximated as the normalised absolute value
        of each factor relative to the total signal magnitude.

        Args:
            factor_values: Dict mapping factor names to float values.

        Returns:
            List of :class:`FactorContribution` objects sorted by
            ``|contribution_pct|`` descending.
        """
        if not factor_values:
            return []

        total_abs = sum(abs(v) for v in factor_values.values()) or 1.0
        contributions = []
        for name, value in factor_values.items():
            contribution_pct = (value / total_abs) * 100.0
            direction = "positive" if value > 0.01 else ("negative" if value < -0.01 else "neutral")
            contributions.append(
                FactorContribution(
                    factor_name=name,
                    factor_value=value,
                    contribution_pct=contribution_pct,
                    direction=direction,
                    description=self._describe_factor(name, value),
                )
            )
        contributions.sort(key=lambda fc: abs(fc.contribution_pct), reverse=True)
        return contributions[:10]  # Top 10 factors

    def _build_agent_votes(self, agent_outputs: list) -> _List[AgentVote]:
        """Convert raw agent output dicts to :class:`AgentVote` objects.

        Args:
            agent_outputs: List of dicts with keys ``agent_type``, ``agent_id``,
                ``action``, ``q_value``, and optionally ``confidence``.

        Returns:
            List of :class:`AgentVote` objects.
        """
        votes = []
        for agent in agent_outputs:
            agent_type = agent.get("agent_type", "Unknown")
            agent_id = int(agent.get("agent_id", 0))
            action = agent.get("action", "hold")
            q_value = float(agent.get("q_value", 0.0))
            # Derive confidence from Q-value magnitude if not supplied
            confidence = float(agent.get("confidence", min(1.0, abs(q_value) / 2.0)))
            reasoning = self._generate_vote_reasoning(agent_type, action, q_value, confidence)
            votes.append(
                AgentVote(
                    agent_type=agent_type,
                    agent_id=agent_id,
                    action=action,
                    q_value=q_value,
                    confidence=confidence,
                    reasoning=reasoning,
                )
            )
        return votes

    def _generate_vote_reasoning(
        self, agent_type: str, action: str, q_value: float, confidence: float
    ) -> str:
        """Generate a short sentence explaining why an agent voted as it did.

        Args:
            agent_type: Agent architecture name.
            action: The action chosen (``"buy"``, ``"sell"``, ``"hold"``).
            q_value: Q-value for the chosen action.
            confidence: Normalised confidence.

        Returns:
            A plain-English reasoning string.
        """
        strength = "strongly" if confidence > 0.75 else ("moderately" if confidence > 0.5 else "weakly")
        if action == "buy":
            return (
                f"{agent_type} {strength} favours buying: Q={q_value:+.3f} indicates "
                f"expected positive return from current market state."
            )
        elif action == "sell":
            return (
                f"{agent_type} {strength} favours selling: Q={q_value:+.3f} signals "
                f"expected value loss from holding the current position."
            )
        else:
            return (
                f"{agent_type} votes hold: Q={q_value:+.3f} suggests insufficient "
                f"directional conviction to justify a trade at this time."
            )

    def _format_consensus(self, votes: _List[AgentVote]) -> str:
        """Classify the ensemble voting pattern.

        Args:
            votes: List of :class:`AgentVote` objects.

        Returns:
            ``"unanimous"`` if all votes agree, ``"majority"`` if >50% agree,
            ``"split"`` otherwise.
        """
        if not votes:
            return "unanimous"
        actions = [v.action for v in votes]
        n = len(actions)
        counts: _Dict[str, int] = {}
        for a in actions:
            counts[a] = counts.get(a, 0) + 1
        max_count = max(counts.values())
        if max_count == n:
            return "unanimous"
        if max_count > n / 2:
            return "majority"
        return "split"

    def _find_similar_signals(
        self,
        direction: str,
        confidence: float,
        regime: str,
        sentiment_score: float,
        n: int = 3,
    ) -> _List[dict]:
        """Return synthetic examples of historically similar signals.

        In a production system this would query a database of past signals.
        Here it generates plausible synthetic examples for illustration.

        Args:
            direction: Current signal direction.
            confidence: Current confidence.
            regime: Current regime label.
            sentiment_score: Current sentiment score.
            n: Number of similar signals to return.

        Returns:
            List of dicts with keys ``date``, ``direction``, ``regime``,
            ``confidence``, and ``outcome_return_pct``.
        """
        import random as _random
        _random.seed(int(confidence * 1000) + len(regime))

        examples = []
        base_return = (sentiment_score - 3.0) * 1.5  # approx expected return
        for i in range(n):
            outcome = base_return + _random.gauss(0, 1.5)
            # Simulate past dates roughly
            days_ago = 30 * (i + 1) + _random.randint(0, 20)
            from datetime import timedelta
            past_date = (_datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            examples.append(
                {
                    "date": past_date,
                    "direction": direction,
                    "regime": regime,
                    "confidence": round(confidence + _random.gauss(0, 0.05), 2),
                    "outcome_return_pct": round(outcome, 2),
                }
            )
        return examples
