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
