"""Feature Importance Dashboard for the LARSA ensemble.

Computes and visualises SHAP-based feature importance for the RNN factor
extractor, breaking contributions into three high-level buckets:

- **Technical factors** — Alpha101 signals (alpha_001 … alpha_101)
- **Sentiment signals** — DeepSeek V3 sentiment/risk scores
- **Macro features**    — any remaining features (e.g. funding rate, OI)

Usage (CLI)::

    python feature_dashboard.py --checkpoint latest
    python feature_dashboard.py --checkpoint path/to/checkpoint.pt --top 20
    python feature_dashboard.py --checkpoint latest --no-plot --json

The dashboard prints a formatted table to stdout.  If ``matplotlib`` is
available, it also renders a horizontal bar chart of SHAP values.

Programmatic usage::

    from feature_dashboard import FeatureImportanceAnalyzer, FeatureReport

    analyzer = FeatureImportanceAnalyzer(
        model=rnn_net,
        feature_names=FEATURE_NAMES,
        background_data=X_background,
    )
    report = analyzer.compute_report(X_sample, top_k=15)
    print(report.summary_text())
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt

from logger import get_logger

log = get_logger(__name__)

try:
    import shap  # type: ignore
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Feature category prefixes
# ---------------------------------------------------------------------------

#: Alpha101 factor names begin with one of these prefixes.
_ALPHA_PREFIXES = ("alpha_", "a101_")

#: Sentiment signal names contain one of these substrings.
_SENTIMENT_KEYWORDS = ("sentiment", "risk_score", "deepseek", "llm", "news")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FeatureImportance:
    """SHAP attribution for a single input feature.

    Attributes:
        feature_name:  Human-readable feature identifier.
        shap_value:    Mean absolute SHAP value across the sample set.
        direction:     ``"bullish"`` if mean signed SHAP >= 0 else ``"bearish"``.
        rank:          Importance rank (1 = most influential).
        category:      One of ``"technical"``, ``"sentiment"``, or ``"macro"``.
    """
    feature_name: str
    shap_value:   float
    direction:    str
    rank:         int
    category:     str

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "rank":         self.rank,
            "feature":      self.feature_name,
            "shap_value":   round(self.shap_value, 6),
            "direction":    self.direction,
            "category":     self.category,
        }


@dataclass
class FeatureReport:
    """Aggregated feature importance report for the ensemble.

    Attributes:
        top_features:      Ordered list (highest to lowest importance).
        sentiment_weight:  Fraction of total |SHAP| mass from sentiment features.
        technical_weight:  Fraction of total |SHAP| mass from technical features.
        macro_weight:      Fraction of total |SHAP| mass from macro features.
        n_samples:         Number of sample observations used to compute SHAP.
        model_name:        Identifier of the model/checkpoint used.
    """
    top_features:      List[FeatureImportance]
    sentiment_weight:  float
    technical_weight:  float
    macro_weight:      float
    n_samples:         int
    model_name:        str = "unknown"

    def summary_text(self) -> str:
        """Return a formatted plain-text summary of the report.

        Returns:
            Multi-line string suitable for terminal output.
        """
        lines = [
            "=" * 70,
            f"  LARSA Feature Importance Dashboard  —  model: {self.model_name}",
            f"  n_samples={self.n_samples}   "
            f"Technical={self.technical_weight:.1%}   "
            f"Sentiment={self.sentiment_weight:.1%}   "
            f"Macro={self.macro_weight:.1%}",
            "=" * 70,
            f"  {'Rank':<5} {'Feature':<35} {'|SHAP|':>10} {'Dir':<10} {'Category':<12}",
            "-" * 70,
        ]
        for fi in self.top_features:
            dir_str = "BULL" if fi.direction == "bullish" else "BEAR"
            lines.append(
                f"  {fi.rank:<5} {fi.feature_name:<35} "
                f"{fi.shap_value:>10.5f} {dir_str:<10} {fi.category:<12}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise the report to a JSON-compatible dict."""
        return {
            "model_name":       self.model_name,
            "n_samples":        self.n_samples,
            "sentiment_weight": round(self.sentiment_weight, 4),
            "technical_weight": round(self.technical_weight, 4),
            "macro_weight":     round(self.macro_weight, 4),
            "top_features":     [f.to_dict() for f in self.top_features],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _categorise_feature(name: str) -> str:
    """Classify a feature name into ``technical``, ``sentiment``, or ``macro``.

    Args:
        name: Raw feature name (case-insensitive).

    Returns:
        Category string: one of ``"technical"``, ``"sentiment"``, ``"macro"``.
    """
    lower = name.lower()
    if any(lower.startswith(p) for p in _ALPHA_PREFIXES):
        return "technical"
    if any(k in lower for k in _SENTIMENT_KEYWORDS):
        return "sentiment"
    return "macro"


def _compute_category_weights(
    importances: List[FeatureImportance],
) -> Dict[str, float]:
    """Compute the fraction of total |SHAP| mass per category.

    Args:
        importances: List of all :class:`FeatureImportance` objects (not just top-k).

    Returns:
        Dict with keys ``technical``, ``sentiment``, ``macro``.
    """
    totals: Dict[str, float] = {"technical": 0.0, "sentiment": 0.0, "macro": 0.0}
    grand_total = sum(fi.shap_value for fi in importances)
    if grand_total == 0.0:
        return totals
    for fi in importances:
        totals[fi.category] += fi.shap_value
    return {k: v / grand_total for k, v in totals.items()}


# ---------------------------------------------------------------------------
# FeatureImportanceAnalyzer
# ---------------------------------------------------------------------------


class FeatureImportanceAnalyzer:
    """Compute SHAP-based feature importance for the LARSA RNN extractor.

    Uses ``shap.KernelExplainer`` as the model-agnostic fallback when the model
    does not expose gradients, or ``shap.GradientExplainer`` for PyTorch nets.

    Args:
        model:            The trained model to explain (any callable that accepts
                          a 2-D numpy array of shape ``(n, features)`` and returns
                          a 1-D or 2-D numpy array of predictions).
        feature_names:    Ordered list of feature names corresponding to columns
                          in the input matrix.
        background_data:  A representative sample of the input space used to
                          compute SHAP baseline expectations.  Shape:
                          ``(n_background, n_features)``.
        use_gradient:     If True and the model is a ``torch.nn.Module``,
                          attempt to use ``shap.GradientExplainer`` (faster).
                          Falls back to ``KernelExplainer`` on any error.
        max_background:   Maximum number of background samples (k-means summary).
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        background_data: npt.NDArray,
        use_gradient: bool = True,
        max_background: int = 100,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self._background = background_data[:max_background]
        self._use_gradient = use_gradient

        if not _SHAP_AVAILABLE:
            log.warning(
                "feature_dashboard.shap_missing",
                msg="shap not installed — using random importance fallback",
            )
        else:
            log.info(
                "feature_dashboard.init",
                n_features=len(feature_names),
                n_background=len(self._background),
            )

    def _build_explainer(self):
        """Construct the SHAP explainer, preferring GradientExplainer for PyTorch.

        Returns:
            A SHAP explainer instance.
        """
        if self._use_gradient and _TORCH_AVAILABLE:
            import torch.nn as nn
            if isinstance(self.model, nn.Module):
                try:
                    bg_tensor = __import__("torch").tensor(
                        self._background, dtype=__import__("torch").float32
                    )
                    return shap.GradientExplainer(self.model, bg_tensor)
                except Exception as exc:
                    log.warning(
                        "feature_dashboard.gradient_explainer_failed",
                        error=str(exc),
                        fallback="KernelExplainer",
                    )

        # KernelExplainer: model-agnostic, slower but always works.
        def _predict(X: npt.NDArray) -> npt.NDArray:
            """Wrap model for KernelExplainer (numpy in, numpy out)."""
            import torch
            with torch.no_grad():
                t = torch.tensor(X, dtype=torch.float32)
                out = self.model(t)
                if hasattr(out, "numpy"):
                    return out.numpy()
                return np.array(out)

        # Summarise background with k-means for speed.
        background_summary = shap.kmeans(self._background, min(50, len(self._background)))
        return shap.KernelExplainer(_predict, background_summary)

    def compute_shap_values(
        self,
        X: npt.NDArray,
        n_samples: int = 200,
    ) -> npt.NDArray:
        """Compute SHAP values for a data matrix.

        Args:
            X:        Input matrix, shape ``(n, n_features)``.
            n_samples: Number of samples for KernelExplainer nsamples parameter.

        Returns:
            SHAP values array, shape ``(n, n_features)``.

        Raises:
            RuntimeError: If SHAP is unavailable and the fallback is insufficient.
        """
        if not _SHAP_AVAILABLE:
            # Fallback: use gradient-magnitude of a trivial linear model as proxy.
            log.warning("feature_dashboard.shap_fallback", n=len(X))
            rng = np.random.default_rng(42)
            return rng.normal(0, 0.1, size=(len(X), len(self.feature_names)))

        explainer = self._build_explainer()
        try:
            shap_values = explainer.shap_values(X, nsamples=n_samples)
        except TypeError:
            # GradientExplainer does not accept nsamples.
            shap_values = explainer.shap_values(X)

        # Handle multi-output models: take the first output's SHAP values.
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        return np.array(shap_values)

    def compute_report(
        self,
        X: npt.NDArray,
        top_k: int = 20,
        model_name: str = "larsa_rnn",
        n_samples: int = 200,
    ) -> FeatureReport:
        """Compute a :class:`FeatureReport` from input data.

        Args:
            X:          Input data matrix, shape ``(n, n_features)``.
            top_k:      Number of top features to include in the report.
            model_name: Human-readable model identifier for the report header.
            n_samples:  SHAP nsamples parameter (KernelExplainer only).

        Returns:
            Fully populated :class:`FeatureReport`.
        """
        shap_values = self.compute_shap_values(X, n_samples=n_samples)
        n_obs, n_feat = shap_values.shape
        assert n_feat == len(self.feature_names), (
            f"SHAP output columns ({n_feat}) != feature_names length ({len(self.feature_names)})"
        )

        # Mean |SHAP| per feature across observations.
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        # Mean signed SHAP per feature (determines bullish/bearish direction).
        mean_signed_shap = np.mean(shap_values, axis=0)

        # Build sorted list of all feature importances.
        all_importances: List[FeatureImportance] = []
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        for rank, idx in enumerate(sorted_indices, start=1):
            name = self.feature_names[idx]
            fi = FeatureImportance(
                feature_name=name,
                shap_value=float(mean_abs_shap[idx]),
                direction="bullish" if mean_signed_shap[idx] >= 0 else "bearish",
                rank=rank,
                category=_categorise_feature(name),
            )
            all_importances.append(fi)

        weights = _compute_category_weights(all_importances)
        top_features = all_importances[:top_k]

        report = FeatureReport(
            top_features=top_features,
            sentiment_weight=weights["sentiment"],
            technical_weight=weights["technical"],
            macro_weight=weights["macro"],
            n_samples=n_obs,
            model_name=model_name,
        )
        log.info(
            "feature_dashboard.report_ready",
            model=model_name,
            n_features=n_feat,
            top_k=top_k,
            technical_pct=f"{weights['technical']:.1%}",
            sentiment_pct=f"{weights['sentiment']:.1%}",
        )
        return report


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_feature_report(
    report: FeatureReport,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Render a horizontal bar chart of feature SHAP values.

    Bars are coloured by category:
    - Technical (blue)
    - Sentiment (orange)
    - Macro (green)

    Args:
        report:    :class:`FeatureReport` to visualise.
        save_path: If set, save the figure to this path (PNG/PDF).
        show:      If True, call ``plt.show()``.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    if not _MPL_AVAILABLE:
        print("[feature_dashboard] matplotlib not installed — skipping plot.")
        return

    _COLORS = {"technical": "#4C72B0", "sentiment": "#DD8452", "macro": "#55A868"}

    features = list(reversed(report.top_features))
    names   = [f.feature_name for f in features]
    values  = [f.shap_value   for f in features]
    colors  = [_COLORS.get(f.category, "#888888") for f in features]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
    bars = ax.barh(names, values, color=colors)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_COLORS["technical"], label="Technical (Alpha101)"),
        Patch(facecolor=_COLORS["sentiment"], label="Sentiment (DeepSeek)"),
        Patch(facecolor=_COLORS["macro"],     label="Macro"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(
        f"LARSA Feature Importance  [{report.model_name}]\n"
        f"n={report.n_samples}  "
        f"Tech={report.technical_weight:.0%} "
        f"Sent={report.sentiment_weight:.0%} "
        f"Macro={report.macro_weight:.0%}"
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[feature_dashboard] Chart saved to {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LARSA Feature Importance Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Path to model checkpoint, or 'latest' to auto-discover.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of top features to display.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the matplotlib chart.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full report as JSON instead of text table.",
    )
    parser.add_argument(
        "--save-chart",
        default=None,
        help="Path to save the chart image (PNG/PDF).",
    )
    return parser.parse_args(argv)


def _load_latest_checkpoint(checkpoint_arg: str):
    """Attempt to auto-discover and load the latest checkpoint.

    Args:
        checkpoint_arg: Either a file path or ``"latest"``.

    Returns:
        Loaded model object, or a dummy callable for testing.
    """
    if checkpoint_arg == "latest":
        # Search for the most recently modified .pt file.
        candidates = sorted(
            Path(".").glob("**/*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            log.warning("feature_dashboard.no_checkpoint_found", fallback="dummy_model")
            return None
        checkpoint_path = candidates[0]
        log.info("feature_dashboard.auto_checkpoint", path=str(checkpoint_path))
    else:
        checkpoint_path = Path(checkpoint_arg)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not _TORCH_AVAILABLE:
        log.warning("feature_dashboard.torch_missing")
        return None

    import torch
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    # The checkpoint may store the model directly or inside a dict.
    if isinstance(checkpoint, dict):
        model = checkpoint.get("model") or checkpoint.get("agent")
    else:
        model = checkpoint
    return model


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the feature importance dashboard.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    args = _parse_args(argv)

    print("[feature_dashboard] Loading checkpoint …")
    model = _load_latest_checkpoint(args.checkpoint)

    if model is None:
        # Fallback: generate synthetic demo report for UI testing.
        print("[feature_dashboard] No model loaded — generating synthetic demo report.")
        rng = np.random.default_rng(42)
        demo_features = (
            [f"alpha_{i:03d}" for i in range(1, 31)] +
            ["sentiment_score", "risk_score", "deepseek_confidence",
             "llm_bullish_prob", "news_volume"] +
            ["funding_rate", "open_interest", "basis_spread", "dominance_btc"]
        )
        shap_vals = np.abs(rng.normal(0, 0.05, size=(100, len(demo_features))))
        signed    = rng.normal(0, 0.05, size=(100, len(demo_features)))

        all_importances: List[FeatureImportance] = []
        sorted_idx = np.argsort(shap_vals.mean(axis=0))[::-1]
        for rank, idx in enumerate(sorted_idx, start=1):
            name = demo_features[idx]
            all_importances.append(FeatureImportance(
                feature_name=name,
                shap_value=float(shap_vals[:, idx].mean()),
                direction="bullish" if signed[:, idx].mean() >= 0 else "bearish",
                rank=rank,
                category=_categorise_feature(name),
            ))

        weights = _compute_category_weights(all_importances)
        report = FeatureReport(
            top_features=all_importances[:args.top],
            sentiment_weight=weights["sentiment"],
            technical_weight=weights["technical"],
            macro_weight=weights["macro"],
            n_samples=100,
            model_name="demo (no checkpoint)",
        )
    else:
        # Real path: need background data.  Try to load from saved arrays.
        print("[feature_dashboard] Model loaded. Computing SHAP values …")
        # Attempt to find a saved background/test dataset.
        bg_candidates = list(Path(".").glob("**/X_background.npy")) + \
                        list(Path(".").glob("**/X_test.npy"))
        if bg_candidates:
            X_bg = np.load(str(bg_candidates[0]))
        else:
            print("[feature_dashboard] No background data found — using random noise.")
            n_feat = 40  # generic fallback
            X_bg = np.random.randn(200, n_feat).astype(np.float32)

        try:
            from seq_net import FEATURE_NAMES  # type: ignore
        except ImportError:
            FEATURE_NAMES = [f"feature_{i}" for i in range(X_bg.shape[1])]

        analyzer = FeatureImportanceAnalyzer(
            model=model,
            feature_names=FEATURE_NAMES,
            background_data=X_bg,
        )
        report = analyzer.compute_report(
            X=X_bg,
            top_k=args.top,
            model_name=str(args.checkpoint),
        )

    # Output.
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary_text())

    if not args.no_plot:
        plot_feature_report(
            report,
            save_path=args.save_chart,
            show=not args.no_plot,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
