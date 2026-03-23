"""Walk-forward validation for crypto trading strategies.

Walk-forward validation splits a time series into expanding-window train/val
folds to produce realistic out-of-sample estimates of strategy performance.
Unlike single-split train/test approaches, walk-forward testing respects
temporal causality and surfaces overfitting that hides behind in-sample fitting.

Usage::

    from src.walk_forward import WalkForwardValidator, WalkForwardConfig

    config = WalkForwardConfig(n_splits=5, train_pct=0.7, val_pct=0.15)
    validator = WalkForwardValidator()

    def model_fn(train_data):
        # fit a model on train_data, return fitted object
        return fitted_model

    def score_fn(model, data):
        # evaluate model on data, return scalar score
        return float(score)

    result = validator.validate(data, model_fn, score_fn, config)
    print(result)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    """Configuration parameters for walk-forward validation.

    Attributes:
        n_splits:       Number of walk-forward folds to generate.
        train_pct:      Fraction of data used for training in each fold.
                        Must satisfy ``train_pct + val_pct <= 1.0``.
        val_pct:        Fraction of data used for validation in each fold.
        min_train_size: Minimum number of observations required in a training
                        window; folds shorter than this are skipped.
        step_size:      Number of observations to advance the window per fold.
                        When ``step_size == 0`` the engine auto-computes a step
                        that yields exactly ``n_splits`` folds.
    """

    n_splits: int = 5
    train_pct: float = 0.70
    val_pct: float = 0.15
    min_train_size: int = 30
    step_size: int = 0

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {self.n_splits}")
        if not (0.0 < self.train_pct < 1.0):
            raise ValueError(f"train_pct must be in (0, 1), got {self.train_pct}")
        if not (0.0 < self.val_pct < 1.0):
            raise ValueError(f"val_pct must be in (0, 1), got {self.val_pct}")
        if self.train_pct + self.val_pct > 1.0:
            raise ValueError(
                f"train_pct + val_pct must be <= 1.0, "
                f"got {self.train_pct + self.val_pct}"
            )
        if self.min_train_size < 1:
            raise ValueError(
                f"min_train_size must be >= 1, got {self.min_train_size}"
            )


# ---------------------------------------------------------------------------
# Split descriptor
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardSplit:
    """Index-based descriptor of a single walk-forward fold.

    All indices are half-open ranges ``[start, end)`` into the original
    sequence, consistent with Python slicing conventions.

    Attributes:
        split_idx:  Zero-based fold number.
        train_start: First index of the training window.
        train_end:   First index *after* the training window.
        val_start:   First index of the validation window.
        val_end:     First index *after* the validation window.
    """

    split_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int

    @property
    def train_size(self) -> int:
        """Number of training observations."""
        return self.train_end - self.train_start

    @property
    def val_size(self) -> int:
        """Number of validation observations."""
        return self.val_end - self.val_start


# ---------------------------------------------------------------------------
# Per-split result
# ---------------------------------------------------------------------------


@dataclass
class SplitResult:
    """Performance metrics for a single walk-forward fold.

    Attributes:
        split_idx:    Zero-based fold number.
        train_score:  Score on the training window (in-sample).
        val_score:    Score on the validation window (out-of-sample).
        params_used:  Optional dict of hyper-parameters / model info.
    """

    split_idx: int
    train_score: float
    val_score: float
    params_used: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregate result
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward folds.

    Attributes:
        splits:             Individual split results.
        avg_train_score:    Mean training score across splits.
        avg_val_score:      Mean validation score across splits.
        overfitting_ratio:  ``avg_train_score / avg_val_score``.  Values close
                            to 1.0 indicate low overfitting.  Values >> 1.0
                            signal that the model memorises the training window
                            but fails to generalise.
    """

    splits: List[SplitResult]
    avg_train_score: float
    avg_val_score: float
    overfitting_ratio: float

    def __str__(self) -> str:  # pragma: no cover
        lines = [
            "WalkForwardResult",
            f"  Folds            : {len(self.splits)}",
            f"  Avg train score  : {self.avg_train_score:.4f}",
            f"  Avg val score    : {self.avg_val_score:.4f}",
            f"  Overfitting ratio: {self.overfitting_ratio:.4f}",
            "",
            "  Per-fold breakdown:",
        ]
        for s in self.splits:
            lines.append(
                f"    Fold {s.split_idx:2d} | "
                f"train={s.train_score:.4f} | "
                f"val={s.val_score:.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class WalkForwardValidator:
    """Walk-forward cross-validator for sequential financial data.

    The validator uses an *expanding* window scheme: the training window
    grows by ``step_size`` observations at each fold while the validation
    window slides forward at the same pace.

    Example::

        validator = WalkForwardValidator()
        config = WalkForwardConfig(n_splits=5, train_pct=0.6, val_pct=0.2)
        splits = validator.split(500, config)
        # splits[i].train_start..train_end is the training slice
        # splits[i].val_start..val_end   is the validation slice
    """

    # ------------------------------------------------------------------
    # Public API — split generation
    # ------------------------------------------------------------------

    def split(
        self,
        n: int,
        config: Optional[WalkForwardConfig] = None,
    ) -> List[WalkForwardSplit]:
        """Generate walk-forward splits for a series of length *n*.

        Parameters
        ----------
        n:
            Total number of observations in the time series.
        config:
            Walk-forward configuration.  Uses :class:`WalkForwardConfig`
            defaults when ``None``.

        Returns
        -------
        list[WalkForwardSplit]
            Ordered (ascending ``split_idx``) list of fold descriptors.

        Raises
        ------
        ValueError
            If *n* is too small to form even one valid split.
        """
        if config is None:
            config = WalkForwardConfig()

        val_size = max(1, math.floor(n * config.val_pct))

        # Determine step size
        if config.step_size > 0:
            step = config.step_size
        else:
            # Auto step: distribute remaining data after first train+val across n_splits
            total_needed = config.n_splits * val_size + math.floor(n * config.train_pct)
            if total_needed > n:
                # Shrink val_size gracefully
                val_size = max(1, (n - math.floor(n * config.train_pct)) // config.n_splits)
            step = max(1, val_size)

        splits: List[WalkForwardSplit] = []
        initial_train = math.floor(n * config.train_pct)
        initial_train = max(initial_train, config.min_train_size)

        for i in range(config.n_splits):
            train_start = 0  # expanding window always starts at 0
            train_end = initial_train + i * step
            val_start = train_end
            val_end = val_start + val_size

            if val_end > n:
                break  # not enough data for this fold
            if (train_end - train_start) < config.min_train_size:
                continue  # training window too small

            splits.append(
                WalkForwardSplit(
                    split_idx=i,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                )
            )

        if not splits:
            raise ValueError(
                f"Cannot generate any valid splits for n={n} with config={config!r}. "
                f"Try increasing n or decreasing min_train_size."
            )

        # Re-index so split_idx is contiguous from 0
        for new_idx, sp in enumerate(splits):
            sp.split_idx = new_idx

        return splits

    # ------------------------------------------------------------------
    # Public API — validation
    # ------------------------------------------------------------------

    def validate(
        self,
        data: Any,
        model_fn: Callable[[Any], Any],
        score_fn: Callable[[Any, Any], float],
        config: Optional[WalkForwardConfig] = None,
    ) -> WalkForwardResult:
        """Run walk-forward validation on *data*.

        Parameters
        ----------
        data:
            Sequence-like object that supports slicing (``data[a:b]``).
            Commonly a :class:`numpy.ndarray`, :class:`pandas.DataFrame`,
            or plain Python list.
        model_fn:
            ``model_fn(train_slice) -> model`` — Fit a model on the
            training slice and return the fitted model object.
        score_fn:
            ``score_fn(model, slice) -> float`` — Evaluate *model* on
            a data slice and return a scalar score (higher is better).
        config:
            Walk-forward configuration.

        Returns
        -------
        WalkForwardResult
            Aggregated results including per-fold scores and overfitting ratio.
        """
        if config is None:
            config = WalkForwardConfig()

        n = len(data)  # type: ignore[arg-type]
        fold_descriptors = self.split(n, config)

        split_results: List[SplitResult] = []

        for fold in fold_descriptors:
            train_slice = data[fold.train_start : fold.train_end]
            val_slice = data[fold.val_start : fold.val_end]

            model = model_fn(train_slice)
            train_score = float(score_fn(model, train_slice))
            val_score = float(score_fn(model, val_slice))

            split_results.append(
                SplitResult(
                    split_idx=fold.split_idx,
                    train_score=train_score,
                    val_score=val_score,
                )
            )

        avg_train = (
            sum(r.train_score for r in split_results) / len(split_results)
            if split_results
            else float("nan")
        )
        avg_val = (
            sum(r.val_score for r in split_results) / len(split_results)
            if split_results
            else float("nan")
        )

        if avg_val != 0.0 and not math.isnan(avg_val):
            overfitting_ratio = avg_train / avg_val
        elif avg_val == 0.0 and avg_train == 0.0:
            overfitting_ratio = 1.0
        else:
            overfitting_ratio = float("inf")

        return WalkForwardResult(
            splits=split_results,
            avg_train_score=avg_train,
            avg_val_score=avg_val,
            overfitting_ratio=overfitting_ratio,
        )
