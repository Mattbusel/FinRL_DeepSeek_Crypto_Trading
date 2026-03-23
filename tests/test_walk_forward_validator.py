"""Tests for src/walk_forward.py (WalkForwardValidator, WalkForwardConfig, etc.)"""
from __future__ import annotations

import math
import pytest

from src.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardSplit,
    WalkForwardValidator,
    SplitResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_model(train_data):
    """Trivial model: returns mean of training data."""
    return sum(train_data) / len(train_data) if train_data else 0.0


def _rmse_score(model, data):
    """Score: negative RMSE (higher is better means lower error)."""
    if not data:
        return 0.0
    return -math.sqrt(sum((x - model) ** 2 for x in data) / len(data))


def _constant_score(model, data):
    """Always returns 1.0 — useful for overfitting ratio tests."""
    return 1.0


def make_data(n=200):
    """Simple linearly increasing series."""
    return list(range(n))


# ---------------------------------------------------------------------------
# WalkForwardConfig validation
# ---------------------------------------------------------------------------


class TestWalkForwardConfig:
    def test_default_config_is_valid(self):
        cfg = WalkForwardConfig()
        assert cfg.n_splits == 5
        assert 0 < cfg.train_pct < 1
        assert 0 < cfg.val_pct < 1
        assert cfg.min_train_size >= 1

    def test_invalid_n_splits_raises(self):
        with pytest.raises(ValueError):
            WalkForwardConfig(n_splits=0)

    def test_invalid_train_pct_zero_raises(self):
        with pytest.raises(ValueError):
            WalkForwardConfig(train_pct=0.0)

    def test_invalid_train_pct_one_raises(self):
        with pytest.raises(ValueError):
            WalkForwardConfig(train_pct=1.0)

    def test_invalid_val_pct_raises(self):
        with pytest.raises(ValueError):
            WalkForwardConfig(val_pct=0.0)

    def test_train_val_sum_exceeds_one_raises(self):
        with pytest.raises(ValueError):
            WalkForwardConfig(train_pct=0.6, val_pct=0.5)

    def test_invalid_min_train_size_raises(self):
        with pytest.raises(ValueError):
            WalkForwardConfig(min_train_size=0)


# ---------------------------------------------------------------------------
# WalkForwardValidator.split()
# ---------------------------------------------------------------------------


class TestWalkForwardValidatorSplit:
    def test_split_returns_list(self):
        v = WalkForwardValidator()
        splits = v.split(500, WalkForwardConfig(n_splits=5))
        assert isinstance(splits, list)

    def test_split_count_at_most_n_splits(self):
        v = WalkForwardValidator()
        cfg = WalkForwardConfig(n_splits=5, train_pct=0.6, val_pct=0.15)
        splits = v.split(500, cfg)
        assert len(splits) <= cfg.n_splits

    def test_split_indices_contiguous(self):
        v = WalkForwardValidator()
        splits = v.split(300, WalkForwardConfig(n_splits=4))
        for i, sp in enumerate(splits):
            assert sp.split_idx == i

    def test_split_train_end_equals_val_start(self):
        """Training window ends exactly where validation begins."""
        v = WalkForwardValidator()
        splits = v.split(300, WalkForwardConfig(n_splits=4))
        for sp in splits:
            assert sp.train_end == sp.val_start

    def test_split_no_look_ahead(self):
        """Validation window does not overlap training window."""
        v = WalkForwardValidator()
        splits = v.split(300, WalkForwardConfig(n_splits=4))
        for sp in splits:
            assert sp.val_start >= sp.train_end

    def test_split_train_size_property(self):
        v = WalkForwardValidator()
        splits = v.split(300, WalkForwardConfig(n_splits=3))
        for sp in splits:
            assert sp.train_size == sp.train_end - sp.train_start

    def test_split_val_size_property(self):
        v = WalkForwardValidator()
        splits = v.split(300, WalkForwardConfig(n_splits=3))
        for sp in splits:
            assert sp.val_size == sp.val_end - sp.val_start

    def test_split_expanding_window(self):
        """Later folds have larger or equal training windows."""
        v = WalkForwardValidator()
        splits = v.split(500, WalkForwardConfig(n_splits=5))
        for i in range(len(splits) - 1):
            assert splits[i + 1].train_size >= splits[i].train_size

    def test_split_val_end_within_bounds(self):
        """Validation end never exceeds dataset length."""
        n = 300
        v = WalkForwardValidator()
        splits = v.split(n, WalkForwardConfig(n_splits=4))
        for sp in splits:
            assert sp.val_end <= n

    def test_split_min_train_size_respected(self):
        v = WalkForwardValidator()
        cfg = WalkForwardConfig(n_splits=3, min_train_size=50)
        splits = v.split(300, cfg)
        for sp in splits:
            assert sp.train_size >= 50

    def test_split_small_n_raises(self):
        v = WalkForwardValidator()
        cfg = WalkForwardConfig(n_splits=10, train_pct=0.9, val_pct=0.09, min_train_size=100)
        with pytest.raises(ValueError):
            v.split(10, cfg)

    def test_split_default_config(self):
        v = WalkForwardValidator()
        splits = v.split(500)
        assert len(splits) >= 1

    def test_split_custom_step_size(self):
        v = WalkForwardValidator()
        cfg = WalkForwardConfig(n_splits=4, train_pct=0.5, val_pct=0.1, step_size=25)
        splits = v.split(400, cfg)
        assert len(splits) >= 1
        # With step_size=25, consecutive training windows differ by 25
        if len(splits) >= 2:
            diff = splits[1].train_size - splits[0].train_size
            assert diff == 25

    def test_split_train_start_always_zero(self):
        """Expanding window — training always starts at index 0."""
        v = WalkForwardValidator()
        splits = v.split(400, WalkForwardConfig(n_splits=5))
        for sp in splits:
            assert sp.train_start == 0

    def test_split_result_type(self):
        v = WalkForwardValidator()
        splits = v.split(300, WalkForwardConfig(n_splits=3))
        for sp in splits:
            assert isinstance(sp, WalkForwardSplit)


# ---------------------------------------------------------------------------
# WalkForwardValidator.validate()
# ---------------------------------------------------------------------------


class TestWalkForwardValidatorValidate:
    def _run(self, n=300, **kwargs):
        v = WalkForwardValidator()
        cfg = WalkForwardConfig(**kwargs) if kwargs else WalkForwardConfig()
        data = make_data(n)
        return v.validate(data, _identity_model, _rmse_score, cfg)

    def test_validate_returns_result(self):
        result = self._run()
        assert isinstance(result, WalkForwardResult)

    def test_validate_splits_count(self):
        result = self._run(n=300, n_splits=4)
        assert 1 <= len(result.splits) <= 4

    def test_validate_avg_train_score_finite(self):
        result = self._run()
        assert math.isfinite(result.avg_train_score)

    def test_validate_avg_val_score_finite(self):
        result = self._run()
        assert math.isfinite(result.avg_val_score)

    def test_validate_overfitting_ratio_positive(self):
        result = self._run()
        assert result.overfitting_ratio >= 0.0 or math.isinf(result.overfitting_ratio)

    def test_validate_constant_score_ratio_near_one(self):
        """When both train and val score are 1.0, ratio should be 1.0."""
        v = WalkForwardValidator()
        data = make_data(300)
        result = v.validate(data, _identity_model, _constant_score)
        assert math.isclose(result.overfitting_ratio, 1.0)

    def test_validate_split_results_ordered(self):
        result = self._run()
        indices = [s.split_idx for s in result.splits]
        assert indices == sorted(indices)

    def test_validate_split_result_type(self):
        result = self._run()
        for s in result.splits:
            assert isinstance(s, SplitResult)

    def test_validate_avg_train_matches_mean(self):
        result = self._run()
        expected = sum(s.train_score for s in result.splits) / len(result.splits)
        assert math.isclose(result.avg_train_score, expected)

    def test_validate_avg_val_matches_mean(self):
        result = self._run()
        expected = sum(s.val_score for s in result.splits) / len(result.splits)
        assert math.isclose(result.avg_val_score, expected)

    def test_validate_with_numpy_array(self):
        pytest.importorskip("numpy")
        import numpy as np
        v = WalkForwardValidator()
        data = np.linspace(0, 1, 300)
        result = v.validate(data, _identity_model, _rmse_score)
        assert isinstance(result, WalkForwardResult)

    def test_validate_params_used_default_empty(self):
        result = self._run()
        for s in result.splits:
            assert isinstance(s.params_used, dict)
