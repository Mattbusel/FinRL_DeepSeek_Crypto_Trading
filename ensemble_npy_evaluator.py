"""Evaluate saved experiment result arrays and print financial metrics.

Scans the ``exps/`` directory for ``*.npy`` files whose names contain
``net_assets`` or ``correct_preds`` and logs Sharpe ratio, max drawdown,
return-over-max-drawdown, win rate, and loss rate for each.

Usage::

    python ensemble_npy_evaluator.py
"""

from __future__ import annotations

import logging
import os

import numpy as np

from metrics import max_drawdown, return_over_max_drawdown, sharpe_ratio

log = logging.getLogger(__name__)


def evaluate_directory(directory_path: str = "exps") -> dict[str, dict]:
    """Load and evaluate all experiment result arrays in *directory_path*.

    For each ``.npy`` file found:

    * Files whose name contains ``net_assets`` are treated as cumulative
      net-asset-value series; period returns are derived and the Sharpe
      ratio, max drawdown, and RoMaD are logged.
    * Files whose name contains ``correct_preds`` are treated as integer
      prediction arrays (``1`` = correct, ``-1`` = wrong) and win/loss
      rates are logged.

    Args:
        directory_path: Path to the directory containing ``.npy`` result files.

    Returns:
        A dictionary mapping filenames to their computed metric dictionaries.
        Metric keys depend on the file type: ``net_assets`` files produce
        ``{"sharpe", "roma", "max_drawdown"}``; ``correct_preds`` files
        produce ``{"win_rate", "loss_rate"}``.
    """
    results: dict[str, dict] = {}

    if not os.path.isdir(directory_path):
        log.warning("evaluate_directory -- directory not found: %s", directory_path)
        return results

    for filename in os.listdir(directory_path):
        if not filename.endswith(".npy"):
            continue

        file_path = os.path.join(directory_path, filename)
        data = np.load(file_path)

        if "net_assets" in filename:
            returns = np.array([(data[t + 1] - data[t]) / data[t] for t in range(len(data) - 1)])
            sr = sharpe_ratio(returns)
            mdd = max_drawdown(returns)
            roma = return_over_max_drawdown(returns)

            log.info(
                "%s  sharpe=%.4f  roma=%.4f  max_drawdown=%.4f",
                filename,
                sr,
                roma,
                mdd,
            )
            results[filename] = {"sharpe": sr, "roma": roma, "max_drawdown": mdd}

        elif "correct_preds" in filename:
            n = len(data)
            win_rate = float(np.count_nonzero(data == 1) / n)
            loss_rate = float(np.count_nonzero(data == -1) / n)

            log.info(
                "%s  win_rate=%.4f  loss_rate=%.4f",
                filename,
                win_rate,
                loss_rate,
            )
            results[filename] = {"win_rate": win_rate, "loss_rate": loss_rate}

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    evaluate_directory()
