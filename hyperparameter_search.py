"""Hyperparameter Search for LARSA.

Grid search and random search over key hyperparameters:
- Learning rate, batch size, gamma, hidden layer sizes
- Ensemble weights, regime switch thresholds
- DeepSeek confidence thresholds

Runs each config through a user-supplied evaluation function and ranks
results by Sharpe ratio (or any other metric).

Example::

    from hyperparameter_search import HyperparamSearch

    def my_eval(params: dict) -> dict:
        # Train/evaluate a LARSA model with these params and return metrics
        return {"sharpe": 1.2, "max_drawdown": -0.15, "total_return": 0.08}

    searcher = HyperparamSearch(eval_fn=my_eval)
    results = searcher.random_search(n_trials=10)
    best = searcher.best_config()
    print(best)
    searcher.save_results("my_search.json")
"""

from __future__ import annotations

import itertools
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HyperparamConfig:
    """A single hyperparameter configuration and its evaluation results.

    Attributes:
        config_id: Unique identifier for this configuration.
        params: Dict mapping hyperparameter names to values.
        sharpe: Sharpe ratio achieved on the validation window.
        max_drawdown: Maximum drawdown (negative float, e.g. ``-0.12``).
        total_return: Total return as a decimal (e.g. ``0.15`` = 15%).
        eval_time_secs: Wall-clock seconds taken to evaluate this config.
    """

    config_id: str
    params: Dict[str, Any]
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_return: Optional[float] = None
    eval_time_secs: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for JSON output."""
        return {
            "config_id": self.config_id,
            "params": self.params,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
            "eval_time_secs": self.eval_time_secs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HyperparamConfig":
        """Deserialise from a plain dict."""
        return cls(
            config_id=d["config_id"],
            params=d["params"],
            sharpe=d.get("sharpe"),
            max_drawdown=d.get("max_drawdown"),
            total_return=d.get("total_return"),
            eval_time_secs=d.get("eval_time_secs"),
        )


# ---------------------------------------------------------------------------
# HyperparamSearch
# ---------------------------------------------------------------------------


class HyperparamSearch:
    """Grid search and random search over LARSA hyperparameters.

    All search methods call ``eval_fn(params: dict) -> dict`` for each
    configuration.  The ``eval_fn`` must return a dict with at least one of
    the keys ``"sharpe"``, ``"max_drawdown"``, or ``"total_return"``.  Missing
    keys are stored as ``None``.

    Args:
        eval_fn: Callable that takes a param dict and returns a metrics dict.
        n_jobs: Number of parallel workers (default 1 = sequential).  Parallel
            execution uses :class:`concurrent.futures.ThreadPoolExecutor`.

    Class attributes:
        SEARCH_SPACE: Default search space used when no override is supplied to
            :meth:`grid_search` or :meth:`random_search`.
    """

    SEARCH_SPACE: Dict[str, List[Any]] = {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "batch_size": [32, 64, 128],
        "gamma": [0.95, 0.97, 0.99],
        "hidden_sizes": [[256, 256], [512, 256], [256, 128, 64]],
        "confidence_threshold": [0.5, 0.6, 0.7],
        "regime_switch_threshold": [0.3, 0.4, 0.5],
    }

    def __init__(
        self,
        eval_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        n_jobs: int = 1,
    ) -> None:
        self.eval_fn = eval_fn
        self.n_jobs = max(1, n_jobs)
        self.results: List[HyperparamConfig] = []

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def grid_search(
        self, param_grid: Optional[Dict[str, List[Any]]] = None
    ) -> List[HyperparamConfig]:
        """Enumerate every combination in *param_grid* and evaluate each one.

        Args:
            param_grid: Dict mapping param names to lists of candidate values.
                If ``None``, :attr:`SEARCH_SPACE` is used.

        Returns:
            List of :class:`HyperparamConfig` objects, sorted by Sharpe ratio
            descending (``None`` values placed last).
        """
        grid = param_grid if param_grid is not None else self.SEARCH_SPACE
        keys = list(grid.keys())
        value_lists = [grid[k] for k in keys]
        combos = list(itertools.product(*value_lists))

        log.info(
            "hyperparam_search.grid_start",
            n_combinations=len(combos),
            params=keys,
        )

        configs: List[Dict[str, Any]] = [
            {keys[i]: combo[i] for i in range(len(keys))} for combo in combos
        ]
        new_results = self._evaluate_all(configs)
        self.results.extend(new_results)
        self._sort_results()

        log.info(
            "hyperparam_search.grid_done",
            n_evaluated=len(new_results),
            best_sharpe=self.best_config("sharpe").sharpe if self.results else None,
        )
        return new_results

    def random_search(
        self,
        n_trials: int = 20,
        param_space: Optional[Dict[str, List[Any]]] = None,
    ) -> List[HyperparamConfig]:
        """Sample *n_trials* random combinations from *param_space* and evaluate each.

        Args:
            n_trials: Number of random configurations to sample.
            param_space: Dict mapping param names to lists of candidate values.
                If ``None``, :attr:`SEARCH_SPACE` is used.

        Returns:
            List of :class:`HyperparamConfig` objects, sorted by Sharpe ratio.
        """
        space = param_space if param_space is not None else self.SEARCH_SPACE

        log.info(
            "hyperparam_search.random_start",
            n_trials=n_trials,
            params=list(space.keys()),
        )

        configs: List[Dict[str, Any]] = []
        for _ in range(n_trials):
            sample = {k: random.choice(v) for k, v in space.items()}
            configs.append(sample)

        new_results = self._evaluate_all(configs)
        self.results.extend(new_results)
        self._sort_results()

        log.info(
            "hyperparam_search.random_done",
            n_evaluated=len(new_results),
            best_sharpe=self.best_config("sharpe").sharpe if self.results else None,
        )
        return new_results

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    def best_config(self, metric: str = "sharpe") -> Optional[HyperparamConfig]:
        """Return the single best :class:`HyperparamConfig` by *metric*.

        Args:
            metric: One of ``"sharpe"``, ``"total_return"``, or
                ``"max_drawdown"`` (for drawdown, lower magnitude is better).

        Returns:
            The best :class:`HyperparamConfig`, or ``None`` if no results exist.
        """
        if not self.results:
            return None
        return self.top_k(k=1, metric=metric)[0]

    def top_k(
        self, k: int = 5, metric: str = "sharpe"
    ) -> List[HyperparamConfig]:
        """Return the top *k* configurations ranked by *metric*.

        Args:
            k: Number of results to return.
            metric: Ranking metric (``"sharpe"``, ``"total_return"``, or
                ``"max_drawdown"``).

        Returns:
            List of up to *k* :class:`HyperparamConfig` objects.
        """
        if not self.results:
            return []

        def _key(cfg: HyperparamConfig) -> float:
            val = getattr(cfg, metric, None)
            if val is None:
                return float("-inf")
            if metric == "max_drawdown":
                # Higher (less negative) drawdown is better
                return val
            return val

        reverse = metric != "max_drawdown"
        sorted_results = sorted(self.results, key=_key, reverse=reverse)
        return sorted_results[:k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, path: str = "hyperparam_results.json") -> None:
        """Save all results to a JSON file.

        Args:
            path: Output file path.
        """
        data = {
            "n_results": len(self.results),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        log.info("hyperparam_search.saved", path=path, n=len(self.results))

    def load_results(self, path: str = "hyperparam_results.json") -> None:
        """Load previously saved results from a JSON file.

        Existing results in :attr:`results` are cleared and replaced.

        Args:
            path: Path to a JSON file written by :meth:`save_results`.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.results = [HyperparamConfig.from_dict(r) for r in data.get("results", [])]
        self._sort_results()
        log.info("hyperparam_search.loaded", path=path, n=len(self.results))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary_table(self) -> str:
        """Render an ASCII table of all results sorted by Sharpe ratio.

        Returns:
            A multi-line string formatted as an ASCII table.
        """
        if not self.results:
            return "No results yet. Run grid_search() or random_search() first."

        # Collect all param keys present across all results
        all_param_keys: List[str] = []
        for r in self.results:
            for k in r.params:
                if k not in all_param_keys:
                    all_param_keys.append(k)

        headers = ["#", "config_id"] + all_param_keys + ["sharpe", "max_dd", "return", "secs"]
        col_widths = [max(3, len(h)) for h in headers]

        rows: List[List[str]] = []
        for rank, cfg in enumerate(self.results, start=1):
            row = [
                str(rank),
                cfg.config_id[:12],
            ]
            for k in all_param_keys:
                v = cfg.params.get(k, "")
                row.append(str(v)[:14])
            row.append(f"{cfg.sharpe:.4f}" if cfg.sharpe is not None else "N/A")
            row.append(f"{cfg.max_drawdown:.4f}" if cfg.max_drawdown is not None else "N/A")
            row.append(f"{cfg.total_return:.4f}" if cfg.total_return is not None else "N/A")
            row.append(f"{cfg.eval_time_secs:.1f}s" if cfg.eval_time_secs is not None else "N/A")
            rows.append(row)

        # Compute column widths
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(cell))

        sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
        lines = [sep, header_line, sep]
        for row in rows:
            padded = [
                (row[i] if i < len(row) else "").ljust(col_widths[i])
                for i in range(len(col_widths))
            ]
            lines.append("| " + " | ".join(padded) + " |")
        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_single(self, params: Dict[str, Any]) -> HyperparamConfig:
        """Evaluate one configuration and return a :class:`HyperparamConfig`.

        Args:
            params: Hyperparameter dict to evaluate.

        Returns:
            A populated :class:`HyperparamConfig`.
        """
        config_id = str(uuid.uuid4())[:8]
        log.info("hyperparam_search.eval_start", config_id=config_id, params=params)
        t0 = time.perf_counter()
        try:
            metrics = self.eval_fn(params)
        except Exception as exc:
            log.error(
                "hyperparam_search.eval_failed",
                config_id=config_id,
                error=str(exc),
            )
            metrics = {}
        elapsed = time.perf_counter() - t0
        cfg = HyperparamConfig(
            config_id=config_id,
            params=params,
            sharpe=metrics.get("sharpe"),
            max_drawdown=metrics.get("max_drawdown"),
            total_return=metrics.get("total_return"),
            eval_time_secs=round(elapsed, 3),
        )
        log.info(
            "hyperparam_search.eval_done",
            config_id=config_id,
            sharpe=cfg.sharpe,
            max_drawdown=cfg.max_drawdown,
            total_return=cfg.total_return,
            elapsed_secs=cfg.eval_time_secs,
        )
        return cfg

    def _evaluate_all(
        self, configs: List[Dict[str, Any]]
    ) -> List[HyperparamConfig]:
        """Evaluate all configs, optionally in parallel.

        Args:
            configs: List of parameter dicts.

        Returns:
            List of :class:`HyperparamConfig` objects in the same order.
        """
        if self.n_jobs == 1:
            return [self._evaluate_single(p) for p in configs]

        # Parallel evaluation via ThreadPoolExecutor
        import concurrent.futures

        results: List[HyperparamConfig] = [None] * len(configs)  # type: ignore[list-item]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_idx = {
                executor.submit(self._evaluate_single, params): i
                for i, params in enumerate(configs)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    log.error(
                        "hyperparam_search.parallel_eval_failed",
                        index=idx,
                        error=str(exc),
                    )
                    results[idx] = HyperparamConfig(
                        config_id=f"ERR-{idx}",
                        params=configs[idx],
                    )
        return results

    def _sort_results(self) -> None:
        """Sort :attr:`results` by Sharpe ratio descending in place."""
        self.results.sort(
            key=lambda c: c.sharpe if c.sharpe is not None else float("-inf"),
            reverse=True,
        )
