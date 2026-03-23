"""In-memory ML model versioning and registry for the FinRL trading system.

Provides ModelVersion dataclass and ModelRegistry with champion selection,
comparison, promotion, and rollback capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class ModelVersion:
    """A versioned model entry in the registry.

    Attributes
    ----------
    model_id : str
        Logical model identifier (e.g. "dqn_btc_agent").
    version : str
        Semantic or arbitrary version string (e.g. "v1.2.3").
    created_at : datetime
        UTC creation timestamp.
    metrics : dict
        Performance metrics keyed by name (e.g. {"sharpe_ratio": 1.8}).
    hyperparams : dict
        Hyperparameter dict used to train this version.
    file_path : str or None
        Optional path to persisted model weights / checkpoint.
    tags : list[str]
        Arbitrary tags (e.g. ["staging", "production", "archived"]).
    """

    model_id: str
    version: str
    created_at: datetime
    metrics: dict[str, Any]
    hyperparams: dict[str, Any]
    file_path: Optional[str] = None
    tags: list[str] = field(default_factory=list)


class ModelRegistry:
    """In-memory model registry supporting versioning, promotion, and rollback.

    Parameters
    ----------
    champion_metric : str
        Name of the metric used to select the champion version
        (default "sharpe_ratio"; higher is better).
    """

    def __init__(self, champion_metric: str = "sharpe_ratio") -> None:
        self.champion_metric = champion_metric
        # _store: model_id → list of ModelVersion (ordered by creation time)
        self._store: dict[str, list[ModelVersion]] = {}

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def register(
        self,
        model_id: str,
        version: str,
        metrics: dict[str, Any],
        hyperparams: dict[str, Any],
        tags: Optional[list[str]] = None,
        file_path: Optional[str] = None,
    ) -> ModelVersion:
        """Register a new model version and return the created entry."""
        mv = ModelVersion(
            model_id=model_id,
            version=version,
            created_at=datetime.now(tz=timezone.utc),
            metrics=metrics,
            hyperparams=hyperparams,
            file_path=file_path,
            tags=list(tags) if tags else [],
        )
        self._store.setdefault(model_id, []).append(mv)
        return mv

    def get(self, model_id: str, version: Optional[str] = None) -> ModelVersion:
        """Retrieve a model version.

        Parameters
        ----------
        model_id : str
            The model to look up.
        version : str or None
            Specific version string. If None, returns the latest (most recently
            registered) version.

        Raises
        ------
        KeyError
            If the model_id or version is not found.
        """
        versions = self._store.get(model_id)
        if not versions:
            raise KeyError(f"No versions found for model_id='{model_id}'")
        if version is None:
            return versions[-1]
        for mv in versions:
            if mv.version == version:
                return mv
        raise KeyError(f"Version '{version}' not found for model_id='{model_id}'")

    def list_versions(self, model_id: str) -> list[ModelVersion]:
        """Return all registered versions for a model, ordered by creation time."""
        return list(self._store.get(model_id, []))

    # ------------------------------------------------------------------
    # Champion selection
    # ------------------------------------------------------------------

    def champion(self, model_id: str) -> ModelVersion:
        """Return the version with the best champion metric value.

        If the metric is missing for some versions, those are ranked last.

        Raises
        ------
        KeyError
            If no versions are registered for model_id.
        """
        versions = self._store.get(model_id)
        if not versions:
            raise KeyError(f"No versions found for model_id='{model_id}'")
        return max(
            versions,
            key=lambda mv: mv.metrics.get(self.champion_metric, float("-inf")),
        )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self, model_id: str, v1: str, v2: str, metric: str
    ) -> dict[str, Any]:
        """Compare two versions on a specific metric.

        Returns a dict with keys:
        - 'v1_value': metric value for v1
        - 'v2_value': metric value for v2
        - 'winner': version string of the better performer (higher is better)
        - 'delta': v2_value - v1_value
        """
        mv1 = self.get(model_id, v1)
        mv2 = self.get(model_id, v2)
        val1 = mv1.metrics.get(metric, float("-inf"))
        val2 = mv2.metrics.get(metric, float("-inf"))
        winner = v1 if val1 >= val2 else v2
        return {
            "v1": v1,
            "v2": v2,
            "metric": metric,
            "v1_value": val1,
            "v2_value": val2,
            "winner": winner,
            "delta": val2 - val1,
        }

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote(self, model_id: str, version: str, stage: str) -> ModelVersion:
        """Tag a version with the given lifecycle stage.

        Common stages: "staging", "production", "archived".
        Promoting to "production" removes the "production" tag from all other
        versions of the same model to maintain a single production champion.

        Returns the updated ModelVersion.
        """
        mv = self.get(model_id, version)
        if stage == "production":
            # Remove production tag from all other versions.
            for other in self._store.get(model_id, []):
                if other.version != version and "production" in other.tags:
                    other.tags.remove("production")
        if stage not in mv.tags:
            mv.tags.append(stage)
        return mv

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, model_id: str) -> ModelVersion:
        """Revert to the previous production version.

        Finds the current production version, removes its "production" tag,
        and promotes the immediately preceding version (by creation order) to
        "production".

        Returns the newly promoted version.

        Raises
        ------
        KeyError
            If no versions are registered.
        ValueError
            If there is no current production version or no previous version
            to roll back to.
        """
        versions = self._store.get(model_id, [])
        if not versions:
            raise KeyError(f"No versions found for model_id='{model_id}'")

        # Find current production version.
        prod_versions = [mv for mv in versions if "production" in mv.tags]
        if not prod_versions:
            raise ValueError(
                f"No production version found for model_id='{model_id}'. "
                "Cannot roll back."
            )
        # If multiple are tagged production (shouldn't happen), take the latest.
        current_prod = prod_versions[-1]
        current_idx = versions.index(current_prod)

        if current_idx == 0:
            raise ValueError(
                f"model_id='{model_id}' has no previous version to roll back to."
            )

        # Remove production tag from current.
        current_prod.tags.remove("production")

        # Promote previous version.
        previous = versions[current_idx - 1]
        return self.promote(model_id, previous.version, "production")
