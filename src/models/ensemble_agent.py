"""Ensemble of multiple RL agents for robust action selection."""
import math
from typing import List, Optional, Dict, Any


class EnsembleMethod:
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    STACKING = "stacking"
    BEST_OF_N = "best_of_n"


class AgentWrapper:
    """Wraps an RL agent with performance tracking."""

    def __init__(self, agent: Any, name: str = "agent"):
        self.agent = agent
        self.name = name
        self._rewards: List[float] = []
        self._n_predictions: int = 0

    def predict(self, state: list) -> int:
        """Call underlying agent predict and return discrete action."""
        self._n_predictions += 1
        if hasattr(self.agent, "predict"):
            result = self.agent.predict(state)
            if isinstance(result, (list, tuple)):
                return int(result[0])
            return int(result)
        return 0

    def record_reward(self, reward: float) -> None:
        self._rewards.append(reward)

    @property
    def mean_reward(self) -> float:
        if not self._rewards:
            return 0.0
        return sum(self._rewards) / len(self._rewards)

    @property
    def n_predictions(self) -> int:
        return self._n_predictions


class EnsembleAgent:
    """Ensemble aggregation over multiple RL agents."""

    def __init__(self, agents: list, method: str = EnsembleMethod.WEIGHTED_AVERAGE, weights: Optional[list] = None):
        self.wrappers: List[AgentWrapper] = []
        for i, ag in enumerate(agents):
            if isinstance(ag, AgentWrapper):
                self.wrappers.append(ag)
            else:
                self.wrappers.append(AgentWrapper(ag, name=f"agent_{i}"))
        self.method = method
        if weights is not None:
            self.weights = list(weights)
        else:
            self.weights = [1.0] * len(self.wrappers)
        self._ema_alpha = 0.1  # EMA smoothing for weight updates

    def predict(self, state: list) -> int:
        """Return ensemble prediction for the given state."""
        if not self.wrappers:
            return 0
        predictions = [w.predict(state) for w in self.wrappers]
        if self.method == EnsembleMethod.MAJORITY_VOTE:
            return self.majority_vote(predictions)
        elif self.method == EnsembleMethod.WEIGHTED_AVERAGE:
            # For discrete actions: use weighted vote (no probabilities available)
            return self._weighted_vote(predictions)
        elif self.method == EnsembleMethod.BEST_OF_N:
            # Return prediction from the agent with the highest weight
            best_idx = self.weights.index(max(self.weights))
            return predictions[best_idx]
        elif self.method == EnsembleMethod.STACKING:
            return self.majority_vote(predictions)
        return self.majority_vote(predictions)

    def majority_vote(self, predictions: list) -> int:
        """Simple majority vote over discrete action predictions."""
        if not predictions:
            return 0
        counts: Dict[int, float] = {}
        for pred in predictions:
            action = int(pred)
            counts[action] = counts.get(action, 0) + 1
        return max(counts, key=lambda a: counts[a])

    def _weighted_vote(self, predictions: list) -> int:
        """Weighted vote using current agent weights."""
        if not predictions:
            return 0
        scores: Dict[int, float] = {}
        total_w = sum(abs(w) for w in self.weights) or 1.0
        for pred, w in zip(predictions, self.weights):
            action = int(pred)
            scores[action] = scores.get(action, 0.0) + (abs(w) / total_w)
        return max(scores, key=lambda a: scores[a])

    def weighted_average(self, predictions: list, probs: list) -> int:
        """Weighted softmax over action probability vectors from each agent.

        Args:
            predictions: list of action indices (unused when probs given)
            probs: list of probability vectors (one per agent)
        """
        if not probs:
            return self.majority_vote(predictions)
        n_actions = len(probs[0]) if probs else 1
        agg = [0.0] * n_actions
        total_w = sum(abs(w) for w in self.weights) or 1.0
        for prob_vec, w in zip(probs, self.weights):
            for i, p in enumerate(prob_vec):
                if i < n_actions:
                    agg[i] += p * abs(w) / total_w
        return int(agg.index(max(agg)))

    def update_weights(self, rewards: list) -> None:
        """EMA weight update based on recent rewards."""
        for i, reward in enumerate(rewards):
            if i < len(self.weights):
                # EMA: w_new = (1-alpha)*w_old + alpha*reward
                self.weights[i] = (1.0 - self._ema_alpha) * self.weights[i] + self._ema_alpha * reward
                if i < len(self.wrappers):
                    self.wrappers[i].record_reward(reward)
        # Normalise weights to be non-negative for voting
        min_w = min(self.weights)
        if min_w < 0:
            self.weights = [w - min_w + 1e-6 for w in self.weights]

    def diversity_score(self, states: list) -> float:
        """Disagreement rate across agents on a list of states."""
        if not self.wrappers or not states:
            return 0.0
        n_disagree = 0
        total = 0
        for state in states:
            preds = [w.predict(state) for w in self.wrappers]
            n_pairs = len(preds) * (len(preds) - 1) // 2
            if n_pairs == 0:
                continue
            total += n_pairs
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    if preds[i] != preds[j]:
                        n_disagree += 1
        return n_disagree / total if total > 0 else 0.0

    def add_agent(self, agent: Any, weight: float = 1.0) -> None:
        """Add a new agent to the ensemble."""
        if isinstance(agent, AgentWrapper):
            self.wrappers.append(agent)
        else:
            self.wrappers.append(AgentWrapper(agent, name=f"agent_{len(self.wrappers)}"))
        self.weights.append(weight)

    def remove_agent(self, idx: int) -> None:
        """Remove agent at index idx."""
        if 0 <= idx < len(self.wrappers):
            self.wrappers.pop(idx)
            self.weights.pop(idx)

    def performance_summary(self) -> dict:
        """Return a summary of each agent's performance."""
        summary = {}
        for i, wrapper in enumerate(self.wrappers):
            summary[wrapper.name] = {
                "weight": self.weights[i] if i < len(self.weights) else 1.0,
                "mean_reward": wrapper.mean_reward,
                "n_predictions": wrapper.n_predictions,
            }
        summary["ensemble_method"] = self.method
        summary["n_agents"] = len(self.wrappers)
        return summary
