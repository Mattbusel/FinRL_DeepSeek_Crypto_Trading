"""
Policy Gradient reinforcement learning agents.

Implements REINFORCE (vanilla policy gradient) and an Actor-Critic agent
using pure Python with no external ML library dependencies.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple


# ─── UTILITY FUNCTIONS ───────────────────────────────────────────────────────

def softmax(x: List[float]) -> List[float]:
    """Numerically stable softmax.

    Args:
        x: Input logits.

    Returns:
        Probability distribution over actions.
    """
    max_x = max(x)
    exps = [math.exp(v - max_x) for v in x]
    total = sum(exps)
    return [e / total for e in exps]


def log_softmax(x: List[float]) -> List[float]:
    """Log-softmax for numerical stability.

    Args:
        x: Input logits.

    Returns:
        Log-probabilities.
    """
    max_x = max(x)
    log_sum_exp = math.log(sum(math.exp(v - max_x) for v in x)) + max_x
    return [v - log_sum_exp for v in x]


# ─── POLICY GRADIENT AGENT (REINFORCE) ───────────────────────────────────────

class PolicyGradientAgent:
    """REINFORCE (Monte Carlo policy gradient) agent.

    Uses a linear policy: logits = W @ state + b.
    Weights are updated by gradient ascent on the log-probability of
    actions weighted by discounted returns.

    Args:
        state_dim:  Dimensionality of the state vector.
        action_dim: Number of discrete actions.
        lr:         Learning rate for gradient ascent.
        gamma:      Discount factor for future rewards.
        seed:       RNG seed for reproducible action sampling.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.01,
        gamma: float = 0.99,
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self._rng = random.Random(seed)

        # Linear policy parameters: W[action][state] and b[action]
        scale = 1.0 / math.sqrt(state_dim)
        self.W: List[List[float]] = [
            [self._rng.gauss(0.0, scale) for _ in range(state_dim)]
            for _ in range(action_dim)
        ]
        self.b: List[float] = [0.0] * action_dim

        # Episode buffer
        self._states: List[List[float]] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []

    def _logits(self, state: List[float]) -> List[float]:
        """Compute action logits from state."""
        return [
            sum(self.W[a][s] * state[s] for s in range(self.state_dim)) + self.b[a]
            for a in range(self.action_dim)
        ]

    def policy(self, state: List[float]) -> List[float]:
        """Compute action probabilities via softmax over linear features.

        Args:
            state: State vector of length state_dim.

        Returns:
            Probability distribution over actions.
        """
        return softmax(self._logits(state))

    def select_action(self, state: List[float], seed: int) -> int:
        """Sample an action from the policy distribution.

        Args:
            state: State vector.
            seed:  RNG seed for this selection (for reproducibility).

        Returns:
            Selected action index.
        """
        rng = random.Random(seed)
        probs = self.policy(state)
        r = rng.random()
        cumulative = 0.0
        for a, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return a
        return self.action_dim - 1  # fallback

    def discount_rewards(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns G_t = Σ_{k≥0} γ^k r_{t+k}.

        Args:
            rewards: List of per-step rewards.

        Returns:
            List of discounted returns (same length).
        """
        returns = [0.0] * len(rewards)
        g = 0.0
        for t in reversed(range(len(rewards))):
            g = rewards[t] + self.gamma * g
            returns[t] = g
        return returns

    def save_episode(
        self,
        states: List[List[float]],
        actions: List[int],
        rewards: List[float],
    ) -> None:
        """Store one episode for a subsequent update call."""
        self._states = list(states)
        self._actions = list(actions)
        self._rewards = list(rewards)

    def update(
        self,
        states: List[List[float]],
        actions: List[int],
        rewards: List[float],
    ) -> None:
        """REINFORCE gradient update.

        ∇J ≈ Σ_t G_t ∇ log π(a_t | s_t)

        Args:
            states:  List of state vectors (one per step).
            actions: List of actions taken.
            rewards: List of rewards received.
        """
        returns = self.discount_rewards(rewards)

        # Normalise returns (reduces variance)
        mean_g = sum(returns) / max(len(returns), 1)
        var_g = sum((g - mean_g) ** 2 for g in returns) / max(len(returns), 1)
        std_g = math.sqrt(var_g) + 1e-8
        returns = [(g - mean_g) / std_g for g in returns]

        for state, action, g in zip(states, actions, returns):
            log_probs = log_softmax(self._logits(state))
            # Gradient of log π(a|s) w.r.t. W[a] and b[a]
            probs = softmax(self._logits(state))
            for a in range(self.action_dim):
                indicator = 1.0 if a == action else 0.0
                grad_logp = indicator - probs[a]  # ∂ log π(action|s) / ∂ logit_a
                for s in range(self.state_dim):
                    self.W[a][s] += self.lr * g * grad_logp * state[s]
                self.b[a] += self.lr * g * grad_logp

    def train_episode(self) -> float:
        """Run one update using the stored episode buffer.

        Returns:
            Total undiscounted reward from the stored episode.
        """
        if not self._rewards:
            return 0.0
        self.update(self._states, self._actions, self._rewards)
        total = sum(self._rewards)
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        return total


# ─── ACTOR-CRITIC AGENT ──────────────────────────────────────────────────────

class ActorCriticAgent:
    """One-step Actor-Critic agent (TD(0) critic).

    The actor is a linear softmax policy.
    The critic is a linear value function V(s) = w_c · s.

    Args:
        state_dim:   Dimensionality of the state vector.
        action_dim:  Number of discrete actions.
        lr_actor:    Actor learning rate.
        lr_critic:   Critic learning rate.
        gamma:       Discount factor.
        seed:        RNG seed for reproducible sampling.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 0.001,
        lr_critic: float = 0.001,
        gamma: float = 0.99,
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self._rng = random.Random(seed)

        scale = 1.0 / math.sqrt(state_dim)
        # Actor weights
        self.W_actor: List[List[float]] = [
            [self._rng.gauss(0.0, scale) for _ in range(state_dim)]
            for _ in range(action_dim)
        ]
        self.b_actor: List[float] = [0.0] * action_dim

        # Critic weights
        self.w_critic: List[float] = [self._rng.gauss(0.0, scale) for _ in range(state_dim)]

    def _actor_logits(self, state: List[float]) -> List[float]:
        return [
            sum(self.W_actor[a][s] * state[s] for s in range(self.state_dim)) + self.b_actor[a]
            for a in range(self.action_dim)
        ]

    def actor_forward(self, state: List[float]) -> List[float]:
        """Compute action probabilities.

        Args:
            state: State vector.

        Returns:
            Probability distribution over actions.
        """
        return softmax(self._actor_logits(state))

    def critic_forward(self, state: List[float]) -> float:
        """Estimate state value V(s).

        Args:
            state: State vector.

        Returns:
            Scalar state-value estimate.
        """
        return sum(self.w_critic[s] * state[s] for s in range(self.state_dim))

    def select_action(self, state: List[float], seed: int) -> int:
        """Sample an action from the actor policy.

        Args:
            state: State vector.
            seed:  Per-step RNG seed.

        Returns:
            Selected action index.
        """
        rng = random.Random(seed)
        probs = self.actor_forward(state)
        r = rng.random()
        cumulative = 0.0
        for a, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return a
        return self.action_dim - 1

    def update(
        self,
        state: List[float],
        action: int,
        reward: float,
        next_state: List[float],
        done: bool,
    ) -> Dict[str, float]:
        """One-step Actor-Critic update.

        TD error: δ = r + γ V(s') - V(s)  (or r - V(s) if done)
        Critic update: w_c += lr_c * δ * s
        Actor update:  W[a] += lr_a * δ * ∇ log π(a|s)

        Args:
            state:      Current state s.
            action:     Action taken a.
            reward:     Reward r received.
            next_state: Next state s'.
            done:       Whether the episode ended.

        Returns:
            Dict with 'actor_loss' and 'critic_loss' (scalar estimates).
        """
        v_s = self.critic_forward(state)
        v_s_next = 0.0 if done else self.critic_forward(next_state)

        td_error = reward + self.gamma * v_s_next - v_s

        # Critic update
        for s in range(self.state_dim):
            self.w_critic[s] += self.lr_critic * td_error * state[s]

        # Actor update
        probs = self.actor_forward(state)
        for a in range(self.action_dim):
            indicator = 1.0 if a == action else 0.0
            grad_logp = indicator - probs[a]
            for s in range(self.state_dim):
                self.W_actor[a][s] += self.lr_actor * td_error * grad_logp * state[s]
            self.b_actor[a] += self.lr_actor * td_error * grad_logp

        actor_loss = -math.log(max(probs[action], 1e-8)) * td_error
        critic_loss = td_error ** 2

        return {"actor_loss": actor_loss, "critic_loss": critic_loss}


__all__ = ["softmax", "log_softmax", "PolicyGradientAgent", "ActorCriticAgent"]
