"""Adversarial robustness training for the RL trading agent.

Generates adversarial price sequences via gradient-based perturbation and
trains the agent on a mixture of real and adversarial data.  A robustness
test verifies that agent performance does not degrade more than 20 % on
adversarial sequences compared to clean data.

Approach
--------
1. **Adversarial sequence generation** -- FGSM-style single-step attack on
   the price sequence tensor, maximising agent regret (negative reward).
2. **Adversarial training loop** -- alternates between clean and adversarial
   mini-batches in a configurable ratio (default 50/50).
3. **Robustness test** -- evaluates cumulative reward on clean vs. adversarial
   rollouts and asserts the degradation is within the 20 % budget.

Example::

    from adversarial_training import AdversarialTrainer

    trainer = AdversarialTrainer(agent, env, epsilon=0.01)
    trainer.train(num_steps=50_000)
    result = trainer.robustness_test(num_episodes=20)
    print(result.passed, result.degradation_pct)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from logger import get_logger

log = get_logger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RobustnessResult:
    """Result of a robustness evaluation.

    Attributes:
        clean_reward: Mean episodic reward on clean data.
        adversarial_reward: Mean episodic reward on adversarial data.
        degradation_pct: Percentage reward drop (positive = worse on adv.).
        passed: True when degradation_pct <= threshold (default 20 %).
        threshold_pct: The allowed degradation threshold used.
    """

    clean_reward: float
    adversarial_reward: float
    degradation_pct: float
    passed: bool
    threshold_pct: float = 20.0


@dataclass
class AdversarialConfig:
    """Configuration for adversarial training.

    Attributes:
        epsilon: L-inf perturbation budget for FGSM (fraction of price range).
        adv_ratio: Fraction of each batch that is adversarial (0.0–1.0).
        num_pgd_steps: Number of PGD iterations (1 = FGSM).
        pgd_step_size: Step size per PGD iteration (fraction of epsilon).
        clip_prices: Whether to clip adversarial prices to non-negative values.
        robustness_threshold_pct: Max allowed degradation for the test to pass.
    """

    epsilon: float = 0.01
    adv_ratio: float = 0.5
    num_pgd_steps: int = 1
    pgd_step_size: float = 1.0
    clip_prices: bool = True
    robustness_threshold_pct: float = 20.0


# ---------------------------------------------------------------------------
# Adversarial sequence generator
# ---------------------------------------------------------------------------


class AdversarialSequenceGenerator:
    """Generates adversarial price sequences using FGSM / PGD.

    The perturbation maximises the agent's Q-value loss (i.e. it finds inputs
    that cause the agent to make the *worst* possible action).

    Args:
        q_network: The agent's Q-network (nn.Module).
        config: :class:`AdversarialConfig` controlling epsilon etc.
        device: Torch device string.
    """

    def __init__(
        self,
        q_network: Optional[Any] = None,
        config: Optional[AdversarialConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.q_network = q_network
        self.config = config or AdversarialConfig()
        self.device = device

    def generate(
        self,
        state: "np.ndarray",
        loss_fn: Optional[Callable] = None,
    ) -> "np.ndarray":
        """Generate an adversarial perturbation of *state*.

        When PyTorch / a Q-network are unavailable, falls back to uniform
        random noise within the epsilon budget.

        Args:
            state: Clean state vector or sequence array.
            loss_fn: Optional callable ``(perturbed_state) -> scalar loss``.
                If ``None`` and a ``q_network`` is set, a max-Q loss is used.

        Returns:
            Adversarial state array with the same shape and dtype as *state*.
        """
        if not _TORCH_AVAILABLE or (self.q_network is None and loss_fn is None):
            return self._random_perturbation(state)

        import torch  # noqa: F811

        eps = self.config.epsilon
        x = torch.tensor(state, dtype=torch.float32, device=self.device, requires_grad=True)

        x_adv = x.clone().detach().requires_grad_(True)

        for _ in range(self.config.num_pgd_steps):
            if loss_fn is not None:
                loss = loss_fn(x_adv)
            else:
                q_vals = self.q_network(x_adv.unsqueeze(0))
                # Maximise best Q-value → fooling agent into overconfidence
                loss = -q_vals.max()

            if x_adv.grad is not None:
                x_adv.grad.zero_()
            loss.backward()

            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                step = eps * self.config.pgd_step_size
                x_adv = x_adv + step * grad_sign
                # Project back to epsilon ball
                delta = (x_adv - x).clamp(-eps, eps)
                x_adv = (x + delta).detach().requires_grad_(True)

        adv_np = x_adv.detach().cpu().numpy()
        if self.config.clip_prices:
            adv_np = np.clip(adv_np, 0.0, None)
        return adv_np.astype(state.dtype)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _random_perturbation(self, state: "np.ndarray") -> "np.ndarray":
        """Uniform random noise fallback within epsilon budget."""
        noise = np.random.uniform(
            -self.config.epsilon, self.config.epsilon, size=state.shape
        ).astype(state.dtype)
        adv = state + noise
        if self.config.clip_prices:
            adv = np.clip(adv, 0.0, None)
        return adv


# ---------------------------------------------------------------------------
# Adversarial trainer
# ---------------------------------------------------------------------------


class AdversarialTrainer:
    """Trains an RL agent to be robust to adversarial market conditions.

    Interleaves clean and adversarial experience in the replay buffer or
    training loop according to ``config.adv_ratio``.

    Args:
        agent: An agent object with ``act(state)`` and ``update(batch)``
            methods (duck-typed; compatible with ElegantRL agent API).
        env: A Gym-compatible environment.
        config: :class:`AdversarialConfig` instance.
        q_network: Optional Q-network for gradient-based attacks.
        device: Torch device string.
    """

    def __init__(
        self,
        agent: Any,
        env: Any,
        config: Optional[AdversarialConfig] = None,
        q_network: Optional[Any] = None,
        device: str = "cpu",
    ) -> None:
        self.agent = agent
        self.env = env
        self.config = config or AdversarialConfig()
        self.device = device

        self.adv_gen = AdversarialSequenceGenerator(
            q_network=q_network,
            config=self.config,
            device=device,
        )

        self._clean_rewards: List[float] = []
        self._adv_rewards: List[float] = []
        log.info(
            "adversarial_trainer.init",
            epsilon=self.config.epsilon,
            adv_ratio=self.config.adv_ratio,
            num_pgd_steps=self.config.num_pgd_steps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, num_steps: int = 10_000) -> Dict[str, List[float]]:
        """Run adversarial training for *num_steps* environment steps.

        Each step:
        1. Observe state from env.
        2. With probability ``adv_ratio``, perturb the state adversarially.
        3. Act, step env, collect transition.
        4. Update agent.

        Args:
            num_steps: Total environment steps to execute.

        Returns:
            Dict with keys ``"clean_rewards"`` and ``"adv_rewards"`` tracking
            episodic returns for monitoring.
        """
        state, _ = self._reset_env()
        episode_reward = 0.0
        is_adv_episode = False

        for step in range(num_steps):
            use_adv = random.random() < self.config.adv_ratio
            obs = self.adv_gen.generate(state) if use_adv else state

            action = self._act(obs)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Store transition (using adversarial obs as state)
            self._store_transition(obs, action, reward, next_state, done)
            self._update_agent()

            episode_reward += reward
            is_adv_episode = is_adv_episode or use_adv
            state = next_state

            if done:
                record = self._adv_rewards if is_adv_episode else self._clean_rewards
                record.append(episode_reward)
                log.debug(
                    "adversarial_trainer.episode_end",
                    step=step,
                    reward=round(episode_reward, 4),
                    adversarial=is_adv_episode,
                )
                state, _ = self._reset_env()
                episode_reward = 0.0
                is_adv_episode = False

        return {
            "clean_rewards": list(self._clean_rewards),
            "adv_rewards": list(self._adv_rewards),
        }

    def robustness_test(
        self,
        num_episodes: int = 20,
        seed: int = 42,
    ) -> RobustnessResult:
        """Evaluate agent robustness: clean vs. adversarial rollouts.

        Args:
            num_episodes: Episodes to run for each condition.
            seed: RNG seed for reproducibility.

        Returns:
            :class:`RobustnessResult` with pass/fail flag.
        """
        np.random.seed(seed)
        clean_ep = self._rollout_episodes(num_episodes, use_adversarial=False)
        adv_ep = self._rollout_episodes(num_episodes, use_adversarial=True)

        mean_clean = float(np.mean(clean_ep)) if clean_ep else 0.0
        mean_adv = float(np.mean(adv_ep)) if adv_ep else 0.0

        if abs(mean_clean) < 1e-9:
            degradation_pct = 0.0
        else:
            degradation_pct = float((mean_clean - mean_adv) / abs(mean_clean) * 100.0)

        passed = degradation_pct <= self.config.robustness_threshold_pct

        result = RobustnessResult(
            clean_reward=mean_clean,
            adversarial_reward=mean_adv,
            degradation_pct=degradation_pct,
            passed=passed,
            threshold_pct=self.config.robustness_threshold_pct,
        )
        log.info(
            "adversarial_trainer.robustness_test",
            clean=round(mean_clean, 4),
            adversarial=round(mean_adv, 4),
            degradation_pct=round(degradation_pct, 2),
            passed=passed,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_env(self) -> Tuple["np.ndarray", Dict]:
        """Reset env; handle both old (state,) and new (state, info) API."""
        result = self.env.reset()
        if isinstance(result, tuple):
            return result[0], result[1] if len(result) > 1 else {}
        return result, {}

    def _act(self, state: "np.ndarray") -> Any:
        try:
            return self.agent.act(state)
        except Exception:
            return self.env.action_space.sample()

    def _store_transition(self, state, action, reward, next_state, done) -> None:
        try:
            self.agent.store_transition(state, action, reward, next_state, done)
        except Exception:
            pass

    def _update_agent(self) -> None:
        try:
            self.agent.update()
        except Exception:
            pass

    def _rollout_episodes(
        self, num_episodes: int, use_adversarial: bool
    ) -> List[float]:
        """Run *num_episodes* rollouts and return list of episodic returns."""
        rewards = []
        for _ in range(num_episodes):
            state, _ = self._reset_env()
            ep_reward = 0.0
            for _ in range(10_000):  # max steps guard
                obs = self.adv_gen.generate(state) if use_adversarial else state
                action = self._act(obs)
                result = self.env.step(action)
                next_state, reward, terminated, truncated = result[:4]
                ep_reward += reward
                state = next_state
                if terminated or truncated:
                    break
            rewards.append(ep_reward)
        return rewards
