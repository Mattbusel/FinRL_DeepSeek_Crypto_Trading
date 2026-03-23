"""Continuous learning pipeline for online adaptation during live trading.

Provides four cooperating components:

1. ``ExperienceReplay``    -- ring-buffer of (state, action, reward, next_state,
                              done) tuples gathered from live trading.
2. ``OnlineFinetuner``     -- periodically fine-tunes a DQN agent on recent
                              experience using small gradient steps.
3. ``ConceptDriftDetector``-- ADWIN-style statistical test that signals when
                              the reward distribution has shifted, indicating a
                              market regime change.
4. ``ModelVersionManager`` -- checkpoint saving/loading with performance tracking
                              and automatic rollback to the best version.

All heavy lifting runs in a background :class:`threading.Thread` so the main
trading loop is never blocked.

Example::

    from continuous_learner import ContinuousLearner
    from erl_agent import AgentD3QN

    agent = AgentD3QN(net_dims=[128, 128], state_dim=32, action_dim=3, gpu_id=-1)
    learner = ContinuousLearner(agent=agent, state_dim=32, action_dim=3)
    learner.start()

    # In the trading loop:
    learner.record(state, action, reward, next_state, done=False)

    # When finished:
    learner.stop()
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from exceptions import ModelError
from logger import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

_StateArray = npt.NDArray[np.float32]


# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    """One (s, a, r, s', done) tuple from a live trade step.

    Attributes:
        state: Observation vector at time *t*.
        action: Integer action index taken.
        reward: Scalar reward received.
        next_state: Observation vector at time *t+1*.
        done: True if the episode ended at this step.
    """

    state: _StateArray
    action: int
    reward: float
    next_state: _StateArray
    done: bool


class ExperienceReplay:
    """Thread-safe ring buffer of :class:`Transition` tuples.

    Args:
        capacity: Maximum number of transitions to keep.
        state_dim: Dimensionality of the state vector.
    """

    def __init__(self, capacity: int = 100_000, state_dim: int = 32) -> None:
        self._capacity = capacity
        self._state_dim = state_dim
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        state: _StateArray,
        action: int,
        reward: float,
        next_state: _StateArray,
        done: bool = False,
    ) -> None:
        """Add a transition to the buffer (thread-safe).

        Args:
            state: Current observation.
            action: Discrete action taken.
            reward: Reward received.
            next_state: Next observation.
            done: Episode-terminal flag.
        """
        transition = Transition(
            state=np.asarray(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            done=bool(done),
        )
        with self._lock:
            self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """Return a uniformly random mini-batch (thread-safe).

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of :class:`Transition` objects.

        Raises:
            ValueError: If the buffer contains fewer transitions than
                *batch_size*.
        """
        with self._lock:
            n = len(self._buffer)
            if n < batch_size:
                raise ValueError(
                    f"Buffer has {n} transitions; {batch_size} requested."
                )
            indices = np.random.choice(n, size=batch_size, replace=False)
            buf_list = list(self._buffer)
            return [buf_list[i] for i in indices]

    def as_arrays(
        self, batch: list[Transition]
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Convert a list of transitions into stacked numpy arrays.

        Args:
            batch: List of :class:`Transition` objects.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones), each as a
            numpy array.
        """
        states = np.stack([t.state for t in batch]).astype(np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch]).astype(np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


# ---------------------------------------------------------------------------
# ADWIN Concept Drift Detector
# ---------------------------------------------------------------------------


@dataclass
class DriftEvent:
    """Record of a single detected drift event.

    Attributes:
        detected_at: Wall-clock time of detection (``time.time()``).
        window_mean_before: Mean reward over the stable window before drift.
        window_mean_after: Mean reward in the new window after drift.
        magnitude: Absolute difference between the two means.
        n_samples: Total samples seen at detection time.
    """

    detected_at: float
    window_mean_before: float
    window_mean_after: float
    magnitude: float
    n_samples: int


class ConceptDriftDetector:
    """ADWIN-inspired adaptive windowing concept drift detector.

    Monitors a stream of scalar reward signals.  When the statistical mean of
    the reward distribution in the oldest sub-window significantly differs from
    that in the most recent sub-window the detector fires.

    Reference: Bifet, A. & Gavalda, R. (2007). "Learning from Time-Changing
    Data with Adaptive Windowing."

    Args:
        delta: Confidence parameter — smaller values make the detector less
            sensitive (default 0.002).
        min_window: Minimum number of samples before any drift can be detected.
    """

    def __init__(self, delta: float = 0.002, min_window: int = 30) -> None:
        self._delta = delta
        self._min_window = min_window
        self._window: deque[float] = deque()
        self._n_total: int = 0
        self._drift_events: list[DriftEvent] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _epsilon_cut(n0: int, n1: int, delta: float) -> float:
        """ADWIN-style Hoeffding cut threshold.

        Args:
            n0: Size of the first sub-window.
            n1: Size of the second sub-window.
            delta: Confidence parameter.

        Returns:
            The epsilon cut value above which a drift is declared.
        """
        n = n0 + n1
        if n0 == 0 or n1 == 0:
            return float("inf")
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        return float(np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 * n / delta)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, reward: float) -> bool:
        """Feed a new reward observation and check for drift.

        Args:
            reward: Scalar reward from the latest live trade step.

        Returns:
            ``True`` if a drift event was detected; ``False`` otherwise.
        """
        with self._lock:
            self._window.append(reward)
            self._n_total += 1

            n = len(self._window)
            if n < self._min_window:
                return False

            window_list = list(self._window)
            global_mean = float(np.mean(window_list))

            # Scan all split points; declare drift at the first significant cut
            for cut in range(1, n):
                w0 = window_list[:cut]
                w1 = window_list[cut:]
                m0, m1 = float(np.mean(w0)), float(np.mean(w1))
                eps = self._epsilon_cut(len(w0), len(w1), self._delta)

                if abs(m0 - m1) >= eps:
                    event = DriftEvent(
                        detected_at=time.time(),
                        window_mean_before=m0,
                        window_mean_after=m1,
                        magnitude=abs(m0 - m1),
                        n_samples=self._n_total,
                    )
                    self._drift_events.append(event)
                    # Reset window to the most recent sub-window
                    self._window = deque(w1)
                    log.warning(
                        "drift.detected",
                        magnitude=round(event.magnitude, 4),
                        mean_before=round(m0, 4),
                        mean_after=round(m1, 4),
                        n_samples=self._n_total,
                    )
                    return True

            return False

    @property
    def drift_events(self) -> list[DriftEvent]:
        """All drift events detected so far (thread-safe copy)."""
        with self._lock:
            return list(self._drift_events)

    def reset(self) -> None:
        """Clear the sliding window (e.g. after a model update)."""
        with self._lock:
            self._window.clear()
        log.info("drift_detector.reset")


# ---------------------------------------------------------------------------
# Model Version Manager
# ---------------------------------------------------------------------------


@dataclass
class ModelCheckpoint:
    """Metadata about a saved model checkpoint.

    Attributes:
        version: Monotonically increasing version integer.
        path: Absolute path to the ``.pt`` checkpoint file.
        sharpe: Sharpe ratio measured on the validation window at save time.
        total_return: Cumulative return on the validation window.
        saved_at: Wall-clock time of the save operation.
    """

    version: int
    path: str
    sharpe: float
    total_return: float
    saved_at: float = field(default_factory=time.time)


class ModelVersionManager:
    """Saves and loads model checkpoints with performance tracking.

    Keeps track of the best-performing checkpoint and optionally prunes old
    ones to avoid filling disk.

    Args:
        checkpoint_dir: Directory where checkpoint files are written.
        max_versions: Maximum number of checkpoint files to keep on disk.
            Older checkpoints beyond this limit are deleted.
    """

    def __init__(
        self,
        checkpoint_dir: str = "./output/checkpoints",
        max_versions: int = 10,
    ) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_versions = max_versions
        self._versions: list[ModelCheckpoint] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        agent: Any,
        sharpe: float,
        total_return: float,
    ) -> ModelCheckpoint:
        """Save the agent's state dict to a versioned checkpoint file.

        Args:
            agent: A LARSA DQN agent with a ``act`` attribute that is a
                :class:`torch.nn.Module` (or any object with ``.state_dict()``
                and ``.load_state_dict()`` methods).
            sharpe: Sharpe ratio measured on the validation window.
            total_return: Cumulative return on the validation window.

        Returns:
            The :class:`ModelCheckpoint` metadata for the saved version.

        Raises:
            ModelError: If the save operation fails.
        """
        import torch

        with self._lock:
            version = len(self._versions) + 1

        path = self._dir / f"agent_v{version:04d}_sharpe{sharpe:.3f}.pt"

        try:
            state = {
                "act_state_dict": agent.act.state_dict(),
                "sharpe": sharpe,
                "total_return": total_return,
                "version": version,
            }
            torch.save(state, str(path))
        except Exception as exc:
            raise ModelError(
                f"Failed to save checkpoint to '{path}'.", cause=exc
            ) from exc

        ckpt = ModelCheckpoint(
            version=version,
            path=str(path),
            sharpe=sharpe,
            total_return=total_return,
        )

        with self._lock:
            self._versions.append(ckpt)
            self._prune()

        log.info(
            "version_manager.saved",
            version=version,
            path=str(path),
            sharpe=round(sharpe, 4),
        )
        return ckpt

    def load(self, agent: Any, version: int | None = None) -> ModelCheckpoint:
        """Load a checkpoint into *agent*.

        Args:
            agent: The agent whose ``act`` module will be updated in-place.
            version: Version to load.  Defaults to the best (highest Sharpe).

        Returns:
            The loaded :class:`ModelCheckpoint` metadata.

        Raises:
            ModelError: If no checkpoints exist or the file is unreadable.
        """
        import torch

        with self._lock:
            if not self._versions:
                raise ModelError("No checkpoints available to load.")

            if version is None:
                ckpt = max(self._versions, key=lambda c: c.sharpe)
            else:
                matches = [c for c in self._versions if c.version == version]
                if not matches:
                    raise ModelError(f"Checkpoint version {version} not found.")
                ckpt = matches[0]

        try:
            state = torch.load(ckpt.path, map_location="cpu")
            agent.act.load_state_dict(state["act_state_dict"])
        except Exception as exc:
            raise ModelError(
                f"Failed to load checkpoint from '{ckpt.path}'.", cause=exc
            ) from exc

        log.info(
            "version_manager.loaded",
            version=ckpt.version,
            sharpe=round(ckpt.sharpe, 4),
        )
        return ckpt

    @property
    def best(self) -> ModelCheckpoint | None:
        """The checkpoint with the highest Sharpe ratio, or ``None``."""
        with self._lock:
            if not self._versions:
                return None
            return max(self._versions, key=lambda c: c.sharpe)

    def _prune(self) -> None:
        """Delete oldest checkpoint files beyond *max_versions* (call under lock)."""
        while len(self._versions) > self._max_versions:
            oldest = self._versions.pop(0)
            try:
                Path(oldest.path).unlink(missing_ok=True)
                log.debug("version_manager.pruned", path=oldest.path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Online Finetuner
# ---------------------------------------------------------------------------


class OnlineFinetuner:
    """Periodically fine-tunes a DQN agent on recent live-trading experience.

    Uses small mini-batch gradient steps (no target network reset) so the
    agent adapts continuously without catastrophic forgetting.

    Args:
        agent: A LARSA DQN agent (must have ``act``, ``cri``, ``act_optimizer``,
            ``cri_optimizer`` attributes).
        replay: :class:`ExperienceReplay` buffer to sample from.
        batch_size: Mini-batch size per fine-tuning step.
        gamma: Discount factor for the Bellman backup.
        finetune_interval: Seconds between fine-tuning calls.
        min_buffer_size: Minimum transitions in the buffer before starting.
        steps_per_finetune: Number of gradient steps per scheduled call.
    """

    def __init__(
        self,
        agent: Any,
        replay: ExperienceReplay,
        batch_size: int = 64,
        gamma: float = 0.99,
        finetune_interval: float = 60.0,
        min_buffer_size: int = 256,
        steps_per_finetune: int = 4,
    ) -> None:
        self._agent = agent
        self._replay = replay
        self._batch_size = batch_size
        self._gamma = gamma
        self._interval = finetune_interval
        self._min_size = min_buffer_size
        self._steps = steps_per_finetune
        self._n_calls: int = 0
        self._last_loss: float = float("nan")

    def step(self) -> float:
        """Perform one scheduled fine-tuning call.

        Samples *steps_per_finetune* mini-batches from the replay buffer and
        applies a Bellman-target gradient update to the agent's Q-network.

        Returns:
            Mean TD loss across all gradient steps, or ``nan`` if the buffer
            is too small.
        """
        if len(self._replay) < self._min_size:
            log.debug(
                "finetuner.skip_small_buffer",
                size=len(self._replay),
                required=self._min_size,
            )
            return float("nan")

        import torch
        import torch.nn.functional as F

        losses = []
        agent = self._agent

        for _ in range(self._steps):
            batch = self._replay.sample(self._batch_size)
            states, actions, rewards, next_states, dones = self._replay.as_arrays(batch)

            s = torch.from_numpy(states)
            a = torch.from_numpy(actions)
            r = torch.from_numpy(rewards)
            s2 = torch.from_numpy(next_states)
            d = torch.from_numpy(dones)

            with torch.no_grad():
                # Use target Q-values from the agent's act network
                q_next = agent.act(s2)
                if q_next.dim() > 1:
                    q_next_max = q_next.max(dim=1).values
                else:
                    q_next_max = q_next
                q_target = r + self._gamma * q_next_max * (1.0 - d)

            q_pred_all = agent.act(s)
            if q_pred_all.dim() > 1:
                q_pred = q_pred_all.gather(1, a.unsqueeze(1)).squeeze(1)
            else:
                q_pred = q_pred_all

            loss = F.smooth_l1_loss(q_pred, q_target)
            agent.act_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.act.parameters(), 2.0)
            agent.act_optimizer.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        self._last_loss = mean_loss
        self._n_calls += 1
        log.info(
            "finetuner.step",
            call=self._n_calls,
            mean_loss=round(mean_loss, 6),
            buffer_size=len(self._replay),
        )
        return mean_loss

    @property
    def last_loss(self) -> float:
        """TD loss from the most recent fine-tuning call."""
        return self._last_loss

    @property
    def n_calls(self) -> int:
        """Total number of fine-tuning calls executed."""
        return self._n_calls


# ---------------------------------------------------------------------------
# ContinuousLearner — background orchestrator
# ---------------------------------------------------------------------------


class ContinuousLearner:
    """Background orchestrator that coordinates all continuous learning components.

    Runs a :class:`threading.Thread` that:
    1. Calls :class:`OnlineFinetuner` on a configurable schedule.
    2. Feeds each new reward through :class:`ConceptDriftDetector`.
    3. On drift: resets the exploration rate and increases fine-tuning frequency.
    4. Saves periodic checkpoints via :class:`ModelVersionManager`.

    The main trading loop only needs to call :meth:`record` to push each
    transition.  Everything else is handled asynchronously.

    Args:
        agent: A LARSA DQN agent.
        state_dim: Observation dimensionality.
        action_dim: Number of discrete actions.
        replay_capacity: Replay buffer size.
        batch_size: Fine-tuning mini-batch size.
        finetune_interval: Seconds between fine-tuning calls (normal rate).
        finetune_interval_drift: Seconds between calls after drift (faster).
        checkpoint_interval: Fine-tuning calls between checkpoint saves.
        checkpoint_dir: Directory for checkpoint files.
        drift_delta: ADWIN delta parameter.
        drift_explore_boost: Epsilon value to set on the agent after drift.
    """

    def __init__(
        self,
        agent: Any,
        state_dim: int = 32,
        action_dim: int = 3,
        replay_capacity: int = 100_000,
        batch_size: int = 64,
        finetune_interval: float = 60.0,
        finetune_interval_drift: float = 15.0,
        checkpoint_interval: int = 10,
        checkpoint_dir: str = "./output/checkpoints",
        drift_delta: float = 0.002,
        drift_explore_boost: float = 0.05,
    ) -> None:
        self._agent = agent
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._drift_explore_boost = drift_explore_boost
        self._checkpoint_interval = checkpoint_interval

        self.replay = ExperienceReplay(capacity=replay_capacity, state_dim=state_dim)
        self.drift_detector = ConceptDriftDetector(delta=drift_delta)
        self.version_manager = ModelVersionManager(checkpoint_dir=checkpoint_dir)
        self.finetuner = OnlineFinetuner(
            agent=agent,
            replay=self.replay,
            batch_size=batch_size,
            finetune_interval=finetune_interval,
        )

        self._normal_interval = finetune_interval
        self._drift_interval = finetune_interval_drift
        self._current_interval = finetune_interval

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._n_drift_events: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        state: _StateArray,
        action: int,
        reward: float,
        next_state: _StateArray,
        done: bool = False,
    ) -> None:
        """Push a live-trading transition into the replay buffer.

        Also feeds *reward* into the drift detector.  If drift is detected,
        the exploration rate is boosted and the fine-tuning interval is
        shortened.

        Args:
            state: Current observation.
            action: Action taken.
            reward: Reward received.
            next_state: Next observation.
            done: Whether the episode terminated.
        """
        self.replay.push(state, action, reward, next_state, done)
        drifted = self.drift_detector.update(reward)

        if drifted:
            self._n_drift_events += 1
            # Boost exploration
            if hasattr(self._agent, "explore_rate"):
                old_eps = self._agent.explore_rate
                self._agent.explore_rate = max(
                    self._drift_explore_boost, self._agent.explore_rate
                )
                log.warning(
                    "learner.drift_response",
                    old_explore=round(old_eps, 4),
                    new_explore=round(self._agent.explore_rate, 4),
                )
            # Increase fine-tuning frequency
            self._current_interval = self._drift_interval
            log.warning(
                "learner.interval_shortened",
                interval=self._drift_interval,
                drift_count=self._n_drift_events,
            )

    def start(self) -> None:
        """Start the background fine-tuning thread.

        Raises:
            RuntimeError: If the learner is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("ContinuousLearner is already running.")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._background_loop,
            name="ContinuousLearner",
            daemon=True,
        )
        self._thread.start()
        log.info("learner.started", interval=self._current_interval)

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the background thread to stop and wait for it to finish.

        Args:
            timeout: Seconds to wait for the thread to terminate.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                log.warning("learner.stop_timeout", timeout=timeout)
        log.info("learner.stopped")

    @property
    def is_running(self) -> bool:
        """Whether the background thread is currently alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def n_drift_events(self) -> int:
        """Total number of concept drift events detected so far."""
        return self._n_drift_events

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _background_loop(self) -> None:
        """Main body of the background fine-tuning thread."""
        log.info("learner.thread.running")
        calls_since_ckpt = 0

        while not self._stop_event.is_set():
            # Wait for the current interval, checking stop_event frequently
            deadline = time.monotonic() + self._current_interval
            while time.monotonic() < deadline:
                if self._stop_event.wait(timeout=1.0):
                    break

            if self._stop_event.is_set():
                break

            loss = self.finetuner.step()
            if not np.isnan(loss):
                calls_since_ckpt += 1

                # Gradually return to normal interval after drift response
                if (
                    self._current_interval == self._drift_interval
                    and calls_since_ckpt % 5 == 0
                ):
                    self._current_interval = self._normal_interval
                    log.info(
                        "learner.interval_restored",
                        interval=self._normal_interval,
                    )

                # Save checkpoint periodically
                if calls_since_ckpt % self._checkpoint_interval == 0:
                    self._save_checkpoint()

        log.info("learner.thread.exiting")

    def _save_checkpoint(self) -> None:
        """Save the current agent as a new versioned checkpoint."""
        try:
            # Compute a simple rolling Sharpe on recent rewards
            recent = list(self.drift_detector._window)
            if len(recent) >= 2:
                arr = np.array(recent, dtype=np.float32)
                sharpe = float(np.mean(arr) / (np.std(arr) + 1e-8))
                total_ret = float(np.sum(arr))
            else:
                sharpe = 0.0
                total_ret = 0.0

            self.version_manager.save(
                self._agent,
                sharpe=sharpe,
                total_return=total_ret,
            )
        except Exception as exc:
            log.error("learner.checkpoint_failed", error=str(exc))
