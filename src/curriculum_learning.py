"""Curriculum learning for RL: progressively harder training tasks."""

from __future__ import annotations

import math
import random
import unittest
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# TaskDifficulty
# ---------------------------------------------------------------------------


@dataclass
class TaskDifficulty:
    """Parameters defining the difficulty of a single RL training task."""

    level: int               # Difficulty level (0 = easiest)
    volatility: float        # Price volatility of synthetic episode data
    trend_strength: float    # Magnitude of the trend component (0 = no trend)
    noise_level: float       # IID noise amplitude added to prices
    episode_length: int      # Number of steps per episode


# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------


class CurriculumScheduler:
    """Manage a sequence of progressively harder RL tasks.

    Automatically advances the curriculum when the agent achieves a
    sufficient performance level at the current difficulty.
    """

    def __init__(self, n_levels: int = 5) -> None:
        self._curriculum: list[TaskDifficulty] = self.create_curriculum(n_levels)
        self._current_idx: int = 0
        self._performance_history: dict[int, list[float]] = {
            i: [] for i in range(n_levels)
        }

    # -----------------------------------------------------------------------
    # Curriculum construction
    # -----------------------------------------------------------------------

    @staticmethod
    def create_curriculum(n_levels: int = 5) -> list[TaskDifficulty]:
        """Generate a list of TaskDifficulty instances from easy to hard.

        Difficulty increases linearly across all parameters.

        Args:
            n_levels: Number of curriculum levels.

        Returns:
            Ordered list of TaskDifficulty objects.
        """
        if n_levels <= 0:
            raise ValueError("n_levels must be positive")

        curriculum = []
        for i in range(n_levels):
            frac = i / max(n_levels - 1, 1)  # 0 at level 0, 1 at max level
            curriculum.append(
                TaskDifficulty(
                    level=i,
                    volatility=0.01 + 0.04 * frac,        # 0.01 → 0.05
                    trend_strength=0.0 + 0.5 * frac,       # 0.0  → 0.5
                    noise_level=0.001 + 0.009 * frac,      # 0.001 → 0.01
                    episode_length=int(50 + 200 * frac),   # 50   → 250
                )
            )
        return curriculum

    # -----------------------------------------------------------------------
    # Current state
    # -----------------------------------------------------------------------

    def current_task(self) -> TaskDifficulty:
        """Return the currently active TaskDifficulty."""
        return self._curriculum[self._current_idx]

    # -----------------------------------------------------------------------
    # Advancement logic
    # -----------------------------------------------------------------------

    def should_advance(
        self,
        performance_history: list[float],
        threshold: float = 0.7,
        min_episodes: int = 10,
    ) -> bool:
        """Determine whether the agent should advance to the next level.

        Args:
            performance_history: Recent episode performance scores in [0, 1].
            threshold: Mean performance required to advance.
            min_episodes: Minimum number of episodes before advancing.

        Returns:
            True if the curriculum should advance.
        """
        if self._current_idx >= len(self._curriculum) - 1:
            return False  # Already at the hardest level
        if len(performance_history) < min_episodes:
            return False
        recent = performance_history[-min_episodes:]
        return sum(recent) / len(recent) >= threshold

    def advance(self) -> None:
        """Move to the next difficulty level (if not already at max)."""
        if self._current_idx < len(self._curriculum) - 1:
            self._current_idx += 1

    # -----------------------------------------------------------------------
    # Episode data generation
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_episode_data(
        task: TaskDifficulty,
        length: int,
        seed: int = 42,
    ) -> list[float]:
        """Generate a synthetic price series for the given task difficulty.

        Uses a simple random walk with trend and noise:
        ```
        price[t] = price[t-1] * exp(trend + volatility * Z + noise * eps)
        ```

        Args:
            task: Task difficulty parameters.
            length: Number of price observations to generate.
            seed: Random seed for reproducibility.

        Returns:
            List of synthetic prices.
        """
        rng = random.Random(seed)
        prices = [100.0]
        for _ in range(length - 1):
            z = rng.gauss(0, 1)
            eps = rng.gauss(0, 1)
            log_return = (
                task.trend_strength / length  # small per-step drift
                + task.volatility * z / math.sqrt(252)
                + task.noise_level * eps
            )
            prices.append(prices[-1] * math.exp(log_return))
        return prices

    # -----------------------------------------------------------------------
    # Performance tracking
    # -----------------------------------------------------------------------

    def record_performance(self, level: int, score: float) -> None:
        """Record an episode performance score for the given level.

        Args:
            level: Curriculum level the score was achieved at.
            score: Performance metric (e.g. normalised return or win rate).
        """
        if level in self._performance_history:
            self._performance_history[level].append(score)

    def performance_at_level(self, level: int) -> list[float]:
        """Return the historical performance scores at the given level.

        Args:
            level: Curriculum level index.

        Returns:
            List of performance scores, or empty list if none recorded.
        """
        return self._performance_history.get(level, [])

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def current_level(self) -> int:
        """Index of the current curriculum level."""
        return self._current_idx

    @property
    def n_levels(self) -> int:
        """Total number of curriculum levels."""
        return len(self._curriculum)

    @property
    def at_max_difficulty(self) -> bool:
        """True if the agent is at the hardest level."""
        return self._current_idx == len(self._curriculum) - 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCurriculumScheduler(unittest.TestCase):
    def test_create_curriculum_default(self):
        curriculum = CurriculumScheduler.create_curriculum()
        self.assertEqual(len(curriculum), 5)

    def test_create_curriculum_custom_levels(self):
        curriculum = CurriculumScheduler.create_curriculum(n_levels=3)
        self.assertEqual(len(curriculum), 3)

    def test_curriculum_difficulty_increases(self):
        curriculum = CurriculumScheduler.create_curriculum(n_levels=5)
        for i in range(1, len(curriculum)):
            self.assertGreater(curriculum[i].volatility, curriculum[i - 1].volatility)
            self.assertGreaterEqual(curriculum[i].episode_length, curriculum[i - 1].episode_length)

    def test_create_curriculum_invalid(self):
        with self.assertRaises(ValueError):
            CurriculumScheduler.create_curriculum(n_levels=0)

    def test_current_task_starts_at_level_0(self):
        sched = CurriculumScheduler()
        self.assertEqual(sched.current_task().level, 0)

    def test_should_advance_false_insufficient_episodes(self):
        sched = CurriculumScheduler()
        self.assertFalse(sched.should_advance([1.0, 1.0], threshold=0.7, min_episodes=10))

    def test_should_advance_true_above_threshold(self):
        sched = CurriculumScheduler()
        high_scores = [0.9] * 15
        self.assertTrue(sched.should_advance(high_scores, threshold=0.7, min_episodes=10))

    def test_should_advance_false_below_threshold(self):
        sched = CurriculumScheduler()
        low_scores = [0.4] * 15
        self.assertFalse(sched.should_advance(low_scores, threshold=0.7, min_episodes=10))

    def test_advance_increments_level(self):
        sched = CurriculumScheduler()
        sched.advance()
        self.assertEqual(sched.current_level, 1)

    def test_advance_does_not_exceed_max(self):
        sched = CurriculumScheduler(n_levels=2)
        sched.advance()
        sched.advance()  # already at max
        self.assertEqual(sched.current_level, 1)

    def test_at_max_difficulty(self):
        sched = CurriculumScheduler(n_levels=2)
        self.assertFalse(sched.at_max_difficulty)
        sched.advance()
        self.assertTrue(sched.at_max_difficulty)

    def test_should_advance_false_at_max(self):
        sched = CurriculumScheduler(n_levels=1)
        self.assertFalse(sched.should_advance([1.0] * 20))

    def test_generate_episode_data_length(self):
        task = TaskDifficulty(0, 0.01, 0.0, 0.001, 100)
        data = CurriculumScheduler.generate_episode_data(task, 100)
        self.assertEqual(len(data), 100)

    def test_generate_episode_data_positive_prices(self):
        task = TaskDifficulty(2, 0.03, 0.2, 0.005, 200)
        data = CurriculumScheduler.generate_episode_data(task, 200)
        self.assertTrue(all(p > 0 for p in data))

    def test_generate_episode_data_reproducible(self):
        task = TaskDifficulty(1, 0.02, 0.1, 0.002, 50)
        d1 = CurriculumScheduler.generate_episode_data(task, 50, seed=7)
        d2 = CurriculumScheduler.generate_episode_data(task, 50, seed=7)
        self.assertEqual(d1, d2)

    def test_performance_at_level_empty(self):
        sched = CurriculumScheduler()
        self.assertEqual(sched.performance_at_level(0), [])

    def test_record_and_retrieve_performance(self):
        sched = CurriculumScheduler()
        sched.record_performance(0, 0.8)
        sched.record_performance(0, 0.75)
        perf = sched.performance_at_level(0)
        self.assertEqual(len(perf), 2)
        self.assertAlmostEqual(perf[0], 0.8)


if __name__ == "__main__":
    unittest.main()
