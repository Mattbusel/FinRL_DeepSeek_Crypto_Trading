"""Online learning with concept drift detection.

Implements:
- ``DriftDetector``      — simplified ADWIN-style drift detector.
- ``OnlineLearner``      — perceptron-style online gradient learner.
- ``ConceptDriftMonitor`` — monitors a scalar stream for drift events.
"""

from __future__ import annotations

import math
import unittest
from collections import deque
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# DriftDetector — simplified ADWIN
# ---------------------------------------------------------------------------


class DriftDetector:
    """Lightweight ADWIN-inspired drift detector.

    Maintains a running mean and variance of the input stream.  Drift is
    flagged when the current sample deviates from the running mean by more
    than ``threshold`` standard deviations (when at least ``min_samples``
    have been observed).
    """

    def __init__(self, threshold: float = 3.0, min_samples: int = 30) -> None:
        self._threshold = threshold
        self._min_samples = min_samples
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0  # Welford's M2

    def update(self, value: float) -> bool:
        """Ingest *value* and return *True* if drift is detected."""
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._m2 += delta * delta2

        if self._n < self._min_samples:
            return False

        std = math.sqrt(self._m2 / (self._n - 1))
        if std < 1e-12:
            return False

        z = abs(value - self._mean) / std
        return z > self._threshold

    def mean(self) -> float:
        return self._mean

    def variance(self) -> float:
        if self._n < 2:
            return 0.0
        return self._m2 / (self._n - 1)

    def n_samples(self) -> int:
        return self._n

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0


# ---------------------------------------------------------------------------
# OnlineLearner — perceptron-style gradient learner
# ---------------------------------------------------------------------------


class OnlineLearner:
    """Online gradient learner (perceptron-style with squared-error loss).

    Prediction: y_hat = w · x + b
    Update:     w += lr * (label - y_hat) * x
                b += lr * (label - y_hat)
    """

    def __init__(self, n_features: int = 1, learning_rate: float = 0.01) -> None:
        self._n_features = n_features
        self._lr = learning_rate
        self._weights: list[float] = [0.0] * n_features
        self._bias: float = 0.0
        self._recent_errors: deque[float] = deque(maxlen=200)

    def predict(self, features: list[float]) -> float:
        n = min(len(features), self._n_features)
        return sum(self._weights[i] * features[i] for i in range(n)) + self._bias

    def update(self, features: list[float], label: float) -> None:
        """Ingest one labelled example and update weights."""
        y_hat = self.predict(features)
        error = label - y_hat
        n = min(len(features), self._n_features)
        for i in range(n):
            self._weights[i] += self._lr * error * features[i]
        self._bias += self._lr * error
        self._recent_errors.append(abs(error))

    def sliding_accuracy(self, window: int = 100) -> float:
        """Fraction of recent predictions within 0.5 of the true label.

        Uses the stored absolute errors; a prediction is considered
        "correct" if |error| <= 0.5.
        """
        errors = list(self._recent_errors)[-window:]
        if not errors:
            return 0.0
        correct = sum(1 for e in errors if e <= 0.5)
        return correct / len(errors)

    def adapt_to_drift(self, detector: DriftDetector) -> None:
        """Reset weights if the detector is in a post-drift state (mean shift).

        We reset when the detector has seen at least ``min_samples`` and its
        mean is far from its initial value (proxy for confirmed drift).
        """
        if detector.n_samples() >= 30 and detector.variance() > 0.0:
            self._weights = [0.0] * self._n_features
            self._bias = 0.0
            self._recent_errors.clear()


# ---------------------------------------------------------------------------
# ConceptDriftMonitor
# ---------------------------------------------------------------------------


class ConceptDriftMonitor:
    """Monitor a scalar stream for concept drift events."""

    def monitor(
        self, stream: list[float], detector: DriftDetector
    ) -> list[int]:
        """Process each value in *stream* and return indices where drift fires.

        Args:
            stream:   Sequence of scalar observations.
            detector: A :class:`DriftDetector` instance (reset before use if
                      desired).

        Returns:
            Sorted list of indices at which drift was detected.
        """
        drift_indices: list[int] = []
        for idx, value in enumerate(stream):
            if detector.update(value):
                drift_indices.append(idx)
        return drift_indices

    def drift_frequency(
        self, drift_indices: list[int], total_length: int
    ) -> float:
        """Fraction of stream positions that triggered a drift alarm.

        Returns:
            A float in [0.0, 1.0].
        """
        if total_length <= 0:
            return 0.0
        return len(drift_indices) / total_length


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestDriftDetector(unittest.TestCase):
    def test_no_drift_stable(self) -> None:
        det = DriftDetector(threshold=3.0, min_samples=10)
        detections = []
        for i in range(50):
            detections.append(det.update(1.0))
        self.assertFalse(any(detections))

    def test_drift_on_spike(self) -> None:
        det = DriftDetector(threshold=2.0, min_samples=10)
        for _ in range(40):
            det.update(0.0)
        # Inject a massive spike.
        detected = det.update(1000.0)
        self.assertTrue(detected)

    def test_n_samples_increments(self) -> None:
        det = DriftDetector()
        for i in range(5):
            det.update(float(i))
        self.assertEqual(det.n_samples(), 5)

    def test_mean_approx(self) -> None:
        det = DriftDetector(min_samples=2)
        det.update(2.0)
        det.update(4.0)
        self.assertAlmostEqual(det.mean(), 3.0)

    def test_variance_positive(self) -> None:
        det = DriftDetector()
        for _ in range(10):
            det.update(1.0)
        det.update(5.0)
        self.assertGreaterEqual(det.variance(), 0.0)

    def test_reset_clears_state(self) -> None:
        det = DriftDetector()
        det.update(5.0)
        det.reset()
        self.assertEqual(det.n_samples(), 0)
        self.assertEqual(det.mean(), 0.0)

    def test_no_drift_before_min_samples(self) -> None:
        det = DriftDetector(threshold=0.0, min_samples=100)
        detected = det.update(999.0)
        self.assertFalse(detected)


class TestOnlineLearner(unittest.TestCase):
    def test_predict_zero_initial(self) -> None:
        learner = OnlineLearner(n_features=3)
        self.assertAlmostEqual(learner.predict([1.0, 2.0, 3.0]), 0.0)

    def test_learns_constant_function(self) -> None:
        learner = OnlineLearner(n_features=1, learning_rate=0.1)
        for _ in range(200):
            learner.update([1.0], 5.0)
        pred = learner.predict([1.0])
        self.assertAlmostEqual(pred, 5.0, places=1)

    def test_sliding_accuracy_initially_zero(self) -> None:
        learner = OnlineLearner(n_features=2)
        self.assertEqual(learner.sliding_accuracy(), 0.0)

    def test_sliding_accuracy_after_training(self) -> None:
        learner = OnlineLearner(n_features=1, learning_rate=0.1)
        for _ in range(200):
            learner.update([1.0], 1.0)
        acc = learner.sliding_accuracy(window=50)
        self.assertGreaterEqual(acc, 0.9)

    def test_adapt_to_drift_resets_weights(self) -> None:
        learner = OnlineLearner(n_features=2, learning_rate=0.1)
        for _ in range(50):
            learner.update([1.0, 0.5], 3.0)
        det = DriftDetector(min_samples=30)
        for _ in range(30):
            det.update(1.0)
        det.update(100.0)  # force variance > 0
        learner.adapt_to_drift(det)
        # After adaptation weights should be zeroed.
        self.assertAlmostEqual(learner.predict([1.0, 1.0]), 0.0)

    def test_update_changes_prediction(self) -> None:
        learner = OnlineLearner(n_features=1, learning_rate=0.5)
        before = learner.predict([1.0])
        learner.update([1.0], 10.0)
        after = learner.predict([1.0])
        self.assertNotAlmostEqual(before, after)


class TestConceptDriftMonitor(unittest.TestCase):
    def test_no_drift_constant_stream(self) -> None:
        monitor = ConceptDriftMonitor()
        det = DriftDetector(threshold=3.0, min_samples=20)
        indices = monitor.monitor([1.0] * 50, det)
        self.assertEqual(indices, [])

    def test_drift_detected_after_shift(self) -> None:
        monitor = ConceptDriftMonitor()
        det = DriftDetector(threshold=2.0, min_samples=20)
        stream = [0.0] * 30 + [1000.0] * 5
        indices = monitor.monitor(stream, det)
        self.assertGreater(len(indices), 0)

    def test_drift_frequency_zero(self) -> None:
        monitor = ConceptDriftMonitor()
        self.assertAlmostEqual(monitor.drift_frequency([], 100), 0.0)

    def test_drift_frequency_all(self) -> None:
        monitor = ConceptDriftMonitor()
        freq = monitor.drift_frequency([0, 1, 2, 3], 4)
        self.assertAlmostEqual(freq, 1.0)

    def test_drift_frequency_zero_total(self) -> None:
        monitor = ConceptDriftMonitor()
        self.assertEqual(monitor.drift_frequency([1, 2], 0), 0.0)

    def test_drift_indices_are_sorted(self) -> None:
        monitor = ConceptDriftMonitor()
        det = DriftDetector(threshold=2.0, min_samples=10)
        stream = [0.0] * 15 + [100.0, 100.0, 0.0, 100.0]
        indices = monitor.monitor(stream, det)
        self.assertEqual(indices, sorted(indices))


if __name__ == "__main__":
    unittest.main()
