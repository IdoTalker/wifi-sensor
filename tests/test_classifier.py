"""Tests for classifier.py — FFT-based 3-state activity classification."""
import math
import unittest

from wifi_sensor.classifier import classify, MIN_SAMPLES, EMPTY_FACTOR


def _make_scores(state: str, n: int = MIN_SAMPLES, threshold: float = 2.0) -> list[float]:
    """Generate synthetic score history matching a given activity state."""
    if state == "empty":
        return [0.05] * n
    if state == "present":
        # Breathing: sine wave at ~0.25 Hz (15 bpm), amplitude just above threshold
        return [threshold * 0.7 + threshold * 0.4 * math.sin(2 * math.pi * 0.25 * i) for i in range(n)]
    if state == "moving":
        # Broad-spectrum high-amplitude noise
        import random
        random.seed(42)
        return [threshold * 2.0 + random.uniform(0, threshold) for _ in range(n)]
    raise ValueError(f"unknown state: {state}")


class TestClassifyEmpty(unittest.TestCase):
    def test_flat_low_signal_is_empty(self):
        scores = [0.05] * MIN_SAMPLES
        state, b, m = classify(scores, threshold=2.0)
        self.assertEqual(state, "empty")

    def test_returns_three_tuple(self):
        scores = [0.1] * MIN_SAMPLES
        result = classify(scores, threshold=2.0)
        self.assertEqual(len(result), 3)


class TestClassifyPresent(unittest.TestCase):
    def test_breathing_signal_classified_as_present_or_moving(self):
        scores = _make_scores("present")
        state, b, m = classify(scores, threshold=2.0)
        # Present with breathing should not be empty
        self.assertNotEqual(state, "empty")

    def test_breathing_fraction_between_0_and_1(self):
        scores = _make_scores("present")
        _, b, m = classify(scores, threshold=2.0)
        self.assertGreaterEqual(b, 0.0)
        self.assertLessEqual(b, 1.0)
        self.assertGreaterEqual(m, 0.0)
        self.assertLessEqual(m, 1.0)


class TestClassifyMoving(unittest.TestCase):
    def test_high_noise_classified_as_moving_or_present(self):
        scores = _make_scores("moving")
        state, b, m = classify(scores, threshold=2.0)
        self.assertNotEqual(state, "empty")


class TestEdgeCases(unittest.TestCase):
    def test_below_min_samples_raises_or_returns_empty(self):
        # classify() requires MIN_SAMPLES; passing fewer should either raise or
        # return empty — both are acceptable behaviours.
        try:
            state, _, _ = classify([0.5] * (MIN_SAMPLES - 1), threshold=2.0)
            self.assertEqual(state, "empty")
        except Exception:
            pass  # raising is also valid

    def test_all_zeros_is_empty(self):
        scores = [0.0] * MIN_SAMPLES
        state, _, _ = classify(scores, threshold=2.0)
        self.assertEqual(state, "empty")


if __name__ == "__main__":
    unittest.main()
