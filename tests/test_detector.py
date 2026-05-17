"""Tests for detector.py — MotionDetector calibration, scoring, hysteresis."""
import unittest
from wifi_sensor.detector import MotionDetector, CALIBRATION_SAMPLES, HYSTERESIS


def _feed(det, value, count):
    results = []
    for _ in range(count):
        results.append(det.update(value))
    return results


class TestCalibration(unittest.TestCase):
    def test_not_calibrated_during_calibration(self):
        det = MotionDetector(name="test")
        for _ in range(CALIBRATION_SAMPLES - 1):
            self.assertFalse(det.update(-60.0))
        self.assertFalse(det.calibrated)

    def test_calibrates_after_required_samples(self):
        det = MotionDetector(name="test")
        _feed(det, -60.0, CALIBRATION_SAMPLES)
        self.assertTrue(det.calibrated)

    def test_returns_false_during_calibration(self):
        det = MotionDetector(name="test")
        results = _feed(det, -60.0, CALIBRATION_SAMPLES)
        self.assertTrue(all(r is False for r in results))

    def test_calibration_progress_increments(self):
        det = MotionDetector(name="test")
        det.update(-60.0)
        self.assertEqual(det.calibration_progress, 1)
        _feed(det, -60.0, 10)
        self.assertEqual(det.calibration_progress, 11)


class TestDetection(unittest.TestCase):
    def _calibrated_det(self, baseline=-60.0, threshold=2.0):
        det = MotionDetector(threshold=threshold, name="test")
        _feed(det, baseline, CALIBRATION_SAMPLES)
        return det

    def test_no_motion_at_baseline(self):
        det = self._calibrated_det()
        for _ in range(10):
            self.assertFalse(det.update(-60.0))

    def test_motion_after_hysteresis_consecutive_anomalies(self):
        det = self._calibrated_det(baseline=-60.0, threshold=1.0)
        # Feed a strongly anomalous signal
        results = _feed(det, -10.0, HYSTERESIS + 2)
        self.assertTrue(results[-1], "Expected motion after hysteresis")

    def test_no_motion_before_full_hysteresis(self):
        det = self._calibrated_det(baseline=-60.0, threshold=1.0)
        results = _feed(det, -10.0, HYSTERESIS - 1)
        self.assertFalse(results[-1], "Should not trigger before hysteresis threshold")

    def test_motion_clears_on_return_to_baseline(self):
        det = self._calibrated_det(baseline=-60.0, threshold=1.0)
        _feed(det, -10.0, HYSTERESIS + 5)
        self.assertTrue(det.is_motion())
        # EMA (α=0.4) takes ~9 steps to converge from a 50dB excursion, then the
        # accumulated hysteresis counter needs to drain — 60 samples is safely enough.
        _feed(det, -60.0, 60)
        self.assertFalse(det.is_motion())

    def test_score_zero_before_calibration(self):
        det = MotionDetector(name="test")
        det.update(-60.0)
        self.assertEqual(det.score(), 0.0)

    def test_score_positive_after_anomaly(self):
        det = self._calibrated_det(baseline=-60.0, threshold=1.0)
        det.update(-10.0)
        self.assertGreater(det.score(), 0.0)

    def test_history_length(self):
        det = self._calibrated_det()
        _feed(det, -60.0, 5)
        self.assertGreater(len(det.history()), 0)


class TestReset(unittest.TestCase):
    def test_reset_clears_calibration(self):
        det = MotionDetector(name="test")
        _feed(det, -60.0, CALIBRATION_SAMPLES)
        self.assertTrue(det.calibrated)
        det.reset()
        self.assertFalse(det.calibrated)

    def test_reset_preserves_threshold(self):
        det = MotionDetector(threshold=3.5, name="test")
        det.reset()
        self.assertEqual(det.threshold, 3.5)


if __name__ == "__main__":
    unittest.main()
