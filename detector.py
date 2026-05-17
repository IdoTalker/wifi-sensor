"""Per-network motion detector using mean-shift and rolling-variance Z-scores.

After a quiet calibration period it scores each incoming RSSI sample and
flips a hysteresis-debounced motion flag when the score exceeds a threshold.
An adaptive baseline slowly tracks environmental drift during clear periods.
"""

from collections import deque
import numpy as np


CALIBRATION_SAMPLES = 30
WINDOW_SIZE = 60
VAR_WINDOW = 8        # short window for rolling variance computation
HYSTERESIS = 3
ADAPT_RATE = 0.015    # EMA rate for slow baseline drift during clear periods
EMA_ALPHA  = 0.4      # smoothing weight on each new RSSI sample (reduces netsh quantization noise)


class MotionDetector:
    """Single-network anomaly detector with two-stage scoring and hysteresis.

    Phase 1 — calibration: collects CALIBRATION_SAMPLES quiet readings to
    establish a baseline mean, std, and typical rolling variance level.

    Phase 2 — detection: scores each new RSSI value as
        score = max(z_mean, z_var)
    where z_mean is the mean-shift normalised by baseline std, and z_var is
    the ratio of current rolling std to the baseline jitter level (minus 1 so
    normal jitter contributes ~0).  The motion flag flips via HYSTERESIS
    consecutive anomalies to avoid single-sample false positives.
    """

    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold
        self._baseline_buf: list[float] = []
        self._window: deque[float] = deque(maxlen=WINDOW_SIZE)
        self._recent: deque[float] = deque(maxlen=VAR_WINDOW)
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 1.0
        self._baseline_var_level: float = 1.0  # typical rolling-std during calibration
        self._calibrated: bool = False
        self._consecutive_anomalies: int = 0
        self._motion: bool = False
        self._score: float = 0.0
        self._ema: float | None = None

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def calibration_progress(self) -> int:
        return min(len(self._baseline_buf), CALIBRATION_SAMPLES)

    def reset(self):
        """Restart calibration, preserving the current threshold."""
        self.__init__(threshold=self.threshold)

    def update(self, rssi: float) -> bool:
        """Feed one RSSI sample; return the current motion flag.

        During calibration returns False.  After calibration updates the score,
        manages the hysteresis counter, and slowly adapts the baseline during
        clear periods.
        """
        if self._ema is None:
            self._ema = rssi
        else:
            self._ema = EMA_ALPHA * rssi + (1.0 - EMA_ALPHA) * self._ema
        rssi = self._ema

        self._window.append(rssi)
        self._recent.append(rssi)

        if not self._calibrated:
            self._baseline_buf.append(rssi)
            if len(self._baseline_buf) >= CALIBRATION_SAMPLES:
                arr = np.array(self._baseline_buf)
                self._baseline_mean = float(np.mean(arr))
                self._baseline_std = max(float(np.std(arr)), 0.5)
                # Typical jitter level: mean of all rolling std devs during calibration
                rolling = [
                    float(np.std(self._baseline_buf[i: i + VAR_WINDOW]))
                    for i in range(len(self._baseline_buf) - VAR_WINDOW + 1)
                ]
                self._baseline_var_level = max(float(np.mean(rolling)), 0.2)
                self._calibrated = True
            return False

        # Mean-shift: how far is this sample from the quiet baseline?
        z_mean = abs(rssi - self._baseline_mean) / self._baseline_std

        # Variance anomaly: is the signal jittering more than usual?
        z_var = 0.0
        if len(self._recent) >= VAR_WINDOW:
            recent_std = float(np.std(list(self._recent)))
            # Subtract 1 so a normal jitter level contributes ~0 to the score
            z_var = max(0.0, (recent_std / self._baseline_var_level) - 1.0)

        self._score = max(z_mean, z_var)

        anomalous = self._score > self.threshold
        if anomalous:
            self._consecutive_anomalies += 1
        else:
            self._consecutive_anomalies = max(0, self._consecutive_anomalies - 1)
            # Adaptive baseline: slowly track environmental drift when clear
            self._baseline_mean += ADAPT_RATE * (rssi - self._baseline_mean)

        if self._consecutive_anomalies >= HYSTERESIS:
            self._motion = True
        elif self._consecutive_anomalies == 0:
            self._motion = False

        return self._motion

    def is_motion(self) -> bool:
        """Return the current debounced motion flag."""
        return self._motion

    def score(self) -> float:
        """Return the most recent anomaly score (0 during calibration)."""
        return self._score

    def history(self) -> list[float]:
        """Return up to WINDOW_SIZE recent RSSI readings, oldest first."""
        return list(self._window)
