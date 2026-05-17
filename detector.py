from collections import deque
import numpy as np


CALIBRATION_SAMPLES = 30
WINDOW_SIZE = 60
VAR_WINDOW = 8        # short window for rolling variance computation
HYSTERESIS = 3
ADAPT_RATE = 0.015    # EMA rate for slow baseline drift during clear periods


class MotionDetector:
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

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def calibration_progress(self) -> int:
        return min(len(self._baseline_buf), CALIBRATION_SAMPLES)

    def reset(self):
        self.__init__(threshold=self.threshold)

    def update(self, rssi: float) -> bool:
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
        return self._motion

    def score(self) -> float:
        return self._score

    def history(self) -> list[float]:
        return list(self._window)
