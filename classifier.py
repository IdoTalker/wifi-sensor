import numpy as np

# With 1 Hz sampling, Nyquist = 0.5 Hz. Walking (1-2 Hz) aliases into our
# detectable range, so the motion band still lights up for walking even though
# we can't resolve the exact stride frequency.
BREATHING_BAND = (0.05, 0.30)  # Hz — rhythmic presence / breathing
MOTION_BAND    = (0.30, 0.50)  # Hz — faster changes, aliased walking

MIN_SAMPLES = 20               # seconds of post-calibration data needed
EMPTY_FACTOR = 0.4             # mean score < threshold * this  →  empty


def classify(scores: list[float], threshold: float) -> tuple[str, float, float]:
    """
    Analyse the fused anomaly score history via FFT.

    Returns (state, breathing_fraction, motion_fraction) where:
      state: 'unknown' | 'empty' | 'present' | 'moving'
      fractions: share of total spectral power in each band (0–1)
    """
    if len(scores) < MIN_SAMPLES:
        return "unknown", 0.0, 0.0

    arr = np.array(scores, dtype=float)
    recent_mean = float(np.mean(arr[-10:]))

    # Remove DC component before FFT so power fractions reflect AC content only
    arr -= np.mean(arr)
    power = np.abs(np.fft.rfft(arr)) ** 2
    freqs = np.fft.rfftfreq(len(arr), d=1.0)  # d=1 second per sample

    total = float(np.sum(power[1:]))  # skip DC bin

    def band_fraction(low: float, high: float) -> float:
        if total == 0:
            return 0.0
        mask = (freqs >= low) & (freqs < high)
        return float(np.sum(power[mask])) / total

    b_frac = band_fraction(*BREATHING_BAND)
    m_frac = band_fraction(*MOTION_BAND)

    if recent_mean < threshold * EMPTY_FACTOR:
        state = "empty"
    elif m_frac >= b_frac:
        state = "moving"
    else:
        state = "present"

    return state, b_frac, m_frac
