"""FFT-based frequency-domain features for telemetry signals.

Extracts dominant frequency, spectral entropy, and power-ratio
features that help detect rapid fluctuations indicative of congestion.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _next_power_of_2(n: int) -> int:
    """Return the next power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def compute_fft_features(
    series: pd.Series, sample_rate: float = 1.0
) -> Dict[str, Any]:
    """Compute frequency-domain features from a time series via FFT.

    The series is zero-padded to the next power of 2 for efficiency.

    Args:
        series: Time-series values (e.g., rtt_ms over a window).
        sample_rate: Sampling rate in Hz (default 1.0 = 1 sample/sec).

    Returns:
        Dictionary of scalar features:
        - dominant_frequency_hz: frequency with highest magnitude
        - dominant_period_sec: 1 / dominant_frequency_hz
        - spectral_entropy: entropy of the normalised power spectrum
        - top3_frequencies: list of top-3 frequency components (Hz)
        - spectral_power_ratio: ratio of power above 0.1 Hz to total
    """
    values = series.dropna().values.astype(np.float64)

    if len(values) < 4:
        return {
            "dominant_frequency_hz": 0.0,
            "dominant_period_sec": float("inf"),
            "spectral_entropy": 0.0,
            "top3_frequencies": [0.0, 0.0, 0.0],
            "spectral_power_ratio": 0.0,
        }

    # Zero-mean the signal
    values = values - np.mean(values)

    # Zero-pad to next power of 2
    padded_len = _next_power_of_2(len(values))
    padded = np.zeros(padded_len)
    padded[: len(values)] = values

    # FFT
    fft_vals = np.fft.rfft(padded)
    magnitudes = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(padded_len, d=1.0 / sample_rate)

    # Ignore DC component (index 0)
    magnitudes_no_dc = magnitudes[1:]
    freqs_no_dc = freqs[1:]

    if len(magnitudes_no_dc) == 0 or magnitudes_no_dc.sum() == 0:
        return {
            "dominant_frequency_hz": 0.0,
            "dominant_period_sec": float("inf"),
            "spectral_entropy": 0.0,
            "top3_frequencies": [0.0, 0.0, 0.0],
            "spectral_power_ratio": 0.0,
        }

    # Dominant frequency
    dominant_idx = np.argmax(magnitudes_no_dc)
    dominant_freq = float(freqs_no_dc[dominant_idx])
    dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else float("inf")

    # Top-3 frequencies
    top3_indices = np.argsort(magnitudes_no_dc)[-3:][::-1]
    top3_freqs: List[float] = [float(freqs_no_dc[i]) for i in top3_indices]
    while len(top3_freqs) < 3:
        top3_freqs.append(0.0)

    # Spectral entropy
    power = magnitudes_no_dc ** 2
    total_power = power.sum()
    if total_power > 0:
        norm_power = power / total_power
        # Avoid log(0)
        norm_power = norm_power[norm_power > 0]
        spectral_entropy = float(-np.sum(norm_power * np.log2(norm_power)))
    else:
        spectral_entropy = 0.0

    # Spectral power ratio (power above 0.1 Hz / total)
    high_freq_mask = freqs_no_dc > 0.1
    high_freq_power = float(power[high_freq_mask].sum()) if high_freq_mask.any() else 0.0
    spectral_power_ratio = high_freq_power / total_power if total_power > 0 else 0.0

    return {
        "dominant_frequency_hz": round(dominant_freq, 6),
        "dominant_period_sec": round(dominant_period, 3),
        "spectral_entropy": round(spectral_entropy, 4),
        "top3_frequencies": [round(f, 6) for f in top3_freqs],
        "spectral_power_ratio": round(float(spectral_power_ratio), 4),
    }
