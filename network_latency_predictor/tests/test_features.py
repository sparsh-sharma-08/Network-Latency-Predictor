"""Tests for the feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.rolling_stats import compute_rolling_stats
from features.ewma import compute_ewma, compute_ewma_std, compute_ewma_gradient
from features.fft_features import compute_fft_features
from features.gradient_estimator import finite_difference_gradient, congestion_onset_score


class TestRollingStats:
    """Tests for rolling window statistics."""

    def test_rolling_stats_correctness(self) -> None:
        """Known array → verify exact p95 value."""
        series = pd.Series(list(range(1, 101)))  # 1 to 100
        stats = compute_rolling_stats(series, window=100)

        # p95 of 1..100 should be ~95.25
        p95_val = stats["p95"].iloc[-1]
        assert abs(p95_val - 95.05) < 1.0, f"p95 expected ~95, got {p95_val}"

    def test_rolling_stats_keys(self) -> None:
        """All expected stat keys must be present."""
        series = pd.Series(np.random.randn(200))
        stats = compute_rolling_stats(series, window=10)
        expected_keys = {"mean", "std", "min", "max", "p50", "p95", "p99",
                         "iqr", "skewness", "kurtosis"}
        assert expected_keys == set(stats.keys())

    def test_rolling_stats_length(self) -> None:
        """Output series must be the same length as input."""
        n = 150
        series = pd.Series(np.random.randn(n))
        stats = compute_rolling_stats(series, window=20)
        for key, s in stats.items():
            assert len(s) == n, f"{key} length mismatch"


class TestEWMA:
    """Tests for exponential weighted moving average features."""

    def test_ewma_convergence(self) -> None:
        """Constant series → EWMA must equal the constant."""
        series = pd.Series([5.0] * 100)
        ewma = compute_ewma(series, alpha=0.3)
        assert np.allclose(ewma.values, 5.0, atol=1e-6)

    def test_ewma_std_constant(self) -> None:
        """Constant series → EWMA std should be ~0."""
        series = pd.Series([10.0] * 100)
        ewma_std = compute_ewma_std(series, alpha=0.3)
        # First value is NaN, rest should be ~0
        assert np.all(ewma_std.iloc[1:] < 1e-6)

    def test_ewma_gradient_constant(self) -> None:
        """Constant series → gradient should be ~0."""
        series = pd.Series([7.0] * 100)
        grad = compute_ewma_gradient(series, alpha=0.3)
        # First value is NaN, rest should be ~0
        assert np.allclose(grad.iloc[1:].values, 0.0, atol=1e-6)


class TestFFTFeatures:
    """Tests for FFT-based frequency features."""

    def test_fft_sine_wave(self) -> None:
        """Pure 1 Hz sine wave → dominant frequency ≈ 1.0 Hz."""
        sample_rate = 100.0  # 100 samples/sec
        t = np.arange(0, 2.56, 1.0 / sample_rate)  # 256 samples
        signal = np.sin(2 * np.pi * 1.0 * t)
        series = pd.Series(signal)

        feats = compute_fft_features(series, sample_rate=sample_rate)
        assert abs(feats["dominant_frequency_hz"] - 1.0) < 0.5, \
            f"Expected ~1.0 Hz, got {feats['dominant_frequency_hz']}"

    def test_fft_returns_scalars(self) -> None:
        """FFT features must all be scalar values (not arrays)."""
        series = pd.Series(np.random.randn(256))
        feats = compute_fft_features(series, sample_rate=1.0)
        for key, val in feats.items():
            if key == "top3_frequencies":
                assert isinstance(val, list) and len(val) == 3
            else:
                assert isinstance(val, (int, float)), f"{key} is not scalar: {type(val)}"

    def test_fft_short_series(self) -> None:
        """Very short series should return defaults without crashing."""
        series = pd.Series([1.0, 2.0])
        feats = compute_fft_features(series, sample_rate=1.0)
        assert feats["dominant_frequency_hz"] == 0.0


class TestGradientEstimator:
    """Tests for finite-difference gradient and congestion score."""

    def test_gradient_rising_trend(self) -> None:
        """Strictly increasing series → gradient > 0 everywhere."""
        series = pd.Series(np.arange(0, 100, 1.0))
        grad = finite_difference_gradient(series, order=1)
        assert (grad > 0).all(), "Gradient of increasing series must be > 0"

    def test_gradient_constant(self) -> None:
        """Constant series → gradient = 0."""
        series = pd.Series([3.0] * 50)
        grad = finite_difference_gradient(series, order=1)
        assert np.allclose(grad.values, 0.0, atol=1e-10)

    def test_congestion_onset_score_range(self) -> None:
        """Score must be in [0.0, 1.0] range."""
        rng = np.random.default_rng(42)
        rtt = pd.Series(rng.normal(20, 5, 200))
        util = pd.Series(rng.normal(1e6, 5e4, 200))
        score = congestion_onset_score(rtt, util)
        assert score.min() >= 0.0
        assert score.max() <= 1.0
