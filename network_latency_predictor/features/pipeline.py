"""Feature engineering pipeline — orchestrates all feature extractors.

Stateless and thread-safe: each ``transform()`` call creates no
side-effects and can be called concurrently from multiple threads.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import get_settings
from features.rolling_stats import compute_rolling_stats
from features.ewma import compute_ewma, compute_ewma_std, compute_ewma_gradient
from features.fft_features import compute_fft_features
from features.gradient_estimator import (
    finite_difference_gradient,
    congestion_onset_score,
)


class FeaturePipeline:
    """Orchestrates all feature extraction steps on a telemetry DataFrame.

    The pipeline is **stateless** — ``transform()`` has no side-effects.
    Column naming convention: ``{feature}_{window}s``
    (e.g. ``rtt_mean_5s``, ``rtt_p95_30s``, ``ewma_gradient_rtt``).

    Expected input columns:
        timestamp, rtt_ms, packet_loss, jitter_ms,
        bytes_per_sec, packets_per_sec
    """

    def __init__(self) -> None:
        """Initialise with global settings."""
        self.settings = get_settings()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature extractors and return an enriched DataFrame.

        Args:
            df: Raw telemetry DataFrame with required columns.

        Returns:
            Enriched DataFrame with ~45 feature columns plus originals.
            NaN rows from window edges are dropped.
        """
        required_cols = [
            "timestamp", "rtt_ms", "packet_loss",
            "jitter_ms", "bytes_per_sec", "packets_per_sec",
        ]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        logger.info("FeaturePipeline.transform — input rows: {}", len(df))

        windows = [
            ("short", self.settings.WINDOW_SHORT),
            ("medium", self.settings.WINDOW_MEDIUM),
            ("long", self.settings.WINDOW_LONG),
        ]

        # ── Rolling statistics for each window ────────────────
        # Build all new columns into a dict, then concat at the end
        # to avoid DataFrame fragmentation warnings.
        new_cols: Dict[str, pd.Series] = {}

        metrics = ["rtt_ms", "jitter_ms", "bytes_per_sec", "packets_per_sec"]
        for _label, win in windows:
            win_size = max(1, win)
            for metric in metrics:
                prefix = metric.replace("_ms", "").replace("_per_sec", "")
                stats = compute_rolling_stats(df[metric], window=win_size)
                for stat_name, stat_series in stats.items():
                    col_name = f"{prefix}_{stat_name}_{win}s"
                    new_cols[col_name] = stat_series.values

        # ── EWMA features on rtt_ms ──────────────────────────
        alpha = self.settings.EWMA_ALPHA
        new_cols["ewma_rtt"] = compute_ewma(df["rtt_ms"], alpha).values
        new_cols["ewma_std_rtt"] = compute_ewma_std(df["rtt_ms"], alpha).values
        new_cols["ewma_gradient_rtt"] = compute_ewma_gradient(df["rtt_ms"], alpha).values

        # ── FFT features over a 256-sample window ─────────────
        # Compute every 64 rows and forward-fill for performance
        fft_window = 256
        fft_step = 64
        fft_dominant = np.zeros(len(df))
        fft_entropy = np.zeros(len(df))
        fft_ratio = np.zeros(len(df))

        if len(df) >= fft_window:
            for i in range(fft_window, len(df) + 1, fft_step):
                window_series = df["rtt_ms"].iloc[i - fft_window : i]
                feats = compute_fft_features(window_series, sample_rate=1.0)
                end = min(i + fft_step, len(df))
                fft_dominant[i - 1 : end] = feats["dominant_frequency_hz"]
                fft_entropy[i - 1 : end] = feats["spectral_entropy"]
                fft_ratio[i - 1 : end] = feats["spectral_power_ratio"]

        new_cols["fft_dominant_freq"] = fft_dominant
        new_cols["fft_spectral_entropy"] = fft_entropy
        new_cols["fft_power_ratio"] = fft_ratio

        # ── Gradient features ─────────────────────────────────
        new_cols["rtt_gradient_1"] = finite_difference_gradient(df["rtt_ms"], order=1).values
        new_cols["rtt_gradient_2"] = finite_difference_gradient(df["rtt_ms"], order=2).values
        onset = congestion_onset_score(df["rtt_ms"], df["bytes_per_sec"])
        new_cols["congestion_onset_score"] = onset.values

        # ── Packet loss rolling mean ──────────────────────────
        new_cols["packet_loss_rate_5s"] = (
            df["packet_loss"].rolling(window=5, min_periods=1).mean().values
        )
        new_cols["packet_loss_rate_30s"] = (
            df["packet_loss"].rolling(window=30, min_periods=1).mean().values
        )

        # ── Concat all new columns at once to avoid fragmentation ──
        new_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_df], axis=1)

        # ── Drop NaN rows from window edges ───────────────────
        max_window = max(w for _, w in windows)
        df = df.iloc[max_window:].reset_index(drop=True)

        logger.info(
            "FeaturePipeline.transform — output rows: {}, columns: {}",
            len(df),
            len(df.columns),
        )

        return df

