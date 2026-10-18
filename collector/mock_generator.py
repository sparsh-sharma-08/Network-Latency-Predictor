"""Synthetic network telemetry generator for development and testing.

Produces realistic mock data including daily/weekly traffic patterns,
congestion events with exponential decay, and correlated packet loss
and jitter signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def generate_mock_telemetry(
    n_days: int = 30,
    interval_sec: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic network telemetry DataFrame.

    The generated data simulates realistic network behaviour:
    - Base RTT ~20 ms with Gaussian noise (std=3 ms).
    - Daily pattern: +10 ms during 09:00-18:00, +5 ms during 18:00-22:00.
    - Weekly pattern: weekends have 30 % lower traffic.
    - Congestion events: ~5 per day, lasting 2–8 minutes, RTT spikes
      +50–200 ms with exponential decay back to baseline.
    - Packet loss: 0.1 % normally, 5–15 % during congestion.
    - Jitter correlated with congestion.

    Args:
        n_days: Number of days of data to generate.
        interval_sec: Sampling interval in seconds.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: timestamp, rtt_ms, packet_loss, jitter_ms,
        bytes_per_sec, packets_per_sec, congestion_label.
    """
    rng = np.random.default_rng(seed)

    total_seconds = int(n_days * 86400)
    n_samples = int(total_seconds / interval_sec)

    logger.info(
        "Generating {} samples ({} days, {}s interval)",
        n_samples, n_days, interval_sec,
    )

    # ── Timestamps (UTC epoch) ─────────────────────────────────
    start_ts = 1_700_000_000.0  # arbitrary fixed start
    timestamps = np.arange(n_samples) * interval_sec + start_ts

    # ── Base RTT with Gaussian noise ───────────────────────────
    base_rtt = 20.0
    noise_std = 3.0
    rtt = base_rtt + rng.normal(0, noise_std, n_samples)

    # ── Daily pattern ──────────────────────────────────────────
    seconds_in_day = (timestamps - start_ts) % 86400
    hours = seconds_in_day / 3600.0

    daily_boost = np.where(
        (hours >= 9) & (hours < 18),
        10.0,
        np.where((hours >= 18) & (hours < 22), 5.0, 0.0),
    )
    rtt += daily_boost

    # ── Weekly pattern (weekends = 30 % lower traffic) ────────
    days_from_start = ((timestamps - start_ts) / 86400).astype(int)
    day_of_week = days_from_start % 7  # 0=Mon … 6=Sun
    is_weekend = (day_of_week >= 5).astype(float)
    rtt -= is_weekend * 0.3 * base_rtt  # lower RTT on weekends

    # ── Base traffic metrics ───────────────────────────────────
    base_bps = 1_000_000.0  # 1 MB/s baseline
    base_pps = 1000.0  # 1000 packets/s baseline
    traffic_mult = 1.0 + daily_boost / 20.0 - is_weekend * 0.3

    bytes_per_sec = (
        base_bps * traffic_mult + rng.normal(0, 50_000, n_samples)
    )
    bytes_per_sec = np.clip(bytes_per_sec, 10_000, None)

    packets_per_sec = (
        base_pps * traffic_mult + rng.normal(0, 50, n_samples)
    )
    packets_per_sec = np.clip(packets_per_sec, 10, None)

    # ── Congestion events ──────────────────────────────────────
    congestion_label = np.zeros(n_samples, dtype=int)
    n_events = int(n_days * 5)

    for _ in range(n_events):
        # Random start point
        event_start = rng.integers(0, max(1, n_samples - 500))
        duration_sec = rng.integers(120, 480)  # 2–8 minutes
        duration_samples = int(duration_sec / interval_sec)
        spike_amplitude = rng.uniform(50, 200)
        decay_rate = rng.uniform(0.005, 0.02)

        event_end = min(event_start + duration_samples, n_samples)
        event_range = np.arange(event_end - event_start)

        # Exponential decay from spike amplitude
        spike = spike_amplitude * np.exp(-decay_rate * event_range)
        rtt[event_start:event_end] += spike
        congestion_label[event_start:event_end] = 1

        # Traffic also spikes during congestion
        bytes_per_sec[event_start:event_end] *= 1.5
        packets_per_sec[event_start:event_end] *= 1.3

    # ── Packet loss ────────────────────────────────────────────
    normal_loss_prob = 0.001
    congestion_loss_prob = rng.uniform(0.05, 0.15, n_samples)
    loss_prob = np.where(
        congestion_label == 1, congestion_loss_prob, normal_loss_prob
    )
    packet_loss = rng.random(n_samples) < loss_prob

    # ── Jitter (correlated with congestion) ────────────────────
    base_jitter = 1.0
    congestion_jitter = rng.uniform(5, 30, n_samples)
    jitter_ms = np.where(
        congestion_label == 1,
        congestion_jitter,
        base_jitter + rng.exponential(0.5, n_samples),
    )

    # ── Ensure RTT > 0 ────────────────────────────────────────
    rtt = np.clip(rtt, 0.1, None)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "rtt_ms": np.round(rtt, 3),
            "packet_loss": packet_loss.astype(int),
            "jitter_ms": np.round(jitter_ms, 3),
            "bytes_per_sec": np.round(bytes_per_sec, 1),
            "packets_per_sec": np.round(packets_per_sec, 1),
            "congestion_label": congestion_label,
        }
    )

    logger.info(
        "Generated {} rows | congestion={:.1%} | mean_rtt={:.1f}ms",
        len(df),
        congestion_label.mean(),
        rtt.mean(),
    )

    return df
