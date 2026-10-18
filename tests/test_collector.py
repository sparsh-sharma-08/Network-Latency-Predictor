"""Tests for the data collection layer."""

from __future__ import annotations

import pandas as pd
import pytest

from collector.mock_generator import generate_mock_telemetry


class TestMockGenerator:
    """Tests for the synthetic telemetry generator."""

    def test_mock_generator_shape(self) -> None:
        """Generate 1 day of data and verify row count (86400 @ 1s interval)."""
        df = generate_mock_telemetry(n_days=1, interval_sec=1.0, seed=42)
        assert len(df) == 86400, f"Expected 86400 rows, got {len(df)}"

    def test_mock_required_columns(self) -> None:
        """All required columns must be present."""
        df = generate_mock_telemetry(n_days=1, seed=42)
        required = [
            "timestamp", "rtt_ms", "packet_loss", "jitter_ms",
            "bytes_per_sec", "packets_per_sec", "congestion_label",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_mock_congestion_labels_present(self) -> None:
        """Congestion label column must contain both 0s and 1s."""
        df = generate_mock_telemetry(n_days=1, seed=42)
        labels = set(df["congestion_label"].unique())
        assert 0 in labels, "congestion_label must contain 0"
        assert 1 in labels, "congestion_label must contain 1"

    def test_mock_rtt_positive(self) -> None:
        """All RTT values must be positive."""
        df = generate_mock_telemetry(n_days=1, seed=42)
        assert (df["rtt_ms"] > 0).all(), "All rtt_ms must be > 0"

    def test_mock_timestamps_increasing(self) -> None:
        """Timestamps must be strictly increasing."""
        df = generate_mock_telemetry(n_days=1, seed=42)
        diffs = df["timestamp"].diff().iloc[1:]
        assert (diffs > 0).all(), "Timestamps must be monotonically increasing"

    def test_mock_reproducible(self) -> None:
        """Same seed should produce identical data."""
        df1 = generate_mock_telemetry(n_days=1, seed=123)
        df2 = generate_mock_telemetry(n_days=1, seed=123)
        pd.testing.assert_frame_equal(df1, df2)
