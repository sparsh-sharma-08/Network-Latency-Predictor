"""Dataset loader with target creation and time-based splits.

Provides ``LatencyDataset`` for loading telemetry data, creating
shifted target columns, and producing train / val / test splits that
avoid temporal data leakage.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger


class LatencyDataset:
    """Manages telemetry data for model training.

    Handles CSV / DataFrame loading, target column creation (shifted
    RTT and congestion labels), and chronological train/val/test
    splitting.
    """

    def __init__(self) -> None:
        self.df: pd.DataFrame = pd.DataFrame()

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """Load telemetry from a CSV file.

        Args:
            path: Path to the CSV file.

        Returns:
            Loaded DataFrame.
        """
        self.df = pd.read_csv(path)
        logger.info("Loaded {} rows from {}", len(self.df), path)
        return self.df

    def load_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load telemetry from an existing DataFrame.

        Args:
            df: Source DataFrame.

        Returns:
            The same DataFrame (stored internally).
        """
        self.df = df.copy()
        logger.info("Loaded {} rows from DataFrame", len(self.df))
        return self.df

    @staticmethod
    def create_targets(df: pd.DataFrame) -> pd.DataFrame:
        """Add shifted target columns for supervised learning.

        Creates:
          - rtt_10s_ahead: rtt_ms shifted backward by 10 rows
          - rtt_30s_ahead: rtt_ms shifted backward by 30 rows
          - rtt_60s_ahead: rtt_ms shifted backward by 60 rows
          - congestion_ahead: congestion_label shifted backward by 10 rows

        Rows with NaN targets (at the tail) are dropped.

        Args:
            df: DataFrame that must contain ``rtt_ms`` and
                ``congestion_label`` columns.

        Returns:
            DataFrame with added target columns, tail NaNs dropped.
        """
        df = df.copy()
        df["rtt_10s_ahead"] = df["rtt_ms"].shift(-10)
        df["rtt_30s_ahead"] = df["rtt_ms"].shift(-30)
        df["rtt_60s_ahead"] = df["rtt_ms"].shift(-60)

        if "congestion_label" in df.columns:
            df["congestion_ahead"] = df["congestion_label"].shift(-10)
        else:
            df["congestion_ahead"] = 0

        df = df.dropna(subset=[
            "rtt_10s_ahead", "rtt_30s_ahead", "rtt_60s_ahead", "congestion_ahead"
        ]).reset_index(drop=True)

        logger.info("Created targets — {} rows remain after dropping NaN tails", len(df))
        return df

    @staticmethod
    def train_val_test_split(
        df: pd.DataFrame,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data chronologically (no random shuffle) to prevent leakage.

        Train = first 70 %, Val = next 15 %, Test = last 15 %.

        Args:
            df: Full DataFrame (should be sorted by timestamp).
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        n = len(df)
        train_end = int(n * (1 - val_ratio - test_ratio))
        val_end = int(n * (1 - test_ratio))

        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = df.iloc[val_end:].reset_index(drop=True)

        logger.info(
            "Split: train={}, val={}, test={}",
            len(train_df), len(val_df), len(test_df),
        )
        return train_df, val_df, test_df
