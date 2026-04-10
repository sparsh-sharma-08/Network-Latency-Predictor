"""Exponential Weighted Moving Average (EWMA) feature extractors.

Provides EWMA smoothing, volatility (EWMA std), and gradient
(rate-of-change of the smoothed signal) — the key congestion onset
indicator.
"""

from __future__ import annotations

import pandas as pd


def compute_ewma(series: pd.Series, alpha: float) -> pd.Series:
    """Compute the exponential weighted moving average.

    Args:
        series: Raw time-series values.
        alpha: Smoothing factor in (0, 1). Higher = more responsive.

    Returns:
        EWMA-smoothed series.
    """
    return series.ewm(alpha=alpha, adjust=False).mean()


def compute_ewma_std(series: pd.Series, alpha: float) -> pd.Series:
    """Compute the exponential weighted standard deviation (volatility).

    Args:
        series: Raw time-series values.
        alpha: Smoothing factor in (0, 1).

    Returns:
        EWMA standard deviation series.
    """
    return series.ewm(alpha=alpha, adjust=False).std()


def compute_ewma_gradient(series: pd.Series, alpha: float) -> pd.Series:
    """Compute the first derivative of the EWMA — rate of change.

    This is the primary congestion onset signal: a positive gradient
    with increasing magnitude suggests latency is rising.

    Args:
        series: Raw time-series values.
        alpha: Smoothing factor for EWMA.

    Returns:
        First-order difference of the EWMA-smoothed series.
    """
    ewma = compute_ewma(series, alpha)
    return ewma.diff()
