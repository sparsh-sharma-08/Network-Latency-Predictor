"""Rolling window statistics for time-series telemetry.

Computes descriptive statistics over sliding windows using pandas
rolling operations — no manual loops.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
def compute_rolling_stats(series: pd.Series, window: int) -> Dict[str, pd.Series]:
    """Compute rolling descriptive statistics for a pandas Series.

    All statistics are computed using ``pd.Series.rolling`` with
    C-optimized aggregations — no Python-level ``apply`` loops.

    Args:
        series: Input time-series values (e.g. rtt_ms).
        window: Rolling window size in number of samples.

    Returns:
        Dictionary mapping stat name to a pandas Series of the same
        length as the input (leading NaNs for incomplete windows).
        Keys: mean, std, min, max, p50, p95, p99, iqr, skewness, kurtosis.
    """
    roller = series.rolling(window=window, min_periods=1)

    result: Dict[str, pd.Series] = {
        "mean": roller.mean(),
        "std": roller.std(),
        "min": roller.min(),
        "max": roller.max(),
        "p50": roller.quantile(0.50),
        "p95": roller.quantile(0.95),
        "p99": roller.quantile(0.99),
        "iqr": roller.quantile(0.75) - roller.quantile(0.25),
        "skewness": roller.skew().fillna(0.0),
        "kurtosis": roller.kurt().fillna(0.0),
    }

    return result
