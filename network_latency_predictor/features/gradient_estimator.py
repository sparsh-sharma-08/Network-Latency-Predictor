"""Finite-difference gradient estimator and congestion onset scorer.

Provides numerical differentiation via ``numpy.gradient`` and a
composite congestion-onset score combining RTT and utilization
gradients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def finite_difference_gradient(
    series: pd.Series, order: int = 1
) -> pd.Series:
    """Compute the nth-order finite-difference gradient of a series.

    Uses ``numpy.gradient`` which applies second-order central
    differences in the interior and first/second-order forward/backward
    differences at the boundaries.

    Args:
        series: Input time-series values.
        order: Derivative order (1 = first derivative, 2 = second, …).

    Returns:
        Gradient series of the same length as input.
    """
    values = series.values.astype(np.float64)

    for _ in range(order):
        values = np.gradient(values)

    return pd.Series(values, index=series.index, name=f"grad_order{order}")


def congestion_onset_score(
    rtt_series: pd.Series,
    utilization_series: pd.Series,
) -> pd.Series:
    """Compute a congestion-onset score combining RTT and utilization gradients.

    The score is min-max normalised to [0.0, 1.0].  Values > 0.7
    indicate congestion is likely developing.

    Args:
        rtt_series: RTT measurements (ms).
        utilization_series: Network utilization proxy (e.g. bytes_per_sec).

    Returns:
        Series of scores in [0.0, 1.0].
    """
    rtt_grad = finite_difference_gradient(rtt_series, order=1)
    util_grad = finite_difference_gradient(utilization_series, order=1)

    # Combine: both positive gradients → congestion onset
    combined = rtt_grad.clip(lower=0) + util_grad.clip(lower=0)

    # Min-max normalisation
    c_min = combined.min()
    c_max = combined.max()

    if c_max - c_min < 1e-12:
        return pd.Series(
            np.zeros(len(combined)), index=combined.index, name="congestion_onset_score"
        )

    normalised = (combined - c_min) / (c_max - c_min)
    normalised = normalised.clip(0.0, 1.0)
    normalised.name = "congestion_onset_score"
    return normalised
