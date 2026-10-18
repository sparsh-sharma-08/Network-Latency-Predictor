"""M/M/1 and M/D/1 queuing-theory analytical models for latency prediction.

Automatically selects M/M/1 or M/D/1 based on the coefficient of
variation of inter-arrival times.
"""

from __future__ import annotations

from typing import List

import numpy as np
from loguru import logger

from models.base_model import BaseLatencyModel


class QueuingModel(BaseLatencyModel):
    """Analytical queuing model for latency estimation.

    M/M/1: latency = 1 / (μ − λ)
    M/D/1: latency = (1 / (μ − λ)) × (1 − λ / (2μ))

    Selection is automatic: if CV of inter-arrival times < 0.5 → M/D/1,
    otherwise M/M/1.

    Attributes:
        mu: Estimated service rate (packets/sec) derived from training.
        model_type: 'mm1' or 'md1'.
    """

    def __init__(self) -> None:
        self.mu: float = 1000.0  # default service rate
        self.model_type: str = "mm1"
        self._baseline_rtt: float = 20.0
        self._feature_names: List[str] = ["packets_per_sec", "jitter_ms"]

    @property
    def name(self) -> str:
        return "queuing"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Estimate service rate μ from training data.

        μ is derived from the minimum observed RTT and average
        packet rate:  μ ≈ avg_pps × (baseline_rtt / min_rtt).

        Also determines whether to use M/M/1 or M/D/1 based on
        inter-arrival time variation.

        Args:
            X: Feature matrix — expects at least columns for
               packets_per_sec (col 0) and jitter_ms (col 1).
            y: Target rtt_ms values (1-D array).
        """
        rtt = y.flatten() if y.ndim > 1 else y
        self._baseline_rtt = float(np.median(rtt))
        min_rtt = float(np.min(rtt[rtt > 0])) if np.any(rtt > 0) else 1.0

        # Extract packets_per_sec (first column)
        if X.ndim == 2 and X.shape[1] >= 1:
            pps = X[:, 0]
        else:
            pps = np.full(len(rtt), 1000.0)

        avg_pps = float(np.mean(pps))
        self.mu = avg_pps * (self._baseline_rtt / min_rtt) * 1.2  # headroom

        # Determine model type from CV of inter-arrival proxy (jitter)
        if X.ndim == 2 and X.shape[1] >= 2:
            jitter = X[:, 1]
            cv = float(np.std(jitter) / (np.mean(jitter) + 1e-9))
            self.model_type = "md1" if cv < 0.5 else "mm1"
        else:
            self.model_type = "mm1"

        logger.info(
            "QueuingModel fitted: mu={:.1f}, type={}, baseline_rtt={:.1f}ms",
            self.mu, self.model_type, self._baseline_rtt,
        )

    def _compute_latency(self, lam: float) -> float:
        """Compute queuing latency for arrival rate λ.

        Args:
            lam: Arrival rate (packets/sec).

        Returns:
            Estimated latency in milliseconds.
        """
        if lam >= self.mu:
            # System at or over capacity — return high latency
            return self._baseline_rtt * 10.0

        if self.model_type == "md1":
            latency_sec = (1.0 / (self.mu - lam)) * (1.0 - lam / (2.0 * self.mu))
        else:
            latency_sec = 1.0 / (self.mu - lam)

        latency_ms = latency_sec * 1000.0
        return max(latency_ms, 0.1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RTT for t+10 s, t+30 s, t+60 s.

        The queuing model returns the same value for all horizons
        (steady-state assumption) but scales by a growth factor if
        utilization is trending up.

        Args:
            X: Feature matrix with packets_per_sec as first column.

        Returns:
            Array of shape (n_samples, 3).
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        results = []
        for i in range(X.shape[0]):
            lam = float(X[i, 0]) if X.shape[1] >= 1 else 500.0
            base_lat = self._compute_latency(lam)

            # Slight scaling for longer horizons (assumption: load may grow)
            rtt_10 = base_lat
            rtt_30 = base_lat * 1.05
            rtt_60 = base_lat * 1.10
            results.append([rtt_10, rtt_30, rtt_60])

        return np.array(results)

    def predict_proba_congestion(self, X: np.ndarray) -> np.ndarray:
        """Return utilization ratio λ/μ as congestion probability.

        Args:
            X: Feature matrix with packets_per_sec as first column.

        Returns:
            Array of shape (n_samples,) clipped to [0, 1].
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        probs = []
        for i in range(X.shape[0]):
            lam = float(X[i, 0]) if X.shape[1] >= 1 else 500.0
            rho = lam / self.mu if self.mu > 0 else 1.0
            probs.append(min(max(rho, 0.0), 1.0))

        return np.array(probs)
