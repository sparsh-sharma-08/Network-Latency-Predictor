"""Model drift detection with Page-Hinkley test.

Monitors prediction accuracy over a rolling window and triggers
retraining when drift is detected.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import get_settings


@dataclass
class DriftReport:
    """Result of a drift check.

    Attributes:
        drift_detected: True if MAE exceeds threshold or PH test signals.
        current_mae: Rolling MAE of recent predictions.
        baseline_mae: MAE from training evaluation.
        ph_statistic: Page-Hinkley test statistic.
        samples_evaluated: Number of samples in the buffer.
    """

    drift_detected: bool = False
    current_mae: float = 0.0
    baseline_mae: float = 0.0
    ph_statistic: float = 0.0
    samples_evaluated: int = 0


class DriftDetector:
    """Detects model drift via MAE monitoring and Page-Hinkley test.

    Stores recent ``(y_true, y_pred)`` pairs in a rolling buffer and
    computes drift metrics.  When drift is detected, it can optionally
    trigger a retrain request via the API (with a minimum 6-hour
    cooldown to avoid retrain loops).

    Attributes:
        settings: Application settings.
        buffer: Rolling buffer of (y_true, y_pred) pairs.
        baseline_mae: MAE recorded at training time.
    """

    def __init__(self, baseline_mae: float = 2.0) -> None:
        """Initialize the drift detector.

        Args:
            baseline_mae: MAE from the initial training evaluation.
        """
        self.settings = get_settings()
        self.baseline_mae = baseline_mae
        self.buffer: Deque[Tuple[float, float]] = deque(
            maxlen=self.settings.DRIFT_WINDOW_SIZE
        )
        self._last_retrain_time: float = 0.0
        self._min_retrain_interval: float = 6 * 3600  # 6 hours

        # Page-Hinkley state
        self._ph_sum: float = 0.0
        self._ph_min: float = float("inf")
        self._ph_count: int = 0
        self._ph_mean: float = 0.0
        self._ph_threshold: float = 50.0  # PH test threshold
        self._ph_delta: float = 0.005  # allowable deviation magnitude

    def add_observation(self, y_true: float, y_pred: float) -> None:
        """Add a single (y_true, y_pred) observation.

        Args:
            y_true: Actual observed RTT.
            y_pred: Model-predicted RTT.
        """
        self.buffer.append((y_true, y_pred))

        # Update Page-Hinkley
        error = abs(y_true - y_pred)
        self._ph_count += 1
        self._ph_mean += (error - self._ph_mean) / self._ph_count
        self._ph_sum += error - self._ph_mean - self._ph_delta
        self._ph_min = min(self._ph_min, self._ph_sum)

    def check(self) -> DriftReport:
        """Evaluate drift on the current buffer.

        Returns:
            DriftReport with drift detection results.
        """
        if len(self.buffer) < 10:
            return DriftReport(samples_evaluated=len(self.buffer))

        errors = np.array([abs(yt - yp) for yt, yp in self.buffer])
        current_mae = float(np.mean(errors))

        # Page-Hinkley statistic
        ph_stat = self._ph_sum - self._ph_min

        mae_drift = current_mae > self.settings.DRIFT_THRESHOLD_MAE
        ph_drift = ph_stat > self._ph_threshold

        report = DriftReport(
            drift_detected=mae_drift or ph_drift,
            current_mae=round(current_mae, 4),
            baseline_mae=round(self.baseline_mae, 4),
            ph_statistic=round(ph_stat, 4),
            samples_evaluated=len(self.buffer),
        )

        if report.drift_detected:
            logger.warning(
                "DRIFT DETECTED — MAE={:.3f} (threshold={:.1f}), PH={:.3f}",
                current_mae,
                self.settings.DRIFT_THRESHOLD_MAE,
                ph_stat,
            )

        return report

    async def trigger_retrain(self) -> bool:
        """Trigger retraining via the API if cooldown has elapsed.

        Returns:
            True if retrain was triggered, False if cooldown prevented it.
        """
        now = time.time()
        if now - self._last_retrain_time < self._min_retrain_interval:
            logger.info(
                "Retrain skipped — cooldown active (last retrain {:.0f}s ago)",
                now - self._last_retrain_time,
            )
            return False

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://localhost:{self.settings.API_PORT}/retrain",
                    json={"days_of_history": 7},
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    self._last_retrain_time = now
                    logger.info("Retrain triggered successfully")
                    return True
                else:
                    logger.error("Retrain request failed: {}", resp.text)
        except Exception as exc:
            logger.error("Failed to trigger retrain: {}", exc)

        return False
