"""Evaluation metrics for latency regression and congestion classification.

Provides MAE, RMSE, MAPE, R² for each prediction horizon and
precision / recall / F1 / AUC-ROC for the congestion classifier.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


class Evaluator:
    """Computes regression and classification metrics."""

    @staticmethod
    def evaluate_regression(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate regression metrics per horizon.

        Args:
            y_true: Ground truth array of shape (n, 3).
            y_pred: Predictions of shape (n, 3).

        Returns:
            Dict mapping horizon label to metric dict (mae, rmse, mape, r2).
        """
        horizons = ["10s", "30s", "60s"]
        results: Dict[str, Dict[str, float]] = {}

        for i, hz in enumerate(horizons):
            yt = y_true[:, i] if y_true.ndim == 2 else y_true
            yp = y_pred[:, i] if y_pred.ndim == 2 else y_pred

            mae = float(mean_absolute_error(yt, yp))
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))

            # MAPE — guard against division by zero
            mask = np.abs(yt) > 1e-6
            mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.any() else 0.0
            r2 = float(r2_score(yt, yp))

            results[hz] = {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 4),
                "r2": round(r2, 4),
            }

        return results

    @staticmethod
    def evaluate_congestion(
        y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, float]:
        """Evaluate binary congestion classification.

        Args:
            y_true: Ground truth labels (0 or 1).
            y_pred_proba: Predicted probabilities.
            threshold: Classification threshold.

        Returns:
            Dict with precision, recall, f1, auc_roc, average_precision.
        """
        y_pred_label = (y_pred_proba >= threshold).astype(int)
        y_true_int = y_true.astype(int)

        metrics: Dict[str, float] = {}

        metrics["precision"] = round(float(precision_score(y_true_int, y_pred_label, zero_division=0)), 4)
        metrics["recall"] = round(float(recall_score(y_true_int, y_pred_label, zero_division=0)), 4)
        metrics["f1"] = round(float(f1_score(y_true_int, y_pred_label, zero_division=0)), 4)

        try:
            metrics["auc_roc"] = round(float(roc_auc_score(y_true_int, y_pred_proba)), 4)
        except ValueError:
            metrics["auc_roc"] = 0.0

        try:
            metrics["average_precision"] = round(float(average_precision_score(y_true_int, y_pred_proba)), 4)
        except ValueError:
            metrics["average_precision"] = 0.0

        return metrics

    @staticmethod
    def evaluate_all(
        y_true_rtt: np.ndarray,
        y_pred_rtt: np.ndarray,
        y_true_cong: np.ndarray,
        y_pred_cong_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """Combined evaluation report for regression + classification.

        Args:
            y_true_rtt: True RTT values (n, 3).
            y_pred_rtt: Predicted RTT values (n, 3).
            y_true_cong: True congestion labels.
            y_pred_cong_proba: Predicted congestion probabilities.

        Returns:
            Combined report dict with 'regression' and 'congestion' keys.
        """
        regression = Evaluator.evaluate_regression(y_true_rtt, y_pred_rtt)
        congestion = Evaluator.evaluate_congestion(y_true_cong, y_pred_cong_proba)

        report = {
            "regression": regression,
            "congestion": congestion,
        }

        logger.info("Evaluation report: {}", report)
        return report
