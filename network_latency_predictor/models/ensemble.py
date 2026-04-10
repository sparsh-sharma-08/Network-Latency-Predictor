"""Weighted ensemble of all prediction models.

Combines ARIMA, XGBoost, LSTM, and queuing model predictions via
configurable weights.  Includes dynamic weight adjustment — if a model's
recent MAE exceeds 2× the average MAE, its weight is halved and
redistributed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from config.settings import get_settings
from models.base_model import BaseLatencyModel


class EnsembleModel:
    """Weighted ensemble combining all individual models.

    Attributes:
        models: Dictionary mapping model name to model instance.
        weights: Dictionary mapping model name to weight (must sum to 1).
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.models: Dict[str, BaseLatencyModel] = {}
        self.weights: Dict[str, float] = dict(self.settings.ENSEMBLE_WEIGHTS)
        self._recent_errors: Dict[str, List[float]] = {}

    def add_model(self, model: BaseLatencyModel) -> None:
        """Register a model in the ensemble.

        Args:
            model: An instance of BaseLatencyModel.
        """
        self.models[model.name] = model
        if model.name not in self.weights:
            self.weights[model.name] = 0.1
        self._recent_errors[model.name] = []
        logger.info("Ensemble: added model '{}'", model.name)

    def _normalize_weights(self) -> Dict[str, float]:
        """Normalize weights to sum to 1.0 for available models.

        Returns:
            Normalized weight dict.
        """
        available = {k: v for k, v in self.weights.items() if k in self.models}
        total = sum(available.values())
        if total <= 0:
            n = len(available)
            return {k: 1.0 / n for k in available} if n > 0 else {}
        return {k: v / total for k, v in available.items()}

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Generate a weighted ensemble prediction.

        If an individual model fails, it is skipped and its weight
        redistributed.

        Args:
            X: Feature matrix.

        Returns:
            Dictionary with keys: rtt_10s, rtt_30s, rtt_60s,
            congestion_prob, congestion_label, model_contributions.
        """
        weights = self._normalize_weights()
        predictions: Dict[str, np.ndarray] = {}
        congestion_probs: Dict[str, float] = {}
        failed_models: List[str] = []

        for model_name, model in self.models.items():
            try:
                pred = model.predict(X)
                if pred.ndim == 1:
                    pred = pred.reshape(1, -1)
                # Use last prediction row
                predictions[model_name] = pred[-1]

                cong = model.predict_proba_congestion(X)
                congestion_probs[model_name] = float(cong[-1] if len(cong) > 0 else 0.0)
            except Exception as exc:
                logger.error("Model '{}' prediction failed: {}", model_name, exc)
                failed_models.append(model_name)

        if not predictions:
            logger.error("All models failed — returning defaults")
            return {
                "rtt_10s": 20.0,
                "rtt_30s": 20.0,
                "rtt_60s": 20.0,
                "congestion_prob": 0.0,
                "congestion_label": False,
                "model_contributions": {},
            }

        # Recalculate weights excluding failed models
        active_weights = {k: v for k, v in weights.items() if k not in failed_models and k in predictions}
        total_w = sum(active_weights.values())
        if total_w > 0:
            active_weights = {k: v / total_w for k, v in active_weights.items()}
        else:
            n = len(predictions)
            active_weights = {k: 1.0 / n for k in predictions}

        # Weighted average
        rtt_ensemble = np.zeros(3)
        cong_ensemble = 0.0
        contributions: Dict[str, Dict[str, float]] = {}

        for model_name, pred in predictions.items():
            w = active_weights.get(model_name, 0.0)
            weighted_pred = pred[:3] * w
            rtt_ensemble += weighted_pred

            cong_w = congestion_probs.get(model_name, 0.0) * w
            cong_ensemble += cong_w

            contributions[model_name] = {
                "weight": round(w, 4),
                "rtt_10s": round(float(pred[0]), 3) if len(pred) > 0 else 0.0,
                "rtt_30s": round(float(pred[1]), 3) if len(pred) > 1 else 0.0,
                "rtt_60s": round(float(pred[2]), 3) if len(pred) > 2 else 0.0,
                "congestion_prob": round(congestion_probs.get(model_name, 0.0), 4),
            }

        return {
            "rtt_10s": round(float(rtt_ensemble[0]), 3),
            "rtt_30s": round(float(rtt_ensemble[1]), 3),
            "rtt_60s": round(float(rtt_ensemble[2]), 3),
            "congestion_prob": round(float(min(cong_ensemble, 1.0)), 4),
            "congestion_label": cong_ensemble > 0.7,
            "model_contributions": contributions,
        }

    def update_errors(self, model_name: str, mae: float) -> None:
        """Record recent MAE for dynamic weight adjustment.

        If a model's MAE exceeds 2× the average MAE across all models,
        its weight is halved and the excess redistributed.

        Args:
            model_name: Name of the model.
            mae: Recent mean absolute error.
        """
        if model_name not in self._recent_errors:
            self._recent_errors[model_name] = []

        self._recent_errors[model_name].append(mae)
        # Keep only last 50 entries
        self._recent_errors[model_name] = self._recent_errors[model_name][-50:]

        # Dynamic adjustment
        all_maes = []
        for errors in self._recent_errors.values():
            if errors:
                all_maes.append(np.mean(errors))

        if len(all_maes) < 2:
            return

        avg_mae = np.mean(all_maes)
        model_avg = np.mean(self._recent_errors[model_name])

        if model_avg > 2 * avg_mae:
            old_weight = self.weights.get(model_name, 0.0)
            reduction = old_weight / 2.0
            self.weights[model_name] = old_weight - reduction

            # Redistribute
            other_models = [n for n in self.models if n != model_name]
            if other_models:
                bonus = reduction / len(other_models)
                for other in other_models:
                    self.weights[other] = self.weights.get(other, 0.0) + bonus

            logger.warning(
                "Model '{}' MAE ({:.2f}) > 2× avg ({:.2f}) — weight reduced from {:.3f} to {:.3f}",
                model_name, model_avg, avg_mae, old_weight, self.weights[model_name],
            )

    def save(self, directory: str) -> None:
        """Save all models to a directory.

        Args:
            directory: Target directory.
        """
        os.makedirs(directory, exist_ok=True)

        for model_name, model in self.models.items():
            model_path = os.path.join(directory, f"{model_name}_model.pkl")
            try:
                model.save(model_path)
            except Exception as exc:
                logger.error("Failed to save model '{}': {}", model_name, exc)

        # Save weights
        import json
        weights_path = os.path.join(directory, "ensemble_weights.json")
        with open(weights_path, "w") as f:
            json.dump(self.weights, f, indent=2)

        logger.info("Ensemble saved to {}", directory)

    @classmethod
    def load(cls, directory: str) -> "EnsembleModel":
        """Load an ensemble from a directory.

        Args:
            directory: Directory containing saved models.

        Returns:
            Loaded EnsembleModel instance.
        """
        import json
        from models.arima_model import ARIMAModel
        from models.xgboost_model import XGBoostModel
        from models.lstm_model import LSTMModel
        from models.queuing_model import QueuingModel

        ensemble = cls()

        # Load weights
        weights_path = os.path.join(directory, "ensemble_weights.json")
        if os.path.exists(weights_path):
            with open(weights_path) as f:
                ensemble.weights = json.load(f)

        # Load each model
        model_classes: Dict[str, type] = {
            "arima": ARIMAModel,
            "xgboost": XGBoostModel,
            "lstm": LSTMModel,
            "queuing": QueuingModel,
        }

        for model_name, model_cls in model_classes.items():
            model_path = os.path.join(directory, f"{model_name}_model.pkl")
            if os.path.exists(model_path):
                try:
                    model = model_cls.load(model_path)
                    ensemble.add_model(model)
                except Exception as exc:
                    logger.error("Failed to load model '{}': {}", model_name, exc)
            else:
                logger.warning("Model file not found: {}", model_path)

        return ensemble
