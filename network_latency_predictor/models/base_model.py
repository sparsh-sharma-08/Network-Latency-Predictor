"""Abstract base class for all latency prediction models.

Every model in the system must inherit from ``BaseLatencyModel`` and
implement the ``fit``, ``predict``, and ``predict_proba_congestion``
methods.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional

import joblib
import numpy as np
from loguru import logger


class BaseLatencyModel(ABC):
    """Abstract base for latency / congestion prediction models.

    Subclasses must implement ``fit``, ``predict``, and
    ``predict_proba_congestion``.  Serialization helpers (``save`` /
    ``load``) are provided via joblib for sklearn-compatible models;
    PyTorch models should override as needed.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier (e.g. 'arima', 'xgboost')."""
        ...

    @property
    def feature_names(self) -> List[str]:
        """Return the list of feature names the model was trained on.

        Override in subclasses that track feature names.
        """
        return getattr(self, "_feature_names", [])

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on feature matrix X and targets y.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target array (shape depends on model).
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict latency for given features.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 3) with predictions for
            t+10 s, t+30 s, t+60 s horizons.
        """
        ...

    @abstractmethod
    def predict_proba_congestion(self, X: np.ndarray) -> np.ndarray:
        """Predict congestion probability.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples,) with probabilities in [0, 1].
        """
        ...

    def save(self, path: str) -> None:
        """Serialize the model to disk.

        Args:
            path: File path (directory will be created if needed).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)
        logger.info("Model '{}' saved to {}", self.name, path)

    @classmethod
    def load(cls, path: str) -> "BaseLatencyModel":
        """Deserialize a model from disk.

        Args:
            path: File path to the saved model.

        Returns:
            Loaded model instance.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        model = joblib.load(path)
        logger.info("Model '{}' loaded from {}", model.name, path)
        return model
