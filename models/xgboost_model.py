"""XGBoost gradient-boosted tree model for latency and congestion prediction.

Uses three sub-models:
  1. XGBRegressor for RTT prediction
  2. XGBRegressor for jitter prediction
  3. XGBClassifier for congestion probability
Supports SHAP-based explainability.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from loguru import logger

from models.base_model import BaseLatencyModel

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost not installed — XGBoostModel unavailable")


class XGBoostModel(BaseLatencyModel):
    """Gradient-boosted model for multi-horizon latency prediction.

    Trains separate regressors for each RTT horizon (10 s, 30 s, 60 s)
    and a single classifier for congestion probability.

    Attributes:
        regressors: Dict mapping horizon label to XGBRegressor.
        classifier: XGBClassifier for congestion labels.
    """

    def __init__(self) -> None:
        self.regressors: Dict[str, Any] = {}
        self.jitter_regressor: Optional[Any] = None
        self.classifier: Optional[Any] = None
        self._feature_names: List[str] = []
        self._feature_importances: Dict[str, np.ndarray] = {}

        self._xgb_params = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """Train regressors and classifier.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target matrix with columns:
               [rtt_10s, rtt_30s, rtt_60s, jitter, congestion_label].
               If y has fewer columns, only available targets are used.
            **kwargs: Optional ``eval_set`` for early stopping.
        """
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost is required for XGBoostModel")

        self._feature_names = kwargs.get("feature_names", [])

        eval_set = kwargs.get("eval_set")
        fit_params: Dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False

        horizons = ["rtt_10s", "rtt_30s", "rtt_60s"]
        for i, hz in enumerate(horizons):
            if y.shape[1] <= i:
                break
            logger.info("Training XGBRegressor for {}", hz)
            reg = XGBRegressor(**self._xgb_params)

            if eval_set is not None:
                X_val, y_val = eval_set[0]
                reg.fit(
                    X, y[:, i],
                    eval_set=[(X_val, y_val[:, i])],
                    verbose=False,
                )
            else:
                reg.fit(X, y[:, i])

            self.regressors[hz] = reg
            self._feature_importances[hz] = reg.feature_importances_

        # Jitter regressor
        if y.shape[1] > 3:
            logger.info("Training XGBRegressor for jitter")
            self.jitter_regressor = XGBRegressor(**self._xgb_params)
            if eval_set is not None:
                X_val, y_val = eval_set[0]
                self.jitter_regressor.fit(
                    X, y[:, 3],
                    eval_set=[(X_val, y_val[:, 3])],
                    verbose=False,
                )
            else:
                self.jitter_regressor.fit(X, y[:, 3])

        # Congestion classifier
        if y.shape[1] > 4:
            logger.info("Training XGBClassifier for congestion")
            self.classifier = XGBClassifier(
                **self._xgb_params,
                objective="binary:logistic",
                eval_metric="logloss",
            )
            if eval_set is not None:
                X_val, y_val = eval_set[0]
                self.classifier.fit(
                    X, y[:, 4].astype(int),
                    eval_set=[(X_val, y_val[:, 4].astype(int))],
                    verbose=False,
                )
            else:
                self.classifier.fit(X, y[:, 4].astype(int))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RTT at 10 s, 30 s, and 60 s horizons.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 3).
        """
        results = []
        for hz in ["rtt_10s", "rtt_30s", "rtt_60s"]:
            if hz in self.regressors:
                pred = self.regressors[hz].predict(X)
                results.append(np.clip(pred, 0, None))
            else:
                results.append(np.full(X.shape[0], 20.0))

        return np.column_stack(results)

    def predict_proba_congestion(self, X: np.ndarray) -> np.ndarray:
        """Predict congestion probability.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples,) with probabilities in [0, 1].
        """
        if self.classifier is None:
            return np.zeros(X.shape[0])

        proba = self.classifier.predict_proba(X)
        # Column 1 = probability of class 1 (congestion)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]

    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        """Compute SHAP values for explainability.

        Args:
            X: Feature matrix.

        Returns:
            Dictionary mapping horizon to SHAP value arrays.
        """
        explanations: Dict[str, Any] = {}
        try:
            import shap
            for hz, reg in self.regressors.items():
                explainer = shap.TreeExplainer(reg)
                shap_values = explainer.shap_values(X)
                explanations[hz] = shap_values
        except ImportError:
            logger.warning("shap not installed — explain() unavailable")
        except Exception as exc:
            logger.error("SHAP explanation failed: {}", exc)

        return explanations

    def save(self, path: str) -> None:
        """Save all sub-models to disk.

        Args:
            path: Base path (without extension).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "regressors": self.regressors,
                "jitter_regressor": self.jitter_regressor,
                "classifier": self.classifier,
                "feature_names": self._feature_names,
                "feature_importances": self._feature_importances,
                "xgb_params": self._xgb_params,
            },
            path,
        )
        logger.info("XGBoostModel saved to {}", path)

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        """Load a saved XGBoostModel.

        Args:
            path: Path to the saved model file.

        Returns:
            Restored XGBoostModel instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        model = cls()
        model.regressors = data["regressors"]
        model.jitter_regressor = data["jitter_regressor"]
        model.classifier = data["classifier"]
        model._feature_names = data.get("feature_names", [])
        model._feature_importances = data.get("feature_importances", {})
        model._xgb_params = data.get("xgb_params", model._xgb_params)
        logger.info("XGBoostModel loaded from {}", path)
        return model
