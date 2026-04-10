"""ARIMA baseline model for univariate RTT forecasting.

Uses ``statsmodels.SARIMAX`` with AIC-based grid search for (p, d, q)
order selection.
"""

from __future__ import annotations

import warnings
from itertools import product
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from models.base_model import BaseLatencyModel

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class ARIMAModel(BaseLatencyModel):
    """Univariate ARIMA model fitted on the rtt_ms time series.

    Selects optimal (p, d, q) via AIC minimisation over a small grid.
    Forecasts RTT at t+10 s, t+30 s, t+60 s horizons.
    """

    def __init__(self) -> None:
        self._model = None
        self._fitted = None
        self._order: Tuple[int, int, int] = (1, 1, 1)
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 1.0
        self._feature_names: List[str] = ["rtt_ms"]

    @property
    def name(self) -> str:
        return "arima"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit ARIMA on the RTT series (y) using AIC grid search.

        Args:
            X: Ignored (ARIMA is univariate).
            y: 1-D array of rtt_ms values.
        """
        if not STATSMODELS_AVAILABLE:
            raise RuntimeError("statsmodels is required for ARIMAModel")

        series = np.asarray(y).flatten()
        self._baseline_mean = float(np.mean(series))
        self._baseline_std = float(np.std(series))

        best_aic = float("inf")
        best_order = (1, 1, 1)

        p_range = [0, 1, 2]
        d_range = [0, 1]
        q_range = [0, 1, 2]

        for p, d, q in product(p_range, d_range, q_range):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(series, order=(p, d, q), enforce_stationarity=False,
                                    enforce_invertibility=False)
                    fitted = model.fit(disp=False, maxiter=100)
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                continue

        self._order = best_order
        logger.info("ARIMA best order: {} with AIC={:.2f}", best_order, best_aic)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = SARIMAX(series, order=best_order,
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
            self._fitted = self._model.fit(disp=False, maxiter=200)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast RTT at 10 s, 30 s, and 60 s ahead.

        Args:
            X: Ignored (uses internal state).

        Returns:
            Array of shape (1, 3) with [rtt_10s, rtt_30s, rtt_60s].
        """
        if self._fitted is None:
            raise RuntimeError("Model not fitted — call fit() first")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._fitted.forecast(steps=60)

        rtt_10 = float(forecast[9]) if len(forecast) > 9 else float(forecast[-1])
        rtt_30 = float(forecast[29]) if len(forecast) > 29 else float(forecast[-1])
        rtt_60 = float(forecast[59]) if len(forecast) > 59 else float(forecast[-1])

        return np.array([[max(0.0, rtt_10), max(0.0, rtt_30), max(0.0, rtt_60)]])

    def predict_proba_congestion(self, X: np.ndarray) -> np.ndarray:
        """Return congestion probability based on forecast vs. baseline.

        Returns 1.0 if forecast > baseline_mean + 2 * baseline_std.

        Args:
            X: Ignored.

        Returns:
            Array of shape (1,) with congestion probability.
        """
        if self._fitted is None:
            return np.array([0.0])

        preds = self.predict(X)
        avg_pred = float(np.mean(preds))
        threshold = self._baseline_mean + 2 * self._baseline_std

        if avg_pred > threshold:
            return np.array([1.0])
        elif avg_pred > self._baseline_mean + self._baseline_std:
            ratio = (avg_pred - self._baseline_mean) / (2 * self._baseline_std + 1e-9)
            return np.array([min(1.0, max(0.0, ratio))])
        else:
            return np.array([0.0])
