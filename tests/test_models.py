"""Tests for prediction models."""

from __future__ import annotations

import numpy as np
import pytest

from models.queuing_model import QueuingModel
from models.xgboost_model import XGBoostModel
from models.ensemble import EnsembleModel


class TestQueuingModel:
    """Tests for the M/M/1 and M/D/1 queuing models."""

    def test_queuing_model_mm1(self) -> None:
        """M/M/1: λ=0.5, μ=1.0 → latency ≈ 1/(μ-λ) = 2.0 s = 2000 ms."""
        model = QueuingModel()
        model.mu = 1.0
        model.model_type = "mm1"
        model._baseline_rtt = 20.0

        X = np.array([[0.5, 1.0]])  # packets_per_sec=0.5, jitter=1.0
        pred = model.predict(X)
        rtt_10 = pred[0, 0]
        # 1/(1.0 - 0.5) = 2.0 s = 2000 ms
        assert abs(rtt_10 - 2000.0) < 100.0, f"Expected ~2000ms, got {rtt_10}"

    def test_queuing_utilization_clipped(self) -> None:
        """Congestion probability must be clipped to [0, 1]."""
        model = QueuingModel()
        model.mu = 100.0

        X = np.array([[150.0, 1.0]])  # λ > μ
        proba = model.predict_proba_congestion(X)
        assert 0.0 <= proba[0] <= 1.0

    def test_queuing_model_fit(self) -> None:
        """Fit should estimate μ from training data."""
        model = QueuingModel()
        rng = np.random.default_rng(42)

        X = np.column_stack([
            rng.uniform(800, 1200, 500),  # packets_per_sec
            rng.uniform(0.5, 3.0, 500),   # jitter
        ])
        y = rng.uniform(15, 30, 500)  # rtt_ms

        model.fit(X, y)
        assert model.mu > 0


class TestXGBoostModel:
    """Tests for the XGBoost gradient-boosted model."""

    def test_xgboost_overfit_small(self) -> None:
        """Tiny dataset → training MAE should be < 1 ms."""
        model = XGBoostModel()
        model._xgb_params["n_estimators"] = 100

        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, (n, 10)).astype(np.float32)
        y_rtt = np.column_stack([
            20 + X[:, 0] * 5,
            22 + X[:, 1] * 3,
            25 + X[:, 2] * 4,
        ]).astype(np.float32)
        jitter = rng.normal(2, 0.5, n).astype(np.float32)
        congestion = (X[:, 0] > 0.5).astype(np.float32)

        y = np.column_stack([y_rtt, jitter, congestion])
        model.fit(X, y)

        preds = model.predict(X)
        mae = np.mean(np.abs(preds[:, 0] - y_rtt[:, 0]))
        assert mae < 1.0, f"Train MAE should be < 1ms, got {mae:.2f}"

    def test_xgboost_output_shape(self) -> None:
        """Predict should return (n_samples, 3)."""
        model = XGBoostModel()
        model._xgb_params["n_estimators"] = 10

        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, (n, 5)).astype(np.float32)
        y = np.column_stack([
            rng.normal(20, 3, n),
            rng.normal(22, 3, n),
            rng.normal(25, 3, n),
            rng.normal(2, 1, n),
            rng.binomial(1, 0.1, n),
        ]).astype(np.float32)

        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (n, 3)


class TestEnsemble:
    """Tests for the weighted ensemble model."""

    def test_ensemble_weights_sum_to_one(self) -> None:
        """Normalised weights must sum to 1.0."""
        ensemble = EnsembleModel()

        # Add a dummy model
        queuing = QueuingModel()
        queuing.mu = 1000.0
        queuing._baseline_rtt = 20.0
        ensemble.add_model(queuing)

        weights = ensemble._normalize_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"

    def test_ensemble_predict_with_single_model(self) -> None:
        """Ensemble with one model should still produce valid output."""
        ensemble = EnsembleModel()

        queuing = QueuingModel()
        queuing.mu = 1000.0
        queuing._baseline_rtt = 20.0

        rng = np.random.default_rng(42)
        X_train = np.column_stack([
            rng.uniform(800, 1200, 100),
            rng.uniform(0.5, 3.0, 100),
        ])
        y_train = rng.uniform(15, 30, 100)
        queuing.fit(X_train, y_train)

        ensemble.add_model(queuing)

        X = np.array([[500.0, 1.0]])
        result = ensemble.predict(X)

        assert "rtt_10s" in result
        assert "rtt_30s" in result
        assert "rtt_60s" in result
        assert "congestion_prob" in result
        assert "congestion_label" in result


class TestLSTMModel:
    """Tests for the LSTM model."""

    def test_lstm_output_shape(self) -> None:
        """Random input → output shape should be (n_pred, 3) for RTT heads."""
        try:
            import torch
            from models.lstm_model import LSTMModel, TORCH_AVAILABLE

            if not TORCH_AVAILABLE:
                pytest.skip("PyTorch not available")

            model = LSTMModel(n_features=10, hidden_size=32, n_layers=1)

            rng = np.random.default_rng(42)
            n = 120  # 2× SEQ_LEN
            X = rng.normal(0, 1, (n, 10)).astype(np.float32)
            y = np.column_stack([
                rng.normal(20, 3, n),
                rng.normal(22, 3, n),
                rng.normal(25, 3, n),
                rng.binomial(1, 0.1, n).astype(np.float32),
            ]).astype(np.float32)

            model.fit(X, y)
            preds = model.predict(X)
            assert preds.shape[1] == 3, f"Expected 3 columns, got {preds.shape[1]}"

        except ImportError:
            pytest.skip("PyTorch not installed")
