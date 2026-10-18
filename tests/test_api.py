"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.main import create_app
from api.routes import set_ensemble
from models.ensemble import EnsembleModel
from models.queuing_model import QueuingModel


def _make_ensemble() -> EnsembleModel:
    """Create a minimal ensemble for testing."""
    ensemble = EnsembleModel()
    queuing = QueuingModel()
    queuing.mu = 1000.0
    queuing._baseline_rtt = 20.0

    rng = np.random.default_rng(42)
    X_train = np.column_stack([
        rng.uniform(800, 1200, 100),
        rng.uniform(0.5, 3.0, 100),
    ])
    queuing.fit(X_train, rng.uniform(15, 30, 100))
    ensemble.add_model(queuing)
    return ensemble


def _make_telemetry(n: int = 100) -> List[Dict[str, Any]]:
    """Generate n mock telemetry records."""
    rng = np.random.default_rng(42)
    base_ts = time.time()
    return [
        {
            "timestamp": base_ts + i,
            "rtt_ms": float(20 + rng.normal(0, 3)),
            "packet_loss": bool(rng.random() < 0.01),
            "jitter_ms": float(1 + rng.exponential(0.5)),
            "bytes_per_sec": float(1e6 + rng.normal(0, 5e4)),
            "packets_per_sec": float(1000 + rng.normal(0, 50)),
        }
        for i in range(n)
    ]


@pytest.fixture
def app():
    """Create test app with model injected."""
    test_app = create_app()
    test_app.state.start_time = time.time()
    from config.settings import get_settings
    test_app.state.settings = get_settings()
    ensemble = _make_ensemble()
    set_ensemble(ensemble)
    return test_app


@pytest.mark.asyncio
async def test_health_endpoint(app) -> None:
    """GET /health → 200, model_loaded=True."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert data["model_loaded"] is True


@pytest.mark.asyncio
async def test_predict_returns_valid_schema(app) -> None:
    """POST /predict with 500 samples → PredictionResponse is valid."""
    readings = _make_telemetry(500)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json=readings)

    assert resp.status_code == 200
    data = resp.json()
    assert "rtt_10s" in data
    assert "rtt_30s" in data
    assert "rtt_60s" in data
    assert "congestion_prob" in data
    assert "processing_time_ms" in data


@pytest.mark.asyncio
async def test_predict_requires_min_60_samples(app) -> None:
    """POST /predict with < 60 samples → 422 error."""
    readings = _make_telemetry(30)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json=readings)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_metrics_endpoint(app) -> None:
    """GET /metrics → contains 'nlp_predictions_total'."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "nlp_predictions_total" in resp.text
