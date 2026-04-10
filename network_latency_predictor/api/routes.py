"""API route definitions — /predict, /health, /metrics, /retrain.

All endpoints log structured data, record Prometheus metrics, and
handle errors gracefully (never returning raw 500s).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    RetrainRequest,
    RetrainStatusResponse,
    TelemetryInput,
)
from features.pipeline import FeaturePipeline
from monitoring.prometheus_exporter import (
    CONGESTION_ALERT_COUNTER,
    CONGESTION_GAUGE,
    LATENCY_HISTOGRAM,
    PREDICTION_COUNTER,
    RTT_GAUGE,
)

router = APIRouter()

# Module-level state (set by lifespan in main.py)
_ensemble = None
_pipeline = FeaturePipeline()
_last_prediction_ts: float | None = None
_retrain_jobs: Dict[str, Dict[str, Any]] = {}


def set_ensemble(ensemble: Any) -> None:
    """Inject the ensemble model (called by main.py lifespan)."""
    global _ensemble
    _ensemble = ensemble


def get_last_prediction_ts() -> float | None:
    """Return the timestamp of the most recent prediction."""
    return _last_prediction_ts


# ──────────────────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse)
async def predict(readings: List[TelemetryInput]) -> PredictionResponse:
    """Generate latency and congestion predictions.

    Requires at least 60 recent telemetry samples to build a valid
    feature window.

    Args:
        readings: List of telemetry input records.

    Returns:
        PredictionResponse with multi-horizon RTT and congestion.
    """
    global _last_prediction_ts

    if len(readings) < 60:
        raise HTTPException(
            status_code=422,
            detail="At least 60 telemetry samples are required for prediction",
        )

    if _ensemble is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()

    try:
        # Convert to DataFrame
        records = [r.model_dump() for r in readings]
        df = pd.DataFrame(records)
        df["congestion_label"] = 0  # placeholder for pipeline

        # Feature engineering
        features_df = _pipeline.transform(df)
        if features_df.empty:
            raise HTTPException(status_code=422, detail="Feature extraction produced no rows")

        # Get feature matrix
        exclude = {
            "timestamp", "rtt_ms", "packet_loss", "jitter_ms",
            "bytes_per_sec", "packets_per_sec", "congestion_label",
        }
        feature_cols = [c for c in features_df.columns if c not in exclude]
        X = features_df[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict
        result = _ensemble.predict(X)

        elapsed_ms = (time.perf_counter() - start) * 1000
        now = time.time()
        _last_prediction_ts = now

        congestion_prob = result["congestion_prob"]
        alert = congestion_prob > 0.8
        alert_msg = None
        if alert:
            alert_msg = (
                f"High congestion probability: {congestion_prob:.1%}. "
                f"Predicted RTT spike to {result['rtt_60s']:.1f}ms within 60s."
            )
            CONGESTION_ALERT_COUNTER.inc()

        # Update Prometheus
        PREDICTION_COUNTER.inc()
        LATENCY_HISTOGRAM.observe(elapsed_ms / 1000)
        RTT_GAUGE.labels(horizon="10s").set(result["rtt_10s"])
        RTT_GAUGE.labels(horizon="30s").set(result["rtt_30s"])
        RTT_GAUGE.labels(horizon="60s").set(result["rtt_60s"])
        CONGESTION_GAUGE.set(congestion_prob)

        return PredictionResponse(
            timestamp=now,
            rtt_10s=result["rtt_10s"],
            rtt_30s=result["rtt_30s"],
            rtt_60s=result["rtt_60s"],
            congestion_prob=congestion_prob,
            congestion_label=result["congestion_label"],
            alert=alert,
            alert_message=alert_msg,
            model_contributions=result["model_contributions"],
            processing_time_ms=round(elapsed_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediction failed: {}", exc)
        raise HTTPException(status_code=503, detail=f"Prediction error: {exc}")


# ──────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """System health check.

    Returns model status, Kafka connectivity, and uptime.
    """
    uptime = time.time() - request.app.state.start_time

    kafka_ok = False
    try:
        from kafka import KafkaConsumer as _KC
        consumer = _KC(
            bootstrap_servers=request.app.state.settings.KAFKA_BOOTSTRAP_SERVERS,
            consumer_timeout_ms=1000,
        )
        consumer.close()
        kafka_ok = True
    except Exception:
        kafka_ok = False

    model_ok = _ensemble is not None and len(_ensemble.models) > 0

    return HealthResponse(
        status="ok" if model_ok else "degraded",
        model_loaded=model_ok,
        kafka_connected=kafka_ok,
        last_prediction_ts=_last_prediction_ts,
        uptime_sec=round(uptime, 1),
    )


# ──────────────────────────────────────────────────────────
# GET /metrics
# ──────────────────────────────────────────────────────────

@router.get("/metrics")
async def metrics():
    """Return Prometheus metrics in text exposition format."""
    from prometheus_client import generate_latest
    from fastapi import Response
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4; charset=utf-8")


# ──────────────────────────────────────────────────────────
# POST /retrain
# ──────────────────────────────────────────────────────────

@router.post("/retrain", response_model=RetrainStatusResponse)
async def retrain(request: RetrainRequest) -> RetrainStatusResponse:
    """Queue an async retraining job.

    Args:
        request: Retrain parameters.

    Returns:
        Job ID and initial status.
    """
    job_id = str(uuid.uuid4())
    _retrain_jobs[job_id] = {
        "status": "queued",
        "days": request.days_of_history,
        "created_at": time.time(),
    }

    logger.info("Retrain job queued: {} ({}d of data)", job_id, request.days_of_history)

    return RetrainStatusResponse(
        job_id=job_id,
        status="queued",
        message=f"Retrain job queued with {request.days_of_history} days of history",
    )


@router.get("/retrain/{job_id}", response_model=RetrainStatusResponse)
async def retrain_status(job_id: str) -> RetrainStatusResponse:
    """Check the status of a retrain job.

    Args:
        job_id: UUID of the retrain job.

    Returns:
        Current job status.
    """
    if job_id not in _retrain_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _retrain_jobs[job_id]
    return RetrainStatusResponse(
        job_id=job_id,
        status=job["status"],
        message=job.get("message"),
    )
