"""Pydantic request / response models for the API layer.

All schemas use strict validation and are documented with field
descriptions for the auto-generated OpenAPI spec.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TelemetryInput(BaseModel):
    """A single network telemetry measurement."""

    timestamp: float = Field(..., description="Unix epoch timestamp (UTC)")
    rtt_ms: float = Field(..., description="Round-trip time in milliseconds")
    packet_loss: bool = Field(..., description="Whether the packet was lost")
    jitter_ms: float = Field(..., description="Jitter in milliseconds")
    bytes_per_sec: float = Field(..., description="Throughput in bytes/second")
    packets_per_sec: float = Field(..., description="Packet rate per second")


class PredictionResponse(BaseModel):
    """Prediction output from the ensemble model."""

    timestamp: float = Field(..., description="Prediction timestamp (UTC)")
    rtt_10s: float = Field(..., description="Predicted RTT at t+10 s (ms)")
    rtt_30s: float = Field(..., description="Predicted RTT at t+30 s (ms)")
    rtt_60s: float = Field(..., description="Predicted RTT at t+60 s (ms)")
    congestion_prob: float = Field(
        ..., ge=0.0, le=1.0, description="Congestion probability"
    )
    congestion_label: bool = Field(
        ..., description="True if congestion_prob > 0.7"
    )
    alert: bool = Field(..., description="True if congestion_prob > 0.8")
    alert_message: Optional[str] = Field(
        None, description="Human-readable alert message"
    )
    model_contributions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-model weighted contributions",
    )
    processing_time_ms: float = Field(
        ..., description="End-to-end processing time in ms"
    )


class HealthResponse(BaseModel):
    """System health status report."""

    status: str = Field(..., description="'ok' or 'degraded'")
    model_loaded: bool = Field(..., description="Whether the ensemble model is loaded")
    kafka_connected: bool = Field(
        ..., description="Whether Kafka connection is active"
    )
    last_prediction_ts: Optional[float] = Field(
        None, description="Timestamp of last prediction"
    )
    uptime_sec: float = Field(..., description="Seconds since API start-up")


class RetrainRequest(BaseModel):
    """Request to trigger model retraining."""

    days_of_history: int = Field(
        default=7, ge=1, le=365, description="Days of history to retrain on"
    )


class RetrainStatusResponse(BaseModel):
    """Status of an async retrain job."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="queued | running | completed | failed")
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """RFC 7807 problem detail format for errors."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: Optional[str] = None
