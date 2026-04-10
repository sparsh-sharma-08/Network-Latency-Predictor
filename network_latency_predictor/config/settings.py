"""Application settings managed via pydantic-settings.

All configuration is loaded from environment variables or a `.env` file.
No value should ever be hard-coded in application code — reference
``get_settings()`` instead.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for every subsystem.

    Attributes are grouped by subsystem and documented inline.
    """

    # ── Collector ───────────────────────────────────────────────
    NETWORK_INTERFACE: str = "eth0"
    PROBE_HOSTS: List[str] = ["8.8.8.8", "1.1.1.1", "208.67.222.222"]
    PROBE_INTERVAL_SEC: float = 1.0
    CAPTURE_FILTER: str = "icmp or tcp"

    # ── Kafka ───────────────────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_RAW: str = "network.telemetry.raw"
    KAFKA_TOPIC_FEATURES: str = "network.telemetry.features"
    KAFKA_TOPIC_PREDICTIONS: str = "network.predictions"

    # ── Feature windows (seconds) ──────────────────────────────
    WINDOW_SHORT: int = 5
    WINDOW_MEDIUM: int = 30
    WINDOW_LONG: int = 300

    # ── EWMA ────────────────────────────────────────────────────
    EWMA_ALPHA: float = 0.3

    # ── Model ───────────────────────────────────────────────────
    MODEL_DIR: str = "./models/saved"
    ENSEMBLE_WEIGHTS: Dict[str, float] = {
        "arima": 0.2,
        "xgboost": 0.4,
        "lstm": 0.3,
        "queuing": 0.1,
    }

    # ── API ─────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ── MLflow ──────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "network_latency_prediction"

    # ── Drift detection ────────────────────────────────────────
    DRIFT_THRESHOLD_MAE: float = 5.0  # ms — retrain if MAE exceeds this
    DRIFT_WINDOW_SIZE: int = 500  # samples to evaluate drift on

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()
