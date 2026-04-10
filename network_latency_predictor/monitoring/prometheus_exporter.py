"""Prometheus metric definitions for the Network Latency Predictor.

Exposes counters, gauges, and histograms for predictions, congestion
alerts, model performance, and drift scores.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Counters ───────────────────────────────────────────────────
PREDICTION_COUNTER = Counter(
    "nlp_predictions_total",
    "Total number of predictions served",
)

CONGESTION_ALERT_COUNTER = Counter(
    "nlp_congestion_alerts_total",
    "Total number of congestion alerts fired",
)

# ── Histograms ─────────────────────────────────────────────────
LATENCY_HISTOGRAM = Histogram(
    "nlp_prediction_latency_seconds",
    "Time taken to serve a prediction request",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# ── Gauges ─────────────────────────────────────────────────────
RTT_GAUGE = Gauge(
    "nlp_predicted_rtt_ms",
    "Most recent predicted RTT in milliseconds",
    ["horizon"],
)

CONGESTION_GAUGE = Gauge(
    "nlp_congestion_probability",
    "Most recent congestion probability prediction",
)

MODEL_MAE_GAUGE = Gauge(
    "nlp_model_mae_ms",
    "Rolling MAE for each model in the ensemble",
    ["model_name"],
)

DRIFT_GAUGE = Gauge(
    "nlp_drift_score",
    "Current drift score from the Page-Hinkley test",
)
