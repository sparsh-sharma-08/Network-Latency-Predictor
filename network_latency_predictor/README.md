# ⚡️ Network Latency Predictor

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?style=flat&logo=Docker&logoColor=white)](https://www.docker.com/)

A production-ready system for predicting network latency (RTT) and congestion probability using an ensemble of ARIMA, XGBoost, LSTM, and queuing-theory models natively pulling real telemetry data.

## 🏗 Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Collector   │────▶│    Kafka      │────▶│   Feature    │────▶│  Ensemble    │
│  (ICMP/Sniff) │     │  (Streaming)  │     │  Pipeline    │     │   Model      │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                     ┌──────────────┐     ┌──────────────┐           │
                     │  Prometheus   │◀────│  FastAPI      │◀──────────┘
                     │  + Grafana    │     │  REST API     │
                     └──────────────┘     └──────────────┘
                                                │
                     ┌──────────────┐           │
                     │   MLflow      │◀──────────┘
                     │  (Tracking)   │
                     └──────────────┘
```

**Models used:**
- **ARIMA**: Univariate time-series baseline (statsmodels)
- **XGBoost**: Gradient-boosted trees with ~45 engineered features
- **LSTM**: PyTorch LSTM with self-attention and dual output heads
- **M/M/1 / M/D/1**: Analytical queuing-theory models
- **Ensemble**: Weighted combination with dynamic weight adjustment

**Prediction targets:**
- RTT at t+10s, t+30s, t+60s (milliseconds)
- Congestion probability (0.0–1.0)

## Quickstart

### 1. Install dependencies

```bash
cd network_latency_predictor
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env as needed
```

### 3. Generate mock data

```bash
python scripts/generate_mock_data.py --days 30 --seed 42
```

### 4. Train models

**Option A: Train locally**
```bash
python scripts/train_initial_models.py --data data/mock_telemetry.csv
```

**Option B: Train on Google Colab (recommended for machines without GPU)**

1. Open `notebooks/train_on_colab.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells — the notebook generates data, trains all 4 models, and downloads a ZIP
4. Import the trained models locally:

```bash
python scripts/sync_models.py --import-zip trained_models_YYYYMMDD_HHMMSS.zip
```

> **Tip**: Mount Google Drive in the notebook to persist models across sessions.

**Sync helpers:**
```bash
# Export local models for upload to Colab
python scripts/sync_models.py --export

# Verify model integrity after import
python scripts/sync_models.py --verify
```

### 5. Start services (Docker)

```bash
docker-compose -f docker/docker-compose.yml up -d
```

### 6. Verify

```bash
curl http://localhost:8000/health
```

## API Reference

### `GET /health`

Returns system health status.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "kafka_connected": true,
  "last_prediction_ts": 1700000000.0,
  "uptime_sec": 120.5
}
```

### `POST /predict`

Generate latency and congestion predictions. Requires at least 60 telemetry samples.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[
    {"timestamp": 1700000000, "rtt_ms": 22.5, "packet_loss": false, "jitter_ms": 1.2, "bytes_per_sec": 1000000, "packets_per_sec": 1000},
    ... (60+ records)
  ]'
```

Response:
```json
{
  "timestamp": 1700000060.0,
  "rtt_10s": 23.1,
  "rtt_30s": 24.5,
  "rtt_60s": 26.2,
  "congestion_prob": 0.15,
  "congestion_label": false,
  "alert": false,
  "alert_message": null,
  "model_contributions": {
    "arima": {"weight": 0.2, "rtt_10s": 22.8},
    "xgboost": {"weight": 0.4, "rtt_10s": 23.5},
    "lstm": {"weight": 0.3, "rtt_10s": 22.9},
    "queuing": {"weight": 0.1, "rtt_10s": 23.2}
  },
  "processing_time_ms": 45.2
}
```

### `GET /metrics`

Returns Prometheus metrics in text exposition format.

```bash
curl http://localhost:8000/metrics
```

### `POST /retrain`

Trigger asynchronous model retraining.

```bash
curl -X POST http://localhost:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{"days_of_history": 7}'
```

### `GET /retrain/{job_id}`

Check retraining job status.

```bash
curl http://localhost:8000/retrain/<job_id>
```

## Adding a New Model

1. Create a new file in `models/` (e.g., `models/transformer_model.py`)
2. Inherit from `BaseLatencyModel`:

```python
from models.base_model import BaseLatencyModel

class TransformerModel(BaseLatencyModel):
    @property
    def name(self) -> str:
        return "transformer"
    
    def fit(self, X, y):
        ...
    
    def predict(self, X) -> np.ndarray:
        # Return shape (n_samples, 3) for t+10/30/60s
        ...
    
    def predict_proba_congestion(self, X) -> np.ndarray:
        # Return shape (n_samples,) with probs in [0, 1]
        ...
```

3. Register in the ensemble at `models/ensemble.py`
4. Add weight in `config/settings.py` → `ENSEMBLE_WEIGHTS`

## 📊 Grafana Dashboard Setup

1. Open Grafana at `http://localhost:3000` (Login with `admin` and the password defined in your `.env` under `GRAFANA_ADMIN_PASSWORD`, defaults to `admin`).
2. The Prometheus data source (`http://prometheus:9090`) and internal dashboards are **automatically provisioned** on startup!
3. To customize, explore panels with these native queries:
   - `nlp_predictions_total` — total predictions served
   - `nlp_predicted_rtt_ms{horizon="10s"}` — current predicted RTT
   - `nlp_congestion_probability` — congestion probability gauge
   - `nlp_prediction_latency_seconds` — API latency histogram
   - `nlp_model_mae_ms{model_name="xgboost"}` — per-model MAE
   - `nlp_drift_score` — model drift indicator

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NETWORK_INTERFACE` | `eth0` | Network interface for packet capture |
| `PROBE_HOSTS` | `["8.8.8.8","1.1.1.1","208.67.222.222"]` | ICMP probe targets |
| `PROBE_INTERVAL_SEC` | `1.0` | Seconds between probes |
| `CAPTURE_FILTER` | `icmp or tcp` | BPF capture filter |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address |
| `KAFKA_TOPIC_RAW` | `network.telemetry.raw` | Raw telemetry topic |
| `KAFKA_TOPIC_FEATURES` | `network.telemetry.features` | Feature topic |
| `KAFKA_TOPIC_PREDICTIONS` | `network.predictions` | Predictions topic |
| `WINDOW_SHORT` | `5` | Short rolling window (seconds) |
| `WINDOW_MEDIUM` | `30` | Medium rolling window (seconds) |
| `WINDOW_LONG` | `300` | Long rolling window (seconds) |
| `EWMA_ALPHA` | `0.3` | EWMA smoothing factor |
| `MODEL_DIR` | `./models/saved` | Saved model directory |
| `ENSEMBLE_WEIGHTS` | `{"arima":0.2,...}` | Model weights (JSON) |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8000` | API bind port |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URI |
| `MLFLOW_EXPERIMENT_NAME` | `network_latency_prediction` | Experiment name |
| `DRIFT_THRESHOLD_MAE` | `5.0` | MAE threshold for drift (ms) |
| `DRIFT_WINDOW_SIZE` | `500` | Samples for drift evaluation |

## Running Tests

```bash
pytest tests/ -v --tb=short
```

## License

MIT
