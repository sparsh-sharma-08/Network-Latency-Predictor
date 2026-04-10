"""Training orchestrator — runs the full train → evaluate → log pipeline.

Trains all four models (ARIMA, XGBoost, LSTM, Queuing), evaluates on
the test set, logs everything to MLflow, and saves the best ensemble.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import get_settings
from features.pipeline import FeaturePipeline
from models.arima_model import ARIMAModel
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from models.queuing_model import QueuingModel
from models.ensemble import EnsembleModel
from training.dataset import LatencyDataset
from training.evaluator import Evaluator

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class Trainer:
    """Orchestrates the end-to-end training pipeline.

    Steps:
      1. Feature pipeline transform
      2. Create train / val / test splits
      3. Train all 4 models sequentially
      4. Evaluate each on the test set
      5. Log to MLflow (params, metrics, artifacts)
      6. Save ensemble to ``MODEL_DIR``
      7. Return evaluation report
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.pipeline = FeaturePipeline()
        self.evaluator = Evaluator()

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Determine feature columns (exclude targets and metadata).

        Args:
            df: Enriched DataFrame with features + targets.

        Returns:
            List of feature column names.
        """
        exclude = {
            "timestamp", "rtt_ms", "packet_loss", "jitter_ms",
            "bytes_per_sec", "packets_per_sec", "congestion_label",
            "rtt_10s_ahead", "rtt_30s_ahead", "rtt_60s_ahead",
            "congestion_ahead",
        }
        return [c for c in df.columns if c not in exclude]

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute the full training pipeline.

        Args:
            df: Raw telemetry DataFrame.

        Returns:
            Evaluation report dictionary.
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE START")
        logger.info("=" * 60)

        # 1. Feature engineering
        logger.info("Step 1/6: Feature engineering")
        enriched = self.pipeline.transform(df)

        # 2. Create targets and split
        logger.info("Step 2/6: Creating targets and splits")
        dataset = LatencyDataset()
        enriched = dataset.create_targets(enriched)
        train_df, val_df, test_df = dataset.train_val_test_split(enriched)

        feature_cols = self._get_feature_columns(enriched)
        target_cols_rtt = ["rtt_10s_ahead", "rtt_30s_ahead", "rtt_60s_ahead"]
        target_col_jitter = "jitter_ms" if "jitter_ms" in train_df.columns else None
        target_col_cong = "congestion_ahead"

        X_train = train_df[feature_cols].values.astype(np.float32)
        X_val = val_df[feature_cols].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)

        y_train_rtt = train_df[target_cols_rtt].values.astype(np.float32)
        y_val_rtt = val_df[target_cols_rtt].values.astype(np.float32)
        y_test_rtt = test_df[target_cols_rtt].values.astype(np.float32)

        y_train_cong = train_df[target_col_cong].values.astype(np.float32)
        y_val_cong = val_df[target_col_cong].values.astype(np.float32)
        y_test_cong = test_df[target_col_cong].values.astype(np.float32)

        # Replace NaN/inf in features
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Construct combined targets for XGBoost (rtt×3 + jitter + congestion)
        if target_col_jitter and target_col_jitter in train_df.columns:
            jitter_train = train_df[target_col_jitter].values.astype(np.float32)
            jitter_val = val_df[target_col_jitter].values.astype(np.float32)
        else:
            jitter_train = np.zeros(len(train_df), dtype=np.float32)
            jitter_val = np.zeros(len(val_df), dtype=np.float32)

        y_xgb_train = np.column_stack([y_train_rtt, jitter_train, y_train_cong])
        y_xgb_val = np.column_stack([y_val_rtt, jitter_val, y_val_cong])

        # LSTM targets (rtt×3 + congestion)
        y_lstm_train = np.column_stack([y_train_rtt, y_train_cong])
        y_lstm_val = np.column_stack([y_val_rtt, y_val_cong])

        # MLflow setup
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(self.settings.MLFLOW_TRACKING_URI)
                mlflow.set_experiment(self.settings.MLFLOW_EXPERIMENT_NAME)
            except Exception as exc:
                logger.warning("MLflow setup failed (will continue without): {}", exc)

        reports: Dict[str, Any] = {}
        ensemble = EnsembleModel()

        mlflow_run = None
        if MLFLOW_AVAILABLE:
            try:
                mlflow_run = mlflow.start_run()
            except Exception:
                mlflow_run = None

        try:
            # 3. Train models
            logger.info("Step 3/6: Training models")

            # ── ARIMA ──
            logger.info("Training ARIMA...")
            arima = ARIMAModel()
            try:
                arima_series = train_df["rtt_ms"].values[-2000:]  # Use last 2000 for speed
                arima.fit(None, arima_series)
                ensemble.add_model(arima)
                logger.info("ARIMA trained successfully")
            except Exception as exc:
                logger.error("ARIMA training failed: {}", exc)

            # ── XGBoost ──
            logger.info("Training XGBoost...")
            xgb_model = XGBoostModel()
            try:
                xgb_model.fit(
                    X_train, y_xgb_train,
                    eval_set=[(X_val, y_xgb_val)],
                    feature_names=feature_cols,
                )
                ensemble.add_model(xgb_model)
                logger.info("XGBoost trained successfully")
            except Exception as exc:
                logger.error("XGBoost training failed: {}", exc)

            # ── LSTM ──
            logger.info("Training LSTM...")
            lstm_model = LSTMModel(n_features=X_train.shape[1])
            try:
                lstm_model.fit(
                    X_train, y_lstm_train,
                    X_val=X_val, y_val=y_lstm_val,
                )
                ensemble.add_model(lstm_model)
                logger.info("LSTM trained successfully")
            except Exception as exc:
                logger.error("LSTM training failed: {}", exc)

            # ── Queuing ──
            logger.info("Training Queuing Model...")
            queuing = QueuingModel()
            try:
                # Use packets_per_sec and jitter as features
                pps_idx = feature_cols.index("packets_per_sec") if "packets_per_sec" in feature_cols else None
                jit_idx = feature_cols.index("jitter_ms") if "jitter_ms" in feature_cols else None

                if pps_idx is not None and jit_idx is not None:
                    q_train = X_train[:, [pps_idx, jit_idx]]
                elif pps_idx is not None:
                    q_train = X_train[:, [pps_idx]]
                else:
                    q_train = X_train[:, :2]

                queuing.fit(q_train, train_df["rtt_ms"].values)
                ensemble.add_model(queuing)
                logger.info("Queuing model trained successfully")
            except Exception as exc:
                logger.error("Queuing model training failed: {}", exc)

            # 4. Evaluate
            logger.info("Step 4/6: Evaluating models")
            for model_name, model in ensemble.models.items():
                try:
                    if model_name == "arima":
                        pred = model.predict(X_test)
                        if pred.shape[0] == 1:
                            pred = np.broadcast_to(pred, (len(X_test), 3)).copy()
                        cong_pred = model.predict_proba_congestion(X_test)
                        if cong_pred.shape[0] == 1:
                            cong_pred = np.broadcast_to(cong_pred, (len(X_test),)).copy()
                    elif model_name == "queuing":
                        if pps_idx is not None and jit_idx is not None:
                            q_test = X_test[:, [pps_idx, jit_idx]]
                        elif pps_idx is not None:
                            q_test = X_test[:, [pps_idx]]
                        else:
                            q_test = X_test[:, :2]
                        pred = model.predict(q_test)
                        cong_pred = model.predict_proba_congestion(q_test)
                    else:
                        pred = model.predict(X_test)
                        cong_pred = model.predict_proba_congestion(X_test)

                    # Ensure shapes match
                    pred = pred[:len(y_test_rtt)]
                    cong_pred = cong_pred[:len(y_test_cong)]

                    report = self.evaluator.evaluate_all(
                        y_test_rtt[:len(pred)], pred,
                        y_test_cong[:len(cong_pred)], cong_pred,
                    )
                    reports[model_name] = report

                    # Log to MLflow
                    if MLFLOW_AVAILABLE and mlflow_run:
                        try:
                            for hz, metrics in report["regression"].items():
                                for metric, value in metrics.items():
                                    mlflow.log_metric(f"{model_name}_{hz}_{metric}", value)
                            for metric, value in report["congestion"].items():
                                mlflow.log_metric(f"{model_name}_congestion_{metric}", value)
                        except Exception:
                            pass

                except Exception as exc:
                    logger.error("Evaluation of '{}' failed: {}", model_name, exc)

            # 5. Ensemble evaluation
            logger.info("Step 5/6: Ensemble evaluation")
            ensemble_pred = ensemble.predict(X_test)

            # 6. Save ensemble
            logger.info("Step 6/6: Saving models")
            model_dir = self.settings.MODEL_DIR
            ensemble.save(model_dir)

            if MLFLOW_AVAILABLE and mlflow_run:
                try:
                    mlflow.log_param("n_train", len(train_df))
                    mlflow.log_param("n_val", len(val_df))
                    mlflow.log_param("n_test", len(test_df))
                    mlflow.log_param("n_features", len(feature_cols))
                    mlflow.log_param("models_trained", list(ensemble.models.keys()))
                except Exception:
                    pass

        finally:
            if MLFLOW_AVAILABLE and mlflow_run:
                try:
                    mlflow.end_run()
                except Exception:
                    pass

        elapsed = time.time() - start_time
        logger.info("TRAINING COMPLETE in {:.1f}s", elapsed)
        logger.info("Model reports: {}", reports)

        return {
            "models": reports,
            "ensemble_prediction_sample": ensemble_pred,
            "training_time_s": round(elapsed, 2),
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
        }
