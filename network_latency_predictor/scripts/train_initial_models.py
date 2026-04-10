#!/usr/bin/env python3
"""Train all models on mock data and save the ensemble.

Usage:
    python scripts/train_initial_models.py --data data/mock_telemetry.csv --model-dir models/saved/
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from loguru import logger

from config.logging_config import setup_logging
from training.trainer import Trainer


def main() -> None:
    """CLI entry point for initial model training."""
    parser = argparse.ArgumentParser(
        description="Train all prediction models on telemetry data"
    )
    parser.add_argument(
        "--data", type=str, default="data/mock_telemetry.csv",
        help="Path to CSV data file (default: data/mock_telemetry.csv)",
    )
    parser.add_argument(
        "--model-dir", type=str, default="models/saved/",
        help="Directory to save trained models (default: models/saved/)",
    )
    parser.add_argument(
        "--experiment", type=str, default="network_latency_prediction",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    setup_logging(level="INFO", json_output=False)

    if not os.path.exists(args.data):
        logger.error("Data file not found: {}", args.data)
        print(f"\nERROR: Data file not found: {args.data}")
        print("Run 'python scripts/generate_mock_data.py' first.")
        sys.exit(1)

    # Override model dir
    os.environ["MODEL_DIR"] = args.model_dir
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment

    logger.info("Loading data from {}", args.data)
    df = pd.read_csv(args.data)
    logger.info("Loaded {} rows", len(df))

    # Use a subset for faster training if data is very large
    max_rows = 200_000  # ~2.3 days at 1s interval
    if len(df) > max_rows:
        logger.info("Subsampling to {} rows for training speed", max_rows)
        df = df.tail(max_rows).reset_index(drop=True)

    trainer = Trainer()
    report = trainer.run(df)

    # Print results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"  Training time:      {report['training_time_s']:.1f}s")
    print(f"  Features used:      {report['n_features']}")
    print()

    max_mae = 0.0
    for model_name, metrics in report["models"].items():
        print(f"  📊 {model_name.upper()}")
        if "regression" in metrics:
            for hz, vals in metrics["regression"].items():
                mae = vals["mae"]
                max_mae = max(max_mae, mae)
                print(f"    {hz}: MAE={vals['mae']:.2f}ms  RMSE={vals['rmse']:.2f}ms  R²={vals['r2']:.3f}")
        if "congestion" in metrics:
            c = metrics["congestion"]
            print(f"    Congestion: F1={c['f1']:.3f}  Prec={c['precision']:.3f}  Rec={c['recall']:.3f}")
        print()

    print("  Ensemble sample prediction:")
    sample = report.get("ensemble_prediction_sample", {})
    if isinstance(sample, dict):
        for k, v in sample.items():
            if k != "model_contributions":
                print(f"    {k}: {v}")
    print("=" * 60)

    # Sanity check
    if max_mae > 20.0:
        logger.error("SANITY CHECK FAILED — max MAE ({:.2f}ms) exceeds 20ms", max_mae)
        print(f"\n❌ SANITY CHECK FAILED: max MAE = {max_mae:.2f}ms > 20ms threshold")
        sys.exit(1)
    else:
        print(f"\n✅ All models passed sanity check (max MAE = {max_mae:.2f}ms)")


if __name__ == "__main__":
    main()
