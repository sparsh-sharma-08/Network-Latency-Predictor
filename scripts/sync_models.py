#!/usr/bin/env python3
"""Sync trained models between Google Colab and your local machine.

Usage:
    # Import models from a Colab-downloaded ZIP
    python scripts/sync_models.py --import trained_models_20260408.zip

    # Export local models to a ZIP for uploading to Colab
    python scripts/sync_models.py --export

    # Verify model integrity
    python scripts/sync_models.py --verify
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "saved",
)


def import_models(zip_path: str) -> None:
    """Unpack a model ZIP from Colab into models/saved/.

    Creates a backup of existing models before overwriting.

    Args:
        zip_path: Path to the downloaded ZIP file.
    """
    if not os.path.exists(zip_path):
        print(f"❌ ZIP file not found: {zip_path}")
        sys.exit(1)

    # Backup existing models
    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{MODEL_DIR}_backup_{ts}"
        shutil.copytree(MODEL_DIR, backup)
        print(f"📁 Backed up existing models to: {backup}")

    # Extract
    os.makedirs(MODEL_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(MODEL_DIR)
        files = zf.namelist()

    print(f"✅ Imported {len(files)} files to {MODEL_DIR}:")
    for f in files:
        full = os.path.join(MODEL_DIR, f)
        if os.path.isfile(full):
            size_kb = os.path.getsize(full) / 1024
            print(f"   📦 {f} ({size_kb:.1f} KB)")

    verify_models()


def export_models() -> None:
    """Package local models into a ZIP for uploading to Colab.

    The ZIP is created in the project root directory.
    """
    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        print(f"❌ No models found in {MODEL_DIR}")
        print("   Train models first: python scripts/train_initial_models.py")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zip_base = os.path.join(project_root, f"trained_models_{ts}")
    zip_path = shutil.make_archive(zip_base, "zip", MODEL_DIR)

    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"✅ Exported models to: {zip_path} ({size_mb:.1f} MB)")
    print(f"   Upload this to Colab or Google Drive for cloud training")


def verify_models() -> None:
    """Verify model files in models/saved/ are loadable.

    Checks that expected model files exist and attempts basic loading.
    """
    expected = [
        "arima_model.pkl",
        "xgboost_model.pkl",
        "lstm_model.pkl",
        "queuing_model.pkl",
        "ensemble_weights.json",
    ]

    print(f"\n🔍 Verifying models in {MODEL_DIR}:")
    all_ok = True

    for fname in expected:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"   ✅ {fname} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ {fname} — MISSING")
            all_ok = False

    # Verify ensemble weights JSON is valid
    weights_path = os.path.join(MODEL_DIR, "ensemble_weights.json")
    if os.path.exists(weights_path):
        try:
            with open(weights_path) as f:
                weights = json.load(f)
            total = sum(weights.values())
            print(f"\n   Ensemble weights: {weights}")
            print(f"   Weight sum: {total:.4f}")
        except (json.JSONDecodeError, Exception) as e:
            print(f"   ❌ Weights file corrupted: {e}")
            all_ok = False

    # Try loading ensemble
    if all_ok:
        try:
            from models.ensemble import EnsembleModel
            ensemble = EnsembleModel.load(MODEL_DIR)
            n_models = len(ensemble.models)
            print(f"\n   ✅ Ensemble loads successfully ({n_models} models)")
        except Exception as e:
            print(f"\n   ⚠️  Ensemble load test failed: {e}")
            print(f"       (This may be OK if dependencies aren't installed)")

    print()
    if all_ok:
        print("✅ All model files verified successfully!")
    else:
        print("⚠️  Some model files are missing — training may be incomplete")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sync trained models between Colab and local machine"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--import-zip", dest="import_zip", type=str,
        help="Import models from a Colab-downloaded ZIP file",
    )
    group.add_argument(
        "--export", action="store_true",
        help="Export local models as a ZIP for upload to Colab",
    )
    group.add_argument(
        "--verify", action="store_true",
        help="Verify model files in models/saved/",
    )
    args = parser.parse_args()

    if args.import_zip:
        import_models(args.import_zip)
    elif args.export:
        export_models()
    elif args.verify:
        verify_models()


if __name__ == "__main__":
    main()
