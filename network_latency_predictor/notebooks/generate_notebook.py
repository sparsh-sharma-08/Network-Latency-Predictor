#!/usr/bin/env python3
"""Generate the Colab training notebook programmatically."""
import json, os, sys

def md(source): return {"cell_type":"markdown","metadata":{},"source":source.split("\n")}
def code(source): return {"cell_type":"code","metadata":{},"source":source.split("\n"),"outputs":[],"execution_count":None}

# Read all source files
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()

cells = []

# ── Cell 1: Title ──
cells.append(md("""# 🚀 Network Latency Predictor — Cloud Training (Google Colab)

This notebook runs the **full training pipeline** on Colab's GPU/CPU resources.

**Models trained:** ARIMA, XGBoost (×5 sub-models), LSTM (PyTorch + Attention), Queuing Theory  
**Output:** Trained model artifacts saved to Google Drive or downloaded as ZIP

> ⚡ **Runtime → Change runtime type → T4 GPU** for fastest LSTM training
"""))

# ── Cell 2: Install deps & verify ──
# No version pins → no numpy binary conflict → no restart needed.
cells.append(code("""# @title 1️⃣ Install & Verify Dependencies
# Colab already has numpy, pandas, scipy, scikit-learn, torch.
# Only install the extras this project needs (NO version pins).
!pip install -q statsmodels xgboost loguru pydantic-settings python-dotenv joblib

import numpy as np, pandas as pd, scipy, sklearn, torch
print(f"NumPy:        {np.__version__}")
print(f"Pandas:       {pd.__version__}")
print(f"SciPy:        {scipy.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA:         {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:          {torch.cuda.get_device_name(0)}")

import statsmodels, xgboost
print(f"Statsmodels:  {statsmodels.__version__}")
print(f"XGBoost:      {xgboost.__version__}")
print("\\n✅ All dependencies ready")
"""))

# ── Cell 3: Mount Drive (optional) ──
cells.append(code("""# @title 2️⃣ Mount Google Drive (Optional — for persistent model storage)
MOUNT_DRIVE = True  # @param {type:"boolean"}

if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DIR = '/content/drive/MyDrive/netPredictor_models'
    import os; os.makedirs(DRIVE_DIR, exist_ok=True)
    print(f"✅ Drive mounted. Models will be saved to: {DRIVE_DIR}")
else:
    DRIVE_DIR = None
    print("⏭️ Drive not mounted. Models will be downloaded as ZIP.")
"""))

# ── Cell 4: Create project structure ──
cells.append(code("""# @title 3️⃣ Create Project Structure on Colab
import os
BASE = '/content/netPredictor'
for d in ['config','collector','features','models','training','data','models/saved','scripts','logs']:
    os.makedirs(f'{BASE}/{d}', exist_ok=True)
    if d not in ['data','models/saved','logs']:
        open(f'{BASE}/{d}/__init__.py','w').write('')

os.chdir(BASE)
import sys; sys.path.insert(0, BASE)
print(f"✅ Project structure created at {BASE}")
"""))

# ── Cell 5-N: Write all source modules ──
modules = [
    ("config/settings.py", "config/settings.py"),
    ("config/logging_config.py", "config/logging_config.py"),
    ("features/rolling_stats.py", "features/rolling_stats.py"),
    ("features/ewma.py", "features/ewma.py"),
    ("features/fft_features.py", "features/fft_features.py"),
    ("features/gradient_estimator.py", "features/gradient_estimator.py"),
    ("features/pipeline.py", "features/pipeline.py"),
    ("models/base_model.py", "models/base_model.py"),
    ("models/arima_model.py", "models/arima_model.py"),
    ("models/xgboost_model.py", "models/xgboost_model.py"),
    ("models/lstm_model.py", "models/lstm_model.py"),
    ("models/queuing_model.py", "models/queuing_model.py"),
    ("models/ensemble.py", "models/ensemble.py"),
    ("collector/mock_generator.py", "collector/mock_generator.py"),
    ("training/dataset.py", "training/dataset.py"),
    ("training/evaluator.py", "training/evaluator.py"),
    ("training/trainer.py", "training/trainer.py"),
]

write_cell_src = '# @title 4️⃣ Write All Source Modules\nimport os\nBASE = "/content/netPredictor"\nmodules = {\n'
for rel, _ in modules:
    content = read(rel)
    escaped = repr(content)
    write_cell_src += f'    "{rel}": {escaped},\n'
write_cell_src += """}\n
for path, content in modules.items():
    full = os.path.join(BASE, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w') as f:
        f.write(content)
    print(f"  ✅ {path}")

print(f"\\n✅ All {len(modules)} source modules written")
"""
cells.append(code(write_cell_src))

# ── Cell: Generate mock data ──
cells.append(code("""# @title 5️⃣ Generate Mock Training Data
import os, gc; os.chdir('/content/netPredictor')
import sys; sys.path.insert(0, '/content/netPredictor')

# ⚠️ Colab free tier has ~12 GB RAM. 7 days ≈ 605K rows is safe.
# Increase to 14-30 only on Colab Pro (25+ GB RAM).
DAYS = 7  # @param {type:"integer"}
SEED = 42  # @param {type:"integer"}

from collector.mock_generator import generate_mock_telemetry
df = generate_mock_telemetry(n_days=DAYS, interval_sec=1.0, seed=SEED)
df.to_csv('data/mock_telemetry.csv', index=False)

print(f"\\n📊 Generated {len(df):,} rows ({DAYS} days)")
print(f"   RTT: mean={df['rtt_ms'].mean():.1f}ms, std={df['rtt_ms'].std():.1f}ms")
print(f"   Congestion rate: {df['congestion_label'].mean():.1%}")
print(f"   File: data/mock_telemetry.csv ({os.path.getsize('data/mock_telemetry.csv')/1e6:.1f} MB)")

del df; gc.collect()
print("🧹 Memory freed")
"""))

# ── Cell: Train ──
cells.append(code("""# @title 6️⃣ Train All Models (ARIMA + XGBoost + LSTM + Queuing)
import os, gc, time
os.chdir('/content/netPredictor')
os.environ['MODEL_DIR'] = 'models/saved'

import pandas as pd
from config.logging_config import setup_logging
from training.trainer import Trainer

setup_logging(level="INFO", json_output=False)
df = pd.read_csv('data/mock_telemetry.csv')

# ⚠️ Colab RAM limit! 80K rows is safe for free tier (~12 GB).
# The LSTM creates sliding-window arrays of shape (N, 60, ~45) which
# use ~N×60×45×4 bytes. At 80K rows this is ~860 MB (manageable).
# Increase to 150K-200K only on Colab Pro.
MAX_ROWS = 80_000  # @param {type:"integer"}
if len(df) > MAX_ROWS:
    print(f"Subsampling {len(df):,} → {MAX_ROWS:,} rows")
    df = df.tail(MAX_ROWS).reset_index(drop=True)

gc.collect()

print(f"\\n🏋️ Training on {len(df):,} rows...")
print(f"   GPU: {('Yes — ' + __import__('torch').cuda.get_device_name(0)) if __import__('torch').cuda.is_available() else 'No (CPU mode)'}")
print()

trainer = Trainer()
start = time.time()
report = trainer.run(df)
elapsed = time.time() - start

del df; gc.collect()
print(f"\\n✅ Training complete in {elapsed:.1f}s")
"""))

# ── Cell: Results ──
cells.append(code("""# @title 7️⃣ Training Results & Evaluation
print("=" * 65)
print("  MODEL EVALUATION RESULTS")
print("=" * 65)

max_mae = 0.0
for model_name, metrics in report["models"].items():
    print(f"\\n  📊 {model_name.upper()}")
    if "regression" in metrics:
        print(f"  {'Horizon':<10} {'MAE (ms)':<12} {'RMSE (ms)':<12} {'R²':<10}")
        print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*10}")
        for hz, vals in metrics["regression"].items():
            mae = vals['mae']
            max_mae = max(max_mae, mae)
            print(f"  {hz:<10} {mae:<12.2f} {vals['rmse']:<12.2f} {vals['r2']:<10.3f}")
    if "congestion" in metrics:
        c = metrics["congestion"]
        print(f"  Congestion: F1={c['f1']:.3f}  Precision={c['precision']:.3f}  Recall={c['recall']:.3f}")

print(f"\\n{'=' * 65}")
if max_mae > 20.0:
    print(f"  ❌ SANITY CHECK FAILED: max MAE = {max_mae:.2f}ms > 20ms")
else:
    print(f"  ✅ All models passed (max MAE = {max_mae:.2f}ms)")

print(f"  ⏱️  Total training time: {report['training_time_s']:.1f}s")
print(f"  📐 Features used: {report['n_features']}")
print("=" * 65)
"""))

# ── Cell: Download ──
cells.append(code("""# @title 8️⃣ Save & Download Trained Models
import shutil, os
from datetime import datetime

os.chdir('/content/netPredictor')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
zip_name = f'trained_models_{timestamp}'

# Create zip
shutil.make_archive(zip_name, 'zip', 'models/saved')
zip_path = f'{zip_name}.zip'
print(f"📦 Models packaged: {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)")

# Save to Drive if mounted
try:
    if DRIVE_DIR:
        drive_zip = os.path.join(DRIVE_DIR, os.path.basename(zip_path))
        shutil.copy2(zip_path, drive_zip)
        # Also copy individual model files
        drive_models = os.path.join(DRIVE_DIR, 'latest')
        if os.path.exists(drive_models):
            shutil.rmtree(drive_models)
        shutil.copytree('models/saved', drive_models)
        print(f"💾 Saved to Google Drive: {DRIVE_DIR}")
        print(f"   ZIP: {drive_zip}")
        print(f"   Models: {drive_models}/")
except:
    pass

# Download to local machine
from google.colab import files
files.download(zip_path)
print("\\n⬇️  Downloading to your local machine...")
print("\\n📋 To use locally, run:")
print(f"   python scripts/sync_models.py --import {zip_name}.zip")
"""))

# ── Build notebook ──
nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4", "name": "Network Latency Predictor — Cloud Training"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": cells
}

out_path = os.path.join(ROOT, "notebooks", "train_on_colab.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Notebook generated: {out_path}")
print(f"   Cells: {len(cells)}")
