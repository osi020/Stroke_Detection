"""
config.py
---------
Single source of truth for all project paths, constants, and
hyperparameters. Import this module everywhere to avoid hard-coded paths.
"""

from pathlib import Path

# ── Project root (two levels up from this file: src/stroke_prediction/config.py)
ROOT = Path(__file__).resolve().parents[2]

# ── Data
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# ── Model artefacts & reports
MODELS_DIR  = ROOT / "models" / "artifacts"
REPORTS_DIR = ROOT / "models" / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ── Notebooks
NOTEBOOKS_DIR = ROOT / "notebooks"

# ── Ensure output directories exist at import time
for _d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Experiment settings
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# ── Decision thresholds (F2-score optimised)
# Sensitive  → calibrated LogReg, recall 72 %, precision 20 %
# Standard   → LogReg_Balanced, recall 68 %, precision 22 %
THRESHOLD_SENSITIVE = 0.14
THRESHOLD_STANDARD  = 0.72

# ── Clinical glucose reference ranges (mg/dL)
GLUCOSE_NORMAL     = (70,  99)
GLUCOSE_PREDIABETIC= (100, 125)
GLUCOSE_DIABETIC   = 126

# ── BMI categories (kg/m²)
BMI_UNDERWEIGHT = 18.5
BMI_NORMAL_MAX  = 24.9
BMI_OVERWEIGHT  = 25.0
BMI_OBESE       = 30.0
BMI_OBESE2      = 35.0
