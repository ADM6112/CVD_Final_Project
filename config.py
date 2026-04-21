"""
config.py — Central Path Configuration
========================================
Edit BASE to match your local machine, then leave everything else alone.
All step files import from here — you never need to touch their paths.

Windows example : BASE = r"E:\CVD data"
Mac/Linux example: BASE = "/home/yourname/cvd_data"
"""

import os

# =========================================================
# *** EDIT THIS ONE LINE TO MATCH YOUR MACHINE ***
# =========================================================
BASE = r"E:\CVD data"

# =========================================================
# DERIVED PATHS — do not edit below this line
# =========================================================

# --- Data directories ---
TICK_DIR        = os.path.join(BASE, "Data", "Tick")
RAW_CSV_DIR     = os.path.join(TICK_DIR, "raw_csv")
SPLIT_DIR       = os.path.join(TICK_DIR, "split_by_stock")
FEATURES_FILE   = os.path.join(TICK_DIR, "clean_features.csv")

# --- Source .gz file folders (never deleted) ---
GZ_BASE_DIR     = os.path.join(TICK_DIR, "Tick GZ primary data files")
GZ_TRAIN_DIR    = os.path.join(GZ_BASE_DIR, "Train")
GZ_VAL_DIR      = os.path.join(GZ_BASE_DIR, "Validation")
GZ_TEST_DIR     = os.path.join(GZ_BASE_DIR, "Test")

# --- Model directories ---
MODEL_DIR       = os.path.join(BASE, "Models", "production")
ANALYSIS_DIR    = os.path.join(MODEL_DIR, "analysis")

# --- Model output files ---
DIRECTION_MODEL = os.path.join(MODEL_DIR, "direction_model.ubj")
HOLD_MODEL      = os.path.join(MODEL_DIR, "hold_model.ubj")
SPLIT_INFO      = os.path.join(MODEL_DIR, "split_info.json")
FEAT_IMPORTANCE = os.path.join(MODEL_DIR, "feature_importance.csv")
SIM_TRADES      = os.path.join(MODEL_DIR, "sim_trades.csv")
EQUITY_CURVE    = os.path.join(MODEL_DIR, "equity_curve.csv")
MODEL_REPORT    = os.path.join(MODEL_DIR, "model_report.txt")
