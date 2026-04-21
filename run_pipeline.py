"""
CVD Pipeline — Master Runner
=============================
Runs the complete pipeline from raw ITCH .gz files
to trained XGBoost model and simulation results.

Steps:
  1. step1_itch_parser.py         — Parse .gz files → per-day trade CSVs
  2. step2_stock_splitter.py      — Split per-day CSVs → per-stock files
  3. step3_feature_engineering.py — Build clean feature matrix
  4. step4_train_model.py         — Train XGBoost direction + hold models
  5. step5_simulate.py            — Walk-forward simulation on test days
  6. step6_model_analysis.py      — Charts, analysis, Excel export

GZ FILE ORGANISATION — place files in subfolders:
  E:\\CVD data\\Data\\Tick\\Tick GZ primary data files\\
    Train\\        ← training day(s)   — minimum 1 file
    Validation\\   ← validation day(s) — minimum 1 file
    Test\\         ← test day(s)       — minimum 1 file, drives simulation

  The pipeline auto-detects how many days are in each folder.
  N_TRAIN / N_VAL / N_TEST adjust automatically — no config needed.
  To add more test days: add more .gz files to Test\\ and rerun.
  To add more training:  add more .gz files to Train\\ and rerun step3+.

  Split logic:
    Train  : ALL files in Train\\     (1 or more)
    Val    : ALL files in Validation\\ (1 or more, used for early stopping)
    Test   : ALL files in Test\\      (1 or more, walk-forward simulation)

Auto-created folders:
  E:\\CVD data\\Data\\Tick\\raw_csv\\          (step 1)
  E:\\CVD data\\Data\\Tick\\split_by_stock\\   (step 2)
  E:\\CVD data\\Data\\Tick\\clean_features.csv (step 3)
  E:\\CVD data\\Models\\production\\           (steps 4-5)
  E:\\CVD data\\Models\\production\\analysis\\ (step 6)

To skip completed steps set the RUN_STEP_N flag to False below.

Usage:
  python run_pipeline.py
"""

import os
import sys
import time
import importlib

# =========================================================
# STEP FLAGS — set to False to skip a completed step
# =========================================================
# STEP 0 — CLEAN SLATE (double confirmation required)
# Deletes all pipeline-generated files for a full fresh run.
# NEVER deletes source .gz files.
# Both flags below must be True for step 0 to execute.
RUN_STEP_0        = False   # Master switch — set True to enable clean
RUN_STEP_0_CONFIRM= False   # Second confirmation failsafe — also set True

RUN_STEP_1 = False   # Parse .gz ITCH files → raw CSVs
RUN_STEP_2 = False   # Split raw CSVs → per-stock files
RUN_STEP_3 = True   # Build feature matrix (with split tags)
RUN_STEP_4 = True   # Train XGBoost models
RUN_STEP_5 = True    # Walk-forward simulation on test days
RUN_STEP_6 = True    # Full model analysis + charts

# =========================================================
# PATHS
# =========================================================
from config import GZ_BASE_DIR, GZ_TRAIN_DIR, GZ_VAL_DIR, GZ_TEST_DIR
BASE_ITCH_DIR = GZ_BASE_DIR
SPLIT_FOLDERS = {
    "Train":      GZ_TRAIN_DIR,
    "Validation": GZ_VAL_DIR,
    "Test":       GZ_TEST_DIR,
}


# =========================================================
# SANITY CHECK — verify subfolder structure and file counts
# =========================================================
def check_split_folders():
    from pathlib import Path
    print("Checking split folder structure...")
    total = 0
    for label, folder in SPLIT_FOLDERS.items():
        p = Path(folder)
        if not p.exists():
            print(f"  WARNING: {folder} does not exist — will be created by step1")
            continue
        gz_files = sorted(p.glob("*.gz"))
        n = len(gz_files)
        total += n
        if n == 0:
            print(f"  {label:12}: (empty)")
        else:
            print(f"  {label:12}: {n} file(s)")
            for f in gz_files:
                size_mb = f.stat().st_size / 1_048_576
                print(f"    {f.name}  ({size_mb:,.0f} MB)")

    if total == 0:
        print(f"\nERROR: No .gz files found in any subfolder of:")
        print(f"  {BASE_ITCH_DIR}")
        print(f"Create Train\\ Validation\\ Test\\ subfolders and place .gz files in them.")
        sys.exit(1)
    print()


# =========================================================
# MAIN
# =========================================================
def run():
    start_total = time.time()

    print("=" * 60)
    print("CVD PIPELINE — FULL RUN")
    print("=" * 60)
    print()

    check_split_folders()

    # ----------------------------------------------------------
    # STEP 0 — Clean slate (double confirmation required)
    # ----------------------------------------------------------
    if RUN_STEP_0:
        print("=" * 60)
        print("STEP 0 — CLEAN SLATE")
        print("=" * 60)
        import step0_clean
        importlib.reload(step0_clean)
        # Inject the confirmation flags from this file into step0
        step0_clean.CONFIRM_CLEAN   = RUN_STEP_0
        step0_clean.CONFIRM_CLEAN_2 = RUN_STEP_0_CONFIRM
        success = step0_clean.run()
        if not success:
            print("\n  Step 0 aborted — pipeline will not continue.")
            print("  Set both RUN_STEP_0 and RUN_STEP_0_CONFIRM to True to proceed.\n")
            return
        print()
    else:
        print("STEP 0 — skipped (RUN_STEP_0 = False)\n")
    # ----------------------------------------------------------
    if RUN_STEP_1:
        print("=" * 60)
        print("STEP 1 — ITCH Parser  (Train + Validation + Test)")
        print("=" * 60)
        t0 = time.time()
        import step1_itch_parser
        importlib.reload(step1_itch_parser)
        _, n_train, n_val, n_test = step1_itch_parser.run()
        print(f"\n  Step 1 elapsed: {(time.time()-t0)/60:.1f} minutes")
        print(f"  Detected: N_TRAIN={n_train}  N_VAL={n_val}  N_TEST={n_test}\n")
    else:
        print("STEP 1 — skipped (RUN_STEP_1 = False)\n")
        

    # ----------------------------------------------------------
    # STEP 2 — Split by stock
    # ----------------------------------------------------------
    if RUN_STEP_2:
        print("=" * 60)
        print("STEP 2 — Stock Splitter")
        print("=" * 60)
        t0 = time.time()
        import step2_stock_splitter
        importlib.reload(step2_stock_splitter)
        step2_stock_splitter.run()
        print(f"\n  Step 2 elapsed: {(time.time()-t0)/60:.1f} minutes\n")
    else:
        print("STEP 2 — skipped (RUN_STEP_2 = False)\n")

    # ----------------------------------------------------------
    # STEP 3 — Feature engineering (tags each row with split_key)
    # ----------------------------------------------------------
    if RUN_STEP_3:
        print("=" * 60)
        print("STEP 3 — Feature Engineering  (tags rows: train/val/test)")
        print("=" * 60)
        t0 = time.time()
        import step3_feature_engineering
        importlib.reload(step3_feature_engineering)
        step3_feature_engineering.run()
        print(f"\n  Step 3 elapsed: {(time.time()-t0)/60:.1f} minutes\n")
    else:
        print("STEP 3 — skipped (RUN_STEP_3 = False)\n")

    # ----------------------------------------------------------
    # STEP 4 — Train models (uses train + validation split_key rows)
    # ----------------------------------------------------------
    if RUN_STEP_4:
        print("=" * 60)
        print("STEP 4 — Train XGBoost Models")
        print("=" * 60)
        t0 = time.time()
        import step4_train_model
        importlib.reload(step4_train_model)
        split_info = step4_train_model.run()
        print(f"\n  Step 4 elapsed: {(time.time()-t0)/60:.1f} minutes\n")
    else:
        print("STEP 4 — skipped (RUN_STEP_4 = False)\n")
        split_info = {}

    # ----------------------------------------------------------
    # STEP 5 — Simulate on test days (auto-detects N_TEST)
    # ----------------------------------------------------------
    if RUN_STEP_5:
        print("=" * 60)
        print("STEP 5 — Walk-Forward Simulation")
        print("=" * 60)
        t0 = time.time()
        import step5_simulate
        importlib.reload(step5_simulate)
        results = step5_simulate.run()
        print(f"\n  Step 5 elapsed: {(time.time()-t0)/60:.1f} minutes\n")
    else:
        print("STEP 5 — skipped (RUN_STEP_5 = False)\n")
        results = {}

    # ----------------------------------------------------------
    # STEP 6 — Full model analysis + charts
    # ----------------------------------------------------------
    if RUN_STEP_6:
        print("=" * 60)
        print("STEP 6 — Model Analysis")
        print("=" * 60)
        t0 = time.time()
        import step6_model_analysis
        importlib.reload(step6_model_analysis)
        step6_model_analysis.run()
        print(f"\n  Step 6 elapsed: {(time.time()-t0)/60:.1f} minutes\n")
    else:
        print("STEP 6 — skipped (RUN_STEP_6 = False)\n")

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    total_mins = (time.time() - start_total) / 60
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total elapsed: {total_mins:.1f} minutes")

    if split_info:
        print(f"\n  Split used:")
        print(f"    Train      : {split_info.get('n_train', '?')} day(s) → {split_info.get('train_days', [])}")
        print(f"    Validation : {split_info.get('n_val', '?')} day(s) → {split_info.get('val_days', [])}")
        print(f"    Test       : {split_info.get('n_test', '?')} day(s) → {split_info.get('test_days', [])}")

    if results:
        print(f"\n  Final model performance (test set):")
        print(f"    PnL          : ${results.get('total_pnl', 0):,.2f}")
        print(f"    Win rate     : {results.get('win_rate', 0):.1%}")
        print(f"    Profit factor: {results.get('profit_factor', 0):.3f}")
        print(f"    Sharpe       : {results.get('sharpe', 0):.4f}")
        print(f"    Trades       : {results.get('n_trades', 0):,}")
        print(f"    Net return   : {results.get('net_pct', 0):+.2f}%")

    from config import MODEL_DIR, ANALYSIS_DIR
    print(f"\n  Models saved → {MODEL_DIR}")
    print(f"    direction_model.ubj  hold_model.ubj  split_info.json")
    print(f"    sim_trades.csv  equity_curve.csv  model_report.txt")
    print(f"  Analysis → {ANALYSIS_DIR}")


if __name__ == "__main__":
    run()
