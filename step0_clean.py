"""
Step 0 — Clean Slate
=====================
Deletes all pipeline-generated files so the next full run starts fresh.
NEVER deletes raw source data (.gz files in Train/Validation/Test folders).

DOUBLE CONFIRMATION REQUIRED — both flags must be True to proceed:
  CONFIRM_CLEAN = True
  CONFIRM_CLEAN_2 = True

What gets deleted:
  raw_csv/              Step 1 output — parsed per-day trade CSVs
  split_by_stock/       Step 2 output — per-stock files
  clean_features.csv    Step 3 output — feature matrix
  Models/production/    Steps 4-6 output — models, sim results, analysis

What is NEVER touched:
  Tick GZ primary data files/Train/       Source .gz files
  Tick GZ primary data files/Validation/  Source .gz files
  Tick GZ primary data files/Test/        Source .gz files

Usage:
  Set both CONFIRM_CLEAN flags to True, then run:
    python step0_clean.py
  Or call step0_clean.run() from run_pipeline.py (RUN_STEP_0 = True)
"""

import os
import shutil
from pathlib import Path

# =========================================================
# DOUBLE CONFIRMATION — BOTH must be True to proceed
# =========================================================
CONFIRM_CLEAN   = False   # Set to True — first confirmation
CONFIRM_CLEAN_2 = False   # Set to True — second confirmation (failsafe)

from config import RAW_CSV_DIR, SPLIT_DIR, FEATURES_FILE, MODEL_DIR, GZ_TRAIN_DIR, GZ_VAL_DIR, GZ_TEST_DIR

# =========================================================
# PATHS — what gets deleted
# =========================================================
PATHS_TO_DELETE = [
    RAW_CSV_DIR,
    SPLIT_DIR,
    FEATURES_FILE,
    MODEL_DIR,
]

# =========================================================
# PATHS — source data, NEVER deleted (sanity check)
# =========================================================
PROTECTED_PATHS = [
    GZ_TRAIN_DIR,
    GZ_VAL_DIR,
    GZ_TEST_DIR,
]


# =========================================================
# MAIN
# =========================================================
def run():
    print("=" * 60)
    print("STEP 0 — CLEAN SLATE")
    print("=" * 60)

    # ---- Double confirmation check ----
    if not CONFIRM_CLEAN or not CONFIRM_CLEAN_2:
        print("\n  ABORTED — double confirmation required.")
        print("  Both flags must be True to proceed:")
        print(f"    CONFIRM_CLEAN   = {CONFIRM_CLEAN}")
        print(f"    CONFIRM_CLEAN_2 = {CONFIRM_CLEAN_2}")
        print("\n  Open step0_clean.py and set both to True, then rerun.")
        return False

    # ---- Verify protected paths are untouched ----
    print("\n  Protected source data (will NOT be deleted):")
    for p in PROTECTED_PATHS:
        path = Path(p)
        if path.exists():
            gz_count = len(list(path.glob("*.gz")))
            print(f"    {path}  ({gz_count} .gz files) — SAFE")
        else:
            print(f"    {path}  (does not exist) — SAFE")

    # ---- Show what will be deleted ----
    print("\n  The following will be permanently deleted:")
    total_size_mb = 0
    items_found = []
    for p in PATHS_TO_DELETE:
        path = Path(p)
        if not path.exists():
            print(f"    {path}  — not found, skipping")
            continue
        if path.is_file():
            size_mb = path.stat().st_size / 1_048_576
            print(f"    FILE  {path}  ({size_mb:,.1f} MB)")
            total_size_mb += size_mb
            items_found.append(path)
        elif path.is_dir():
            size_mb = sum(
                f.stat().st_size for f in path.rglob("*") if f.is_file()
            ) / 1_048_576
            n_files = sum(1 for f in path.rglob("*") if f.is_file())
            print(f"    DIR   {path}  ({n_files:,} files, {size_mb:,.1f} MB)")
            total_size_mb += size_mb
            items_found.append(path)

    if not items_found:
        print("\n  Nothing to delete — already clean.")
        return True

    print(f"\n  Total to free: ~{total_size_mb:,.0f} MB")
    print("\n  Proceeding with deletion...")

    # ---- Delete ----
    deleted = []
    errors  = []
    for path in items_found:
        try:
            if path.is_file():
                path.unlink()
                print(f"    Deleted file: {path.name}")
            elif path.is_dir():
                shutil.rmtree(path)
                print(f"    Deleted dir:  {path.name}/")
            deleted.append(path)
        except Exception as e:
            print(f"    ERROR deleting {path}: {e}")
            errors.append((path, e))

    print(f"\n{'=' * 60}")
    print(f"  Step 0 complete — {len(deleted)} item(s) deleted")
    if errors:
        print(f"  WARNING: {len(errors)} deletion(s) failed — check above")
    print(f"  Pipeline-generated files cleared. Source .gz files untouched.")
    print(f"{'=' * 60}")

    # ---- Reset both flags reminder ----
    print("\n  REMINDER: Set both CONFIRM_CLEAN flags back to False")
    print("  before your next run to prevent accidental re-deletion.\n")

    return len(errors) == 0


if __name__ == "__main__":
    run()
