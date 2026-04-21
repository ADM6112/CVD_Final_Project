"""
Step 2 — Stock Splitter
=======================
Splits each per-day raw CSV (output of step1_itch_parser.py)
into individual per-stock files, preserving the split prefix.

Input  : E:\\CVD data\\Data\\Tick\\raw_csv\\<split>_<DATE_TAG>.csv
           e.g. train_S081321.csv  val_S083019.csv  test_S101819.csv
Output : E:\\CVD data\\Data\\Tick\\split_by_stock\\<SYMBOL>_<split>_<DATE_TAG>_trades_depth.csv
           e.g. TSLA_train_S081321_trades_depth.csv

Usage (called from run_pipeline.py or standalone):
  python step2_stock_splitter.py
"""

import csv
import os
import re
from pathlib import Path

# =========================================================
# USER SETTINGS
# =========================================================
from config import RAW_CSV_DIR, SPLIT_DIR
INPUT_DIR  = RAW_CSV_DIR
OUTPUT_DIR = SPLIT_DIR

# Known split prefixes — used to parse filenames
SPLIT_PREFIXES = ("train", "validation", "test")


# =========================================================
# SPLIT ONE FILE
# =========================================================
def split_file(input_path: str, output_dir: str) -> int:
    basename = os.path.basename(input_path)

    # Parse filename: <split>_<DATE_TAG>.csv
    # e.g. train_S081321.csv  or  test_S013019.csv
    m = re.match(r'^(train|validation|test)_(S\d{6})\.csv$', basename)
    if not m:
        print(f"  Cannot parse split/date from {basename} — skipping")
        return 0

    split_key = m.group(1)   # e.g. "train"
    date_tag  = m.group(2)   # e.g. "S081321"

    writers = {}
    files   = {}

    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames

        for row in reader:
            symbol = row["symbol"]
            key    = f"{symbol}_{split_key}_{date_tag}"

            if key not in writers:
                out_path = os.path.join(
                    output_dir,
                    f"{symbol}_{split_key}_{date_tag}_trades_depth.csv"
                )
                fh     = open(out_path, "w", newline="", encoding="utf-8")
                writer = csv.DictWriter(fh, fieldnames=header)
                writer.writeheader()
                writers[key] = writer
                files[key]   = fh

            writers[key].writerow(row)

    for fh in files.values():
        fh.close()

    print(f"  [{split_key}] {date_tag} → {len(files)} stock files")
    return len(files)


# =========================================================
# PROCESS ALL RAW CSVs
# =========================================================
def run():
    # Match both old format (selected_stocks_S*.csv) and new format (<split>_S*.csv)
    raw_csvs = sorted(
        list(Path(INPUT_DIR).glob("train_*.csv")) +
        list(Path(INPUT_DIR).glob("validation_*.csv")) +
        list(Path(INPUT_DIR).glob("test_*.csv"))
    )

    if not raw_csvs:
        raise FileNotFoundError(
            f"No split CSV files found in {INPUT_DIR}\n"
            f"Expected: train_S*.csv, validation_S*.csv, test_S*.csv\n"
            f"Run step1_itch_parser.py first."
        )

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Found {len(raw_csvs)} raw CSV files\n")

    total_files = 0
    for path in raw_csvs:
        print(f"Splitting {path.name}...")
        n = split_file(str(path), OUTPUT_DIR)
        total_files += n

    print(f"\n{'='*50}")
    print(f"Step 2 complete — {total_files} stock files written to {OUTPUT_DIR}")
    return OUTPUT_DIR


if __name__ == "__main__":
    run()
