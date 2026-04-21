# CVD Pipeline

XGBoost-based intraday trading pipeline using NASDAQ ITCH order flow data.
Trains direction and hold-time models from raw tick data and runs a walk-forward simulation.

---

## Requirements

Python 3.9+ recommended.

Install dependencies:
```bash
pip install xgboost pandas numpy matplotlib
```

---

## Setup — One Step

Open `config.py` and change the `BASE` path to wherever your data lives:

```python
# Windows
BASE = r"E:\CVD data"

# Mac / Linux
BASE = "/home/yourname/cvd_data"
```

That is the **only file you need to edit**. Every step imports its paths from `config.py`.

---

## Folder Structure

Create this folder structure under your `BASE` directory before running.  
The pipeline will auto-create subfolders it writes to — you only need to create the source data folders.

```
<BASE>/
│
├── Data/
│   └── Tick/
│       ├── Tick GZ primary data files/
│       │   ├── Train/          ← place training .gz files here (1 or more)
│       │   ├── Validation/     ← place validation .gz files here (1 or more)
│       │   └── Test/           ← place test .gz files here (1 or more)
│       │
│       ├── raw_csv/            ← auto-created by Step 1
│       ├── split_by_stock/     ← auto-created by Step 2  ← START HERE if skipping Step 1+2
│       └── clean_features.csv  ← auto-created by Step 3
│
└── Models/
    └── production/             ← auto-created by Step 4
        └── analysis/           ← auto-created by Step 6
```

---

## Large Data Files

The following files are **not included in this repo** due to size.  
Download them from **[https://drive.google.com/drive/folders/0ALhcrDeWOPhYUk9PVA?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto]** and place them in the correct folders above.

| File | Destination | Created by |
|------|-------------|------------|
| `split_by_stock/*.csv` | `Data/Tick/split_by_stock/` | Step 2 |
| `clean_features.csv` | `Data/Tick/` | Step 3 |
| `direction_model.ubj` | `Models/production/` | Step 4 |
| `hold_model.ubj` | `Models/production/` | Step 4 |
| `split_info.json` | `Models/production/` | Step 4 |

If you have the `split_by_stock/` CSVs but not the others, start from **Step 3**.  
If you have `clean_features.csv` but not the models, start from **Step 4**.

---

## Running the Pipeline

Open `run_pipeline.py` and set the step flags to match where you are starting from:

```python
RUN_STEP_0 = False   # Clean slate — wipes all generated files (needs double confirmation)
RUN_STEP_1 = False   # Parse raw .gz ITCH files → per-day CSVs
RUN_STEP_2 = False   # Split per-day CSVs → per-stock files
RUN_STEP_3 = True    # Build feature matrix
RUN_STEP_4 = True    # Train XGBoost models
RUN_STEP_5 = True    # Walk-forward simulation on test days
RUN_STEP_6 = True    # Charts, analysis, Excel export
```

Then run:
```bash
python run_pipeline.py
```

---

## What Each Step Does

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 0 | `step0_clean.py` | — | Deletes all generated files (never touches .gz source data) |
| 1 | `step1_itch_parser.py` | `.gz` files in Train/Val/Test | `raw_csv/<split>_<DATE>.csv` |
| 2 | `step2_stock_splitter.py` | `raw_csv/` | `split_by_stock/<SYMBOL>_<split>_<DATE>_trades_depth.csv` |
| 3 | `step3_feature_engineering.py` | `split_by_stock/` | `clean_features.csv` |
| 4 | `step4_train_model.py` | `clean_features.csv` | `direction_model.ubj`, `hold_model.ubj`, `split_info.json` |
| 5 | `step5_simulate.py` | models + `clean_features.csv` | `sim_trades.csv`, `equity_curve.csv`, `model_report.txt` |
| 6 | `step6_model_analysis.py` | Step 5 outputs | Charts + `model_summary.txt` in `analysis/` |

---

## Adding More Data

- **More training days**: drop additional `.gz` files into `Train/` and rerun from Step 3 onwards
- **More test days**: drop additional `.gz` files into `Test/` and rerun from Step 3 onwards  
- No config changes needed — the pipeline auto-detects how many files are in each folder

---

## Output Files

After a full run, results land in `Models/production/`:

```
direction_model.ubj       trained direction classifier
hold_model.ubj            trained hold-time regressor
feature_importance.csv    feature gain scores
split_info.json           train/val/test day counts and dates
sim_trades.csv            all simulated trades across test days
equity_curve.csv          day-by-day equity tracker
model_report.txt          summary statistics

analysis/
  model_summary.txt
  equity_curve.png
  daily_performance.png
  band_analysis.png
  feature_importance.png
  trade_characteristics.png
  confidence_calibration.png
```
