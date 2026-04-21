"""
Step 3 — Feature Engineering
=============================
Reads all per-stock files from split_by_stock/ and builds the
clean feature matrix used for XGBoost training.

Features (39 total — pure order flow, no price or session features):
  CVD        : cvd_z, cvd_vel_3/5/10/20, cvd_accel
  Pressure   : pressure_3/5/10/20, pressure_roc_5/10, intensity_5/10
  Depth      : depth_l1_avg_norm, depth_l5_total_norm,
               depth_l5_weighted_norm, bid/ask_depth_l5_weighted_norm
  Depth ROC  : depth_roc_3/5/10, bid_ask_depth_roc_3/5/10
  Imbalance  : l1/l5/l5_weighted_imbalance
  Imb ROC    : l1/l5/l5_weighted_imbalance_roc_3/5/10
  Imb Accel  : l1/l5/l5_weighted_imbalance_accel

Settings:
  CVD bucket : 3 seconds (empirically best from buffer study)
  Threshold  : $0.10 (gives ~50% flat labels)
  Max hold   : 30 trades

Output:
  E:\\CVD data\\Data\\Tick\\clean_features.csv

Usage (called from run_pipeline.py or standalone):
  python step3_feature_engineering.py
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================
# USER SETTINGS
# =========================================================
from config import SPLIT_DIR, FEATURES_FILE, RAW_CSV_DIR
INPUT_DIR   = SPLIT_DIR
OUTPUT_FILE = FEATURES_FILE

# THRESHOLD is 0.05% of entry price — price-agnostic, percentage-based.
# Captures minor price bumps consistently across all price levels:
#   INTC $50 → $0.025   MSFT $150 → $0.075   TSLA $480 → $0.240   AMZN $1800 → $0.900
# Must stay consistent with tp_pct in step5 CONFIDENCE_BANDS.
# NEVER revert to a flat dollar amount — that biases toward low-priced stocks.
THRESHOLD_PCT       = 0.05            # % of trade_price — minor price bump target
MAX_HOLD_TRADES     = 30              # upper cap on hold window
BUCKET_NS           = 3_000_000_000   # 3s — buffer study winner
ROLLING_NORM_WINDOW = 100
PRESSURE_WINDOWS    = [3, 5, 10, 20]
# No MIN_SIGNAL_RATE filter — never filter any split by signal rate.
# Quiet days are real market behaviour the model must learn.


# =========================================================
# STEP 1 — LOAD & FILTER
# =========================================================
def load_and_filter(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[df["msg_type"] != "Q"].copy()
    df = df[df["aggressor_side"].isin(["B", "S"])].copy()
    df["trade_price"]  = pd.to_numeric(df["trade_price"],  errors="coerce")
    df["trade_shares"] = pd.to_numeric(df["trade_shares"], errors="coerce")
    df = df[(df["trade_price"] > 0) & (df["trade_shares"] > 0)].copy()
    df["_hhmm"] = df["time_str"].str[:5]
    df = df[(df["_hhmm"] >= "09:30") & (df["_hhmm"] < "16:00")].drop(columns=["_hhmm"])
    df["order_reference"] = df["order_reference"].fillna(0)
    return df.reset_index(drop=True)


# =========================================================
# STEP 2 — CVD FEATURES  (causal — no intra-bucket look-ahead)
# =========================================================
def add_cvd_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each row uses its own causal CVD (cumsum up to that trade).
    Velocity features use time-based lookback: cvd_vel_N is the change
    in CVD over the last N × BUCKET_NS nanoseconds, anchored to each
    row's own timestamp.  No future information is used.
    """
    df["timestamp_ns"] = pd.to_numeric(df["timestamp_ns"], errors="coerce")
    df["_delta"]       = np.where(
        df["aggressor_side"] == "B", df["trade_shares"], -df["trade_shares"]
    )
    df["bucket_idx"] = (df["timestamp_ns"] // BUCKET_NS).astype("int64")
    df["cvd"]        = df["_delta"].cumsum()

    ts  = df["timestamp_ns"].values
    cvd = df["cvd"].values
    n   = len(ts)

    # ---- Velocity: CVD change over N × BUCKET_NS lookback ----
    # searchsorted finds the row closest to (t − lookback) efficiently.
    vel_3 = None

    # 0.5-second lookback (independent of BUCKET_NS)
    HALF_SEC_NS = 500_000_000
    target_times_05s = ts - HALF_SEC_NS
    idx_05s = np.searchsorted(ts, target_times_05s, side="right") - 1
    idx_05s = np.clip(idx_05s, 0, n - 1)
    vel_05s = (cvd - cvd[idx_05s]).astype(np.float64)
    vel_05s[target_times_05s < ts[0]] = np.nan
    df["cvd_vel_05s"] = vel_05s

    # 1-second lookback (independent of BUCKET_NS)
    ONE_SEC_NS = 1_000_000_000
    target_times_1s = ts - ONE_SEC_NS
    idx_1s = np.searchsorted(ts, target_times_1s, side="right") - 1
    idx_1s = np.clip(idx_1s, 0, n - 1)
    vel_1s = (cvd - cvd[idx_1s]).astype(np.float64)
    vel_1s[target_times_1s < ts[0]] = np.nan
    df["cvd_vel_1s"] = vel_1s

    for num_buckets in [3, 5, 10, 20]:
        lookback_ns  = num_buckets * BUCKET_NS
        target_times = ts - lookback_ns
        # Index of the last row at or before target_time
        idx = np.searchsorted(ts, target_times, side="right") - 1
        idx = np.clip(idx, 0, n - 1)
        vel = (cvd - cvd[idx]).astype(np.float64)
        # Mark rows where the lookback reaches before the first trade
        vel[target_times < ts[0]] = np.nan
        df[f"cvd_vel_{num_buckets}"] = vel
        if num_buckets == 3:
            vel_3 = vel

    # ---- Acceleration: change in vel_3 over another 3-bucket span ----
    accel_lookback = 3 * BUCKET_NS
    target_times   = ts - accel_lookback
    idx            = np.searchsorted(ts, target_times, side="right") - 1
    idx            = np.clip(idx, 0, n - 1)
    prev_vel_3     = np.where(target_times >= ts[0], vel_3[idx], np.nan)
    df["cvd_accel"] = np.where(~np.isnan(vel_3) & ~np.isnan(prev_vel_3),
                               vel_3 - prev_vel_3, np.nan)

    # ---- Z-score: rolling normalisation on per-row CVD ----
    cvd_s = df["cvd"]
    rm    = cvd_s.rolling(ROLLING_NORM_WINDOW, min_periods=5).mean()
    rs    = cvd_s.rolling(ROLLING_NORM_WINDOW, min_periods=5).std().replace(0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["cvd_z"] = (cvd_s - rm) / rs

    df = df.drop(columns=["_delta"])
    return df


# =========================================================
# STEP 3 — PRESSURE FEATURES
# =========================================================
def add_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    shares  = df["trade_shares"].values.astype(float)
    is_buy  = (df["aggressor_side"].values == "B").astype(float)
    n       = len(df)
    cum_buy = np.cumsum(shares * is_buy)
    cum_sel = np.cumsum(shares * (1 - is_buy))

    for w in PRESSURE_WINDOWS:
        bv = np.zeros(n); sv = np.zeros(n)
        bv[w:] = cum_buy[w:] - cum_buy[:-w]
        sv[w:] = cum_sel[w:] - cum_sel[:-w]
        tv = bv + sv
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"pressure_{w}"] = np.where(tv > 0, (bv - sv) / tv, 0.0)
        raw = pd.Series(tv)
        med = raw.rolling(ROLLING_NORM_WINDOW, min_periods=5).median()
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"intensity_{w}"] = raw / (med + 1e-8)

    for w in [5, 10]:
        df[f"pressure_roc_{w}"] = pd.Series(df[f"pressure_{w}"].values).diff(w)
    return df


# =========================================================
# STEP 4 — DEPTH FEATURES
# =========================================================
def add_depth_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["depth_l5_total", "depth_l5_weighted", "depth_l1_avg",
                "bid_depth_l5_weighted", "ask_depth_l5_weighted"]:
        if col not in df.columns: continue
        num = pd.to_numeric(df[col], errors="coerce")
        med = num.rolling(ROLLING_NORM_WINDOW, min_periods=5).median().replace(0, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"{col}_norm"] = num / med

    if "depth_l5_total" in df.columns:
        d = pd.to_numeric(df["depth_l5_total"], errors="coerce")
        for w in [3, 5, 10]:
            df[f"depth_roc_{w}"] = d.diff(w)

    if "bid_depth_l5" in df.columns and "ask_depth_l5" in df.columns:
        bid = pd.to_numeric(df["bid_depth_l5"], errors="coerce")
        ask = pd.to_numeric(df["ask_depth_l5"], errors="coerce")
        for w in [3, 5, 10]:
            df[f"bid_ask_depth_roc_{w}"] = (bid - ask).diff(w)

    for col in ["l1_imbalance", "l5_imbalance", "l5_weighted_imbalance"]:
        if col not in df.columns: continue
        imb = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = imb
        for w in [3, 5, 10]:
            df[f"{col}_roc_{w}"] = imb.diff(w)
        df[f"{col}_accel"] = imb.diff(3).diff(3)
    return df


# =========================================================
# STEP 5 — TARGET LABELS
# =========================================================
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    prices     = df["trade_price"].values
    n          = len(prices)
    directions = np.zeros(n, dtype=np.int8)
    hold_n     = np.full(n, np.nan)

    # Dynamic hold window — 0.05% of daily tick count, clamped 10-30.
    # Keeps the lookahead proportional to each stock's natural trading pace.
    max_hold = int(np.clip(n * 0.0005, 10, MAX_HOLD_TRADES))

    for i in range(n):
        p0 = prices[i]
        # Percentage threshold — price-agnostic labelling.
        # threshold_dollar = price * THRESHOLD_PCT / 100
        # Must match tp_pct in step5 CONFIDENCE_BANDS. NEVER use flat dollar.
        threshold_dollar = p0 * THRESHOLD_PCT / 100.0
        for j in range(i + 1, min(i + max_hold + 1, n)):
            diff = prices[j] - p0
            if diff >= threshold_dollar:
                directions[i] = 1;  hold_n[i] = j - i; break
            elif diff <= -threshold_dollar:
                directions[i] = -1; hold_n[i] = j - i; break

    df["target_direction"] = directions
    df["target_hold_n"]    = hold_n
    df["realized_pnl"]     = np.where(
        directions != 0,
        prices * THRESHOLD_PCT / 100.0,
        0.0
    )
    return df


# =========================================================
# STEP 6 — SAMPLE WEIGHTS
# =========================================================
def add_sample_weights(df: pd.DataFrame) -> pd.DataFrame:
    pressure   = df.get("pressure_5", pd.Series(0, index=df.index)).abs().fillna(0)
    vel        = df.get("cvd_vel_10", pd.Series(0, index=df.index)).abs().fillna(0)
    vel_max    = vel.quantile(0.99)
    vel_norm   = (vel / vel_max).clip(0, 1) if vel_max > 0 else vel * 0
    depth_norm = df.get("depth_l5_weighted_norm",
                        pd.Series(1, index=df.index)).fillna(1)
    thinness   = (1 / depth_norm.clip(lower=0.1)).clip(upper=3)
    df["sample_weight"] = (
        pressure * (1 + vel_norm) * thinness * df["realized_pnl"].abs()
    )
    if df["sample_weight"].sum() == 0:
        df["sample_weight"] = 1.0
    return df


# =========================================================
# FEATURE + META COLUMNS
# =========================================================
FEATURE_COLS = [
    "cvd_z", "cvd_vel_05s", "cvd_vel_1s", "cvd_vel_3", "cvd_vel_5", "cvd_vel_10", "cvd_vel_20", "cvd_accel",
    "pressure_3", "pressure_5", "pressure_10", "pressure_20",
    "pressure_roc_5", "pressure_roc_10", "intensity_5", "intensity_10",
    "depth_l1_avg_norm", "depth_l5_total_norm", "depth_l5_weighted_norm",
    "bid_depth_l5_weighted_norm", "ask_depth_l5_weighted_norm",
    "depth_roc_3", "depth_roc_5", "depth_roc_10",
    "bid_ask_depth_roc_3", "bid_ask_depth_roc_5", "bid_ask_depth_roc_10",
    "l1_imbalance", "l5_imbalance", "l5_weighted_imbalance",
    "l1_imbalance_roc_3", "l1_imbalance_roc_5",
    "l5_imbalance_roc_3", "l5_imbalance_roc_5", "l5_imbalance_roc_10",
    "l5_weighted_imbalance_roc_3", "l5_weighted_imbalance_roc_5",
    "l5_weighted_imbalance_roc_10",
    "l1_imbalance_accel", "l5_imbalance_accel", "l5_weighted_imbalance_accel",
]

META_COLS = [
    "timestamp_ns", "time_str", "symbol", "date_tag", "split_key",
    "trade_price", "trade_shares", "msg_type", "aggressor_side",
    "bucket_idx", "target_direction", "target_hold_n",
    "realized_pnl", "sample_weight",
]


# =========================================================
# PROCESS ONE STOCK FILE
# =========================================================
def process_file(path: str, split_lookup: dict = None):
    basename  = os.path.basename(path)
    # New format: SYMBOL_split_DATETAG_trades_depth.csv
    m_new = re.match(r'^(.+?)_(train|validation|test)_(S\d{6})_trades_depth\.csv$', basename)
    if m_new:
        symbol    = m_new.group(1)
        split_key = m_new.group(2)
        date_tag  = m_new.group(3)
    else:
        # Legacy format: SYMBOL_DATETAG_trades_depth.csv
        m_leg    = re.search(r'(S\d{6})', basename)
        date_tag = m_leg.group(1) if m_leg else "UNKNOWN"
        symbol   = re.sub(r'_(S\d{6})_trades_depth\.csv$', '', basename)
        # Look up split from raw_csv scan — correct even for legacy filenames
        split_key = (split_lookup or {}).get(date_tag, "train")
    print(f"  {symbol} [{split_key}] ({date_tag})...", end=" ", flush=True)

    df = load_and_filter(path)
    # Row filter is TRAIN ONLY — val/test always kept including quiet days.
    if len(df) < 100 and split_key == "train":
        print("skipped — too few rows (train only)")
        return None
    elif len(df) < 10:
        print(f"skipped — only {len(df)} rows (hard minimum)")
        return None

    df["date_tag"]  = date_tag
    df["split_key"] = split_key
    df = add_cvd_features(df)
    df = add_pressure_features(df)
    df = add_depth_features(df)
    df = add_targets(df)

    df = add_sample_weights(df)

    keep = [c for c in FEATURE_COLS + META_COLS if c in df.columns]
    df   = df[keep].copy()
    feat_present = [c for c in FEATURE_COLS if c in df.columns]
    df   = df.dropna(subset=feat_present).reset_index(drop=True)

    print(f"{len(df):,} rows  "
          f"flat={(df['target_direction']==0).mean():.0%}  "
          f"up={(df['target_direction']==1).mean():.0%}  "
          f"dn={(df['target_direction']==-1).mean():.0%}")
    return df


# =========================================================
# RUN
# =========================================================
# RAW_CSV_DIR imported from config above


def build_split_lookup():
    """
    Scan the raw_csv folder for files named <split>_<DATE_TAG>.csv
    (produced by the new step1) and return a dict: date_tag → split_key.
    Falls back gracefully if only legacy format files exist.
    """
    from pathlib import Path
    lookup = {}
    raw_dir = Path(RAW_CSV_DIR)
    if raw_dir.exists():
        for f in raw_dir.glob("*.csv"):
            m = re.match(r'^(train|validation|test)_(S\d{6})\.csv$', f.name)
            if m:
                lookup[m.group(2)] = m.group(1)
    if lookup:
        print(f"  Split lookup built from raw_csv: {lookup}")
    else:
        print(f"  No split-prefixed CSVs found in raw_csv — all files will be tagged 'train'")
    return lookup


def run():
    # Build date_tag → split_key mapping from raw_csv filenames
    split_lookup = build_split_lookup()

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_trades_depth.csv")))
    if not files:
        raise FileNotFoundError(
            f"No *_trades_depth.csv files in {INPUT_DIR}\n"
            f"Run step2_stock_splitter.py first."
        )

    print(f"Found {len(files)} stock-day files")
    print(f"CVD bucket  : {BUCKET_NS/1e9:.1f}s")
    print(f"Threshold   : {THRESHOLD_PCT}% of price")
    print(f"Features    : {len(FEATURE_COLS)}\n")

    frames = []
    for path in files:
        df = process_file(path, split_lookup)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("All files skipped — check input data.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["date_tag", "symbol", "timestamp_ns"]
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    # Print split breakdown
    if "split_key" in combined.columns:
        split_counts = combined.groupby("split_key")["date_tag"].nunique().to_dict()
        print(f"\n{'='*50}")
        print(f"Step 3 complete")
        print(f"  Rows     : {len(combined):,}")
        print(f"  Split    : {split_counts}")
        print(f"  Days     : {combined['date_tag'].value_counts().to_dict()}")
        print(f"  Labels   : {combined['target_direction'].value_counts(normalize=True).round(3).to_dict()}")
        nulls = combined[[c for c in FEATURE_COLS if c in combined.columns]].isnull().sum()
        print(f"  Nulls    : {nulls[nulls>0].to_dict() if nulls.sum() > 0 else 'None'}")
        print(f"  Output   : {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    run()
