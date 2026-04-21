"""
Step 4 — XGBoost Model Training
================================
Trains direction + hold models on clean_features.csv.

The feature matrix produced by step3 contains a 'split_key' column
that marks each row as 'train', 'validation', or 'test' — derived
directly from which folder the source .gz file was placed in.

  Train      rows → model learning
  Validation rows → early stopping / hyperparameter signal
  Test       rows → withheld; simulated in step5

The number of days in each split is determined entirely by how many
.gz files you placed in each subfolder in step1. No config needed.

Outputs: E:\\CVD data\\Models\\production\\
  direction_model.ubj
  hold_model.ubj
  feature_importance.csv
  split_info.json          N_TRAIN / N_VAL / N_TEST for step5

Usage:
  python step4_train_model.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import xgboost as xgb

# =========================================================
# USER SETTINGS
# =========================================================
from config import FEATURES_FILE, MODEL_DIR
INPUT_FILE = FEATURES_FILE
OUTPUT_DIR = MODEL_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# FEATURE SET
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
assert "symbol"    not in FEATURE_COLS
assert "date_tag"  not in FEATURE_COLS
assert "split_key" not in FEATURE_COLS

# =========================================================
# MODEL PARAMS
# =========================================================
DIR_PARAMS = {
    "objective": "multi:softprob", "num_class": 3,
    "eval_metric": "mlogloss", "max_depth": 4,
    "learning_rate": 0.05, "n_estimators": 500,
    "subsample": 0.7, "colsample_bytree": 0.7,
    "min_child_weight": 50, "gamma": 2.0,
    "reg_alpha": 1.0, "reg_lambda": 5.0,
    "tree_method": "hist", "random_state": 42,
    "n_jobs": -1, "verbosity": 0,
}
HOLD_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "mae",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.7, "colsample_bytree": 0.7,
    "min_child_weight": 50, "gamma": 1.0,
    "reg_alpha": 0.5, "reg_lambda": 3.0,
    "tree_method": "hist", "random_state": 42,
    "n_jobs": -1, "verbosity": 0,
}

label_map     = {-1: 0, 0: 1, 1: 2}
label_inv_map = {0: -1, 1: 0, 2: 1}


# =========================================================
# MAIN
# =========================================================
def run():
    print("Loading feature matrix...")
    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values(["date_tag", "timestamp_ns"]).reset_index(drop=True)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    if "split_key" not in df.columns:
        raise ValueError(
            "'split_key' column not found in feature matrix.\n"
            "Re-run step3_feature_engineering.py to regenerate with split tags."
        )

    # ---- Derive split info from the data itself ----
    train_days = sorted(df[df["split_key"] == "train"]["date_tag"].unique())
    val_days   = sorted(df[df["split_key"] == "validation"]["date_tag"].unique())
    test_days  = sorted(df[df["split_key"] == "test"]["date_tag"].unique())

    n_train = len(train_days)
    n_val   = len(val_days)
    n_test  = len(test_days)

    if n_train == 0:
        raise ValueError("No 'train' rows found. Check split_key column and step1/step3.")
    if n_val == 0:
        raise ValueError("No 'validation' rows found. Check split_key column and step1/step3.")
    if n_test == 0:
        raise ValueError("No 'test' rows found. Check split_key column and step1/step3.")

    total_rows = len(df)
    print(f"  {total_rows:,} rows total")
    print(f"  Train      : {n_train} day(s) → {train_days}")
    print(f"  Validation : {n_val} day(s) → {val_days}")
    print(f"  Test       : {n_test} day(s) → {test_days}  ← simulated in step5")

    df_train = df[df["split_key"] == "train"].copy()
    df_val   = df[df["split_key"] == "validation"].copy()
    fc       = [c for c in FEATURE_COLS if c in df.columns]

    X_tr = df_train[fc].values.astype(np.float32)
    X_va = df_val[fc].values.astype(np.float32)
    y_tr = df_train["target_direction"].map(label_map).values.astype(int)
    y_va = df_val["target_direction"].map(label_map).values.astype(int)
    w_tr = df_train["sample_weight"].fillna(0).values

    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=fc)
    dval   = xgb.DMatrix(X_va, label=y_va,              feature_names=fc)

    # ---- Direction model ----
    print("\nTraining direction model...")
    dir_mdl = xgb.train(
        DIR_PARAMS, dtrain,
        num_boost_round=DIR_PARAMS["n_estimators"],
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30, verbose_eval=50,
    )
    probs_va = dir_mdl.predict(
        xgb.DMatrix(X_va, feature_names=fc)
    ).reshape(len(X_va), -1)
    acc_va = (probs_va.argmax(axis=1) == y_va).mean()
    print(f"  Trees: {dir_mdl.num_boosted_rounds()}  Val acc: {acc_va:.1%}")

    # ---- Hold model ----
    print("\nTraining hold model...")
    hm = df_train["target_direction"].values != 0
    hv = df_val["target_direction"].values   != 0
    hold_mdl = xgb.train(
        HOLD_PARAMS,
        xgb.DMatrix(X_tr[hm],
                    label=df_train["target_hold_n"].fillna(0).values[hm],
                    weight=w_tr[hm], feature_names=fc),
        num_boost_round=HOLD_PARAMS["n_estimators"],
        evals=[
            (xgb.DMatrix(X_tr[hm],
                         label=df_train["target_hold_n"].fillna(0).values[hm],
                         feature_names=fc), "train"),
            (xgb.DMatrix(X_va[hv],
                         label=df_val["target_hold_n"].fillna(0).values[hv],
                         feature_names=fc), "val"),
        ],
        early_stopping_rounds=20, verbose_eval=False,
    )
    print(f"  Hold model trained.")

    # ---- Feature importance ----
    imp_df = pd.DataFrame(
        dir_mdl.get_score(importance_type="gain").items(),
        columns=["feature", "gain"]
    ).sort_values("gain", ascending=False).reset_index(drop=True)

    # ---- Save ----
    dir_mdl.save_model(os.path.join(OUTPUT_DIR, "direction_model.ubj"))
    hold_mdl.save_model(os.path.join(OUTPUT_DIR, "hold_model.ubj"))
    imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    # Save split info so step5 knows how many test days to simulate
    split_info = {
        "n_train":    n_train,
        "n_val":      n_val,
        "n_test":     n_test,
        "train_days": train_days,
        "val_days":   val_days,
        "test_days":  test_days,
        "val_acc":    round(acc_va, 4),
        "n_trees":    dir_mdl.num_boosted_rounds(),
        "features":   fc,
    }
    with open(os.path.join(OUTPUT_DIR, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nStep 4 complete → {OUTPUT_DIR}")
    print(f"  direction_model.ubj  hold_model.ubj")
    print(f"  feature_importance.csv  split_info.json")
    print(f"  Val acc: {acc_va:.1%}  |  Train days: {n_train}  "
          f"Val days: {n_val}  Test days: {n_test}")
    print(f"\nRun step5_simulate.py next.")

    return split_info


if __name__ == "__main__":
    run()
