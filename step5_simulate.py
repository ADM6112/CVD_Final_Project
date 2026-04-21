"""
Step 5 — Walk-Forward Simulation
==================================
Loads the trained models from step4 and simulates the test days
in chronological walk-forward order.

Reads:
  E:\\CVD data\\Models\\production\\direction_model.ubj
  E:\\CVD data\\Models\\production\\hold_model.ubj
  E:\\CVD data\\Models\\production\\split_info.json   ← N_TEST, test_days
  E:\\CVD data\\Data\\Tick\\clean_features.csv

The number of test days is determined automatically from split_info.json,
which reflects how many .gz files you placed in the Test\\ folder in step1.
No manual config changes needed when you add more test data.

EQUITY MODEL:
  Starting equity : $100,000
  Ratio-based sizing — units auto-scale as equity grows:
    units = max(1, floor(equity * BAND_PCT / entry_price))
  No MAX_UNITS hardcap — position size grows with equity forever.

CONCURRENT ENGINE:
  All symbols processed tick-by-tick in true timestamp order.
  Multiple positions can be open simultaneously across symbols.
  Shared capital pool — MAX_DEPLOYED_PCT=90% of equity at peak.

EOD ENFORCEMENT:
  All positions closed at end of each day. No overnight carry.

Outputs: E:\\CVD data\\Models\\production\\
  sim_trades.csv           all test days combined
  sim_trades_day{N}.csv    one per test day
  equity_curve.csv         day-by-day equity tracker
  model_report.txt

Usage:
  python step5_simulate.py
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
INPUT_FILE  = FEATURES_FILE
MODEL_DIR   = MODEL_DIR
OUTPUT_DIR  = MODEL_DIR

BASE_EQUITY      = 100_000.0

# ---- Ratio-based position sizing ----
BAND4_PCT        = 0.030   # Band4_strong   — 3.0% of equity per trade
BAND3_PCT        = 0.050   # Band3_moderate — 5.0% of equity per trade
BAND2_PCT        = 0.025   # Band2_base     — 2.5% of equity per trade
MAX_POSITION_PCT = 0.10    # hard ceiling: no single position > 10% of equity
MAX_DEPLOYED_PCT = 0.90    # target ~90% of equity deployed at peak
# Min-hold set per-band below (bootstrap max-PF optimised)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# CONFIDENCE BANDS
# =========================================================
# tp_pct = 0.05% of price — matches THRESHOLD_PCT in step3 (both 0.05%).
# sl_pct = 1.0% flat — intentional asymmetric risk/reward.
# NEVER change tp_pct independently of step3 THRESHOLD_PCT.
CONFIDENCE_BANDS = [
    {"name": "Band4_strong",   "min_conf": 0.65, "pos_pct": BAND4_PCT,
     "sl_pct": 1.0, "tp_pct": 0.05, "min_hold": 5,
     "pressure_thresh_long": 0.3, "pressure_thresh_short": 0.5},
    {"name": "Band3_moderate", "min_conf": 0.60, "pos_pct": BAND3_PCT,
     "sl_pct": 1.0, "tp_pct": 0.05, "min_hold": 1,
     "pressure_thresh_long": 0.20, "pressure_thresh_short": 0.10},
    {"name": "Band2_base",     "min_conf": 0.575, "pos_pct": BAND2_PCT,
     "sl_pct": 1.0, "tp_pct": 0.05, "min_hold": 1,
     "pressure_thresh_long": 0.10, "pressure_thresh_short": 0.10},
]
BAND_INFO             = {b["name"]: b for b in CONFIDENCE_BANDS}
PRESSURE_THRESH_LONG  = {b["name"]: b["pressure_thresh_long"]  for b in CONFIDENCE_BANDS}
PRESSURE_THRESH_SHORT = {b["name"]: b["pressure_thresh_short"] for b in CONFIDENCE_BANDS}

label_map     = {-1: 0, 0: 1, 1: 2}
label_inv_map = {0: -1, 1: 0, 2: 1}


# =========================================================
# BAND ROUTING
# =========================================================
def get_band(confidence, pressure_roc, imbalance, direction=0):
    # Noise filter: both flat
    if abs(pressure_roc) < 0.05 and abs(imbalance) < 0.05:
        return 0.0, "no_signal"
    if confidence >= 0.70:
        return 0.0, "ignored_b5"
    if confidence >= 0.675 and imbalance < 0:
        return 0.0, "ignored_hc_short"
    for band in CONFIDENCE_BANDS:
        if confidence >= band["min_conf"]:
            # Entry gate: block if pressure already past exit threshold
            # Long exit: roc < -plong  |  Short exit: roc > +pshort
            if direction ==  1 and pressure_roc < -band["pressure_thresh_long"]:
                return 0.0, "no_signal"
            if direction == -1 and pressure_roc >  band["pressure_thresh_short"]:
                return 0.0, "no_signal"
            return band["pos_pct"], band["name"]
    return 0.0, "below_threshold"


# =========================================================
# SIMULATION — GENUINE REAL-TIME CONCURRENT ENGINE
# =========================================================
def simulate_day(df_day, pred_dir, pred_conf, arr_rocs, arr_imbs,
                 extra_list, equity=BASE_EQUITY):
    """
    Processes every tick across ALL symbols in true timestamp order.
    Fills at tick i+1 (next tick after signal) — not at signal tick.
    Multiple positions open simultaneously via shared capital pool.
    Per-symbol tick counter ensures correct hold-time measurement.
    """
    n        = len(df_day)
    prices   = df_day["trade_price"].values
    symbols  = df_day["symbol"].values
    orig_idx = df_day.index.values

    _A, _B, _C  = 0.1763, 0.000363, 0.03
    sym_counts  = df_day.groupby("symbol").size().to_dict()

    open_positions  = {}
    sym_tick_counts = {}
    deployed        = 0.0
    max_pool        = equity * MAX_DEPLOYED_PCT
    results         = []

    def close_position(sym, exit_price, exit_reason, sym_tick_at_exit):
        nonlocal deployed
        pos      = open_positions.pop(sym)
        cost     = pos["entry_price"] * pos["units"]
        deployed = max(0.0, deployed - cost)
        pnl      = (exit_price - pos["entry_price"]) * pos["direction"] * pos["units"]
        results.append({
            "symbol":         sym,
            "date_tag":       pos["date_tag"],
            "entry_idx":      int(pos["entry_orig_idx"]),
            "direction":      int(pos["direction"]),
            "entry_price":    float(pos["entry_price"]),
            "exit_price":     float(exit_price),
            "bars_held":      int(sym_tick_at_exit - pos["entry_sym_tick"]),
            "units":          int(pos["units"]),
            "desired_dollar": float(pos["desired_dollar"]),
            "tier":           pos["tier"],
            "sl_pct":         pos["sl_pct"],
            "tp_pct":         pos["tp_pct"],
            "exit_reason":    exit_reason,
            "pnl":            float(pnl),
            "confidence":     float(pos["confidence"]),
            "equity_scalar":  round(equity / BASE_EQUITY, 4),
            **pos["extra"],
        })

    for i in range(n):
        sym   = symbols[i]
        price = prices[i]

        sym_tick_counts[sym] = sym_tick_counts.get(sym, 0) + 1
        sym_tick = sym_tick_counts[sym]

        # ---- Step 1: manage open position ----
        if sym in open_positions:
            pos        = open_positions[sym]
            entry      = pos["entry_price"]
            direction  = pos["direction"]
            sl_dollar  = pos["sl_dollar"]
            tp_dollar  = pos["tp_dollar"]
            mv         = (price - entry) * direction
            ticks_held = sym_tick - pos["entry_sym_tick"]

            if mv >= tp_dollar:
                close_position(sym, entry + tp_dollar * direction, "take_profit", sym_tick)
            elif mv <= -sl_dollar:
                close_position(sym, entry - sl_dollar * direction, "stop_loss", sym_tick)
            else:
                p_tl = PRESSURE_THRESH_LONG[pos["tier"]]
                p_ts = PRESSURE_THRESH_SHORT[pos["tier"]]
                triggered = (
                    ticks_held >= BAND_INFO[pos["tier"]]["min_hold"] and (
                        (direction ==  1 and arr_rocs[i] < -p_tl) or
                        (direction == -1 and arr_rocs[i] >  p_ts)
                    )
                )
                if triggered and price != entry:
                    # price != entry: eliminates phantom exits at zero PnL
                    raw_exit = price
                    if (raw_exit - entry) * direction < -sl_dollar:
                        raw_exit = entry - sl_dollar * direction
                    close_position(sym, raw_exit, "pressure_exit", sym_tick)
                elif ticks_held >= pos["max_hold_bars"]:
                    mv2 = (price - entry) * direction
                    if mv2 >= tp_dollar:
                        close_position(sym, entry + tp_dollar * direction, "take_profit", sym_tick)
                    elif mv2 <= -sl_dollar:
                        close_position(sym, entry - sl_dollar * direction, "stop_loss", sym_tick)
                    else:
                        close_position(sym, price, "eod_close", sym_tick)

        # ---- Step 2: consider new entry ----
        if sym in open_positions:
            continue
        if pred_dir[i] == 0 or pred_dir[i] == -1 or price <= 0:
            continue

        pos_pct, tier = get_band(pred_conf[i], arr_rocs[i], arr_imbs[i], pred_dir[i])
        if pos_pct == 0.0:
            continue

        # Fill at next tick — never at the triggering tick's price
        sym_future = np.where((symbols == sym) & (np.arange(n) > i))[0]
        if len(sym_future) == 0:
            continue
        fill_idx   = sym_future[0]
        fill_price = prices[fill_idx]
        if fill_price <= 0:
            continue

        dollar_target = min(equity * pos_pct, equity * MAX_POSITION_PCT)
        desired_units = max(1, int(dollar_target / fill_price))
        available     = max(0.0, max_pool - deployed)
        units_by_pool = max(1, int(available / fill_price))
        units         = min(desired_units, units_by_pool)

        binfo      = BAND_INFO[tier]
        sym_n      = sym_counts.get(sym, n)
        max_hold_b = max(10, int(sym_n * (_A * np.exp(-_B * sym_n) + _C)))

        deployed += fill_price * units
        open_positions[sym] = {
            "entry_price":    fill_price,
            "entry_sym_tick": sym_tick,
            "entry_orig_idx": orig_idx[fill_idx],
            "direction":      pred_dir[i],
            "units":          units,
            "desired_dollar": dollar_target,
            "tier":           tier,
            "sl_pct":         binfo["sl_pct"],
            "tp_pct":         binfo["tp_pct"],
            "sl_dollar":      fill_price * binfo["sl_pct"] / 100.0,
            "tp_dollar":      fill_price * binfo["tp_pct"] / 100.0,
            "max_hold_bars":  max_hold_b,
            "confidence":     pred_conf[i],
            "date_tag":       df_day["date_tag"].iloc[i],
            "extra":          extra_list[i],
        }

    # ---- EOD: close all remaining ----
    for sym in list(open_positions.keys()):
        pos        = open_positions[sym]
        final_tick = sym_tick_counts.get(sym, 0)
        sym_rows   = np.where(symbols == sym)[0]
        eod_price  = prices[sym_rows[-1]] if len(sym_rows) else prices[-1]
        entry      = pos["entry_price"]
        direction  = pos["direction"]
        mv         = (eod_price - entry) * direction
        if mv >= pos["tp_dollar"]:
            close_position(sym, entry + pos["tp_dollar"] * direction, "take_profit", final_tick)
        elif mv <= -pos["sl_dollar"]:
            close_position(sym, entry - pos["sl_dollar"] * direction, "stop_loss", final_tick)
        else:
            close_position(sym, eod_price, "eod_close", final_tick)

    return pd.DataFrame(results) if results else pd.DataFrame()


# =========================================================
# MAIN
# =========================================================
def run():
    # ---- Load split info from step4 ----
    split_info_path = os.path.join(MODEL_DIR, "split_info.json")
    if not os.path.exists(split_info_path):
        raise FileNotFoundError(
            f"{split_info_path} not found.\n"
            f"Run step4_train_model.py first."
        )
    with open(split_info_path) as f:
        split_info = json.load(f)

    test_days  = split_info["test_days"]
    train_days = split_info["train_days"]
    val_days   = split_info["val_days"]
    n_test     = split_info["n_test"]
    acc_va     = split_info["val_acc"]
    fc         = split_info["features"]

    print("Loading feature matrix...")
    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values(["date_tag", "timestamp_ns"]).reset_index(drop=True)

    print(f"  {len(df):,} rows")
    print(f"  Train      : {train_days}")
    print(f"  Validation : {val_days}")
    print(f"  Test       : {test_days}  ({n_test} unseen days)")

    # ---- Load models ----
    dir_mdl  = xgb.Booster()
    dir_mdl.load_model(os.path.join(MODEL_DIR, "direction_model.ubj"))
    hold_mdl = xgb.Booster()
    hold_mdl.load_model(os.path.join(MODEL_DIR, "hold_model.ubj"))
    print(f"\nModels loaded.  Val acc: {acc_va:.1%}")

    # ---- Walk-forward simulation ----
    SEP = "=" * 62
    print(f"\n{SEP}")
    print(f"WALK-FORWARD — {n_test} test day(s)  |  Start equity: ${BASE_EQUITY:,.2f}")
    print(f"  Sizing: B4={BAND4_PCT:.1%}  B3={BAND3_PCT:.1%}  B2={BAND2_PCT:.1%}  "
          f"max_pos={MAX_POSITION_PCT:.0%}  max_deployed={MAX_DEPLOYED_PCT:.0%}")
    print(SEP + "\n")

    extra_cols = [
        "cvd_vel_3", "cvd_vel_5", "cvd_accel",
        "pressure_3", "pressure_5", "pressure_roc_10",
        "l5_weighted_imbalance", "l5_imbalance",
        "depth_l5_weighted_norm", "bid_depth_l5_weighted_norm",
    ]

    equity      = BASE_EQUITY
    all_sim_dfs = []
    equity_log  = []

    for day_num, test_day in enumerate(test_days, 1):
        df_day = df[df["date_tag"] == test_day].copy().reset_index(drop=True)
        if len(df_day) == 0:
            print(f"  Day {day_num} ({test_day}): NO DATA — skipping")
            continue

        X_te   = df_day[fc].values.astype(np.float32)
        dtest  = xgb.DMatrix(X_te, feature_names=fc)
        y_te   = df_day["target_direction"].map(label_map).values.astype(int)

        probs     = dir_mdl.predict(dtest).reshape(len(X_te), -1)
        pred_enc  = probs.argmax(axis=1)
        pred_conf = probs.max(axis=1)
        pred_dir  = np.array([label_inv_map[c] for c in pred_enc])
        acc       = (pred_enc == y_te).mean()

        arr_rocs = df_day.get("pressure_roc_10",
                              pd.Series(0, index=df_day.index)).fillna(0).values
        arr_imbs = df_day.get("l5_weighted_imbalance",
                              pd.Series(0, index=df_day.index)).fillna(0).values
        extra_list = [
            {c: float(df_day.at[idx, c]) if c in df_day.columns else 0.0
             for c in extra_cols}
            for idx in df_day.index
        ]

        equity_scalar = equity / BASE_EQUITY
        sim_df = simulate_day(df_day, pred_dir, pred_conf,
                              arr_rocs, arr_imbs, extra_list,
                              equity=equity)

        if len(sim_df) == 0:
            day_pnl = 0.0
            print(f"  Day {day_num} ({test_day}): NO TRADES")
        else:
            day_pnl = sim_df["pnl"].sum()
            gw = sim_df[sim_df["pnl"] > 0]["pnl"].sum()
            gl = sim_df[sim_df["pnl"] < 0]["pnl"].abs().sum()
            print(f"  Day {day_num} ({test_day}):  "
                  f"scalar={equity_scalar:.3f}  n={len(sim_df):,}  "
                  f"PnL=${day_pnl:+,.2f}  WR={(sim_df['pnl']>0).mean():.1%}  "
                  f"PF={gw/(gl+1e-8):.3f}  acc={acc:.1%}")
            for b in CONFIDENCE_BANDS:
                t = sim_df[sim_df["tier"] == b["name"]]
                if len(t) > 0:
                    print(f"    {b['name']:22}: n={len(t):5,}  "
                          f"WR={(t['pnl']>0).mean():.1%}  "
                          f"PnL=${t['pnl'].sum():+,.2f}")
            sim_df.to_csv(
                os.path.join(OUTPUT_DIR, f"sim_trades_day{day_num}.csv"),
                index=False)
            all_sim_dfs.append(sim_df)

        prev_equity = equity
        equity     += day_pnl
        pct         = day_pnl / prev_equity * 100

        equity_log.append({
            "day_num":       day_num,
            "date_tag":      test_day,
            "equity_start":  round(prev_equity, 2),
            "equity_scalar": round(equity_scalar, 4),
            "n_trades":      len(sim_df) if len(sim_df) > 0 else 0,
            "day_pnl":       round(day_pnl, 2),
            "day_pct":       round(pct, 4),
            "equity_end":    round(equity, 2),
            "dir_acc":       round(acc, 4),
        })
        print(f"    → ${prev_equity:,.2f}  {day_pnl:+,.2f}  =  ${equity:,.2f}  ({pct:+.2f}%)\n")

    net_profit = equity - BASE_EQUITY
    net_pct    = net_profit / BASE_EQUITY * 100
    print(SEP)
    print(f"  Start: ${BASE_EQUITY:,.2f}   Final: ${equity:,.2f}   "
          f"Net: ${net_profit:+,.2f}  ({net_pct:+.2f}%)")
    print(SEP)

    equity_df = pd.DataFrame(equity_log)
    combined  = pd.concat(all_sim_dfs, ignore_index=True) if all_sim_dfs else pd.DataFrame()

    if len(combined) > 0:
        pnl = combined["pnl"]
        gw  = pnl[pnl > 0].sum()
        gl  = pnl[pnl < 0].abs().sum()
        eq  = pd.Series(pnl.cumsum())
        dd  = float((eq - eq.cummax()).min())
        print(f"\nCOMBINED ({n_test} test day(s)):")
        print(f"  Trades={len(combined):,}  PnL=${pnl.sum():,.2f}  "
              f"WR={(pnl>0).mean():.1%}  PF={gw/(gl+1e-8):.3f}  "
              f"Sh={pnl.mean()/(pnl.std()+1e-8):.4f}  DD=${dd:,.2f}")
    else:
        dd = 0.0

    # ---- Save outputs ----
    equity_df.to_csv(os.path.join(OUTPUT_DIR, "equity_curve.csv"), index=False)
    if len(combined) > 0:
        combined.to_csv(os.path.join(OUTPUT_DIR, "sim_trades.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "model_report.txt"), "w", encoding="utf-8") as f:
        f.write("CVD PRODUCTION MODEL REPORT\n" + "="*55 + "\n\n")
        f.write(f"Input         : {INPUT_FILE}\n")
        f.write(f"Train days    : {train_days}\n")
        f.write(f"Val days      : {val_days}\n")
        f.write(f"Test days     : {test_days}\n")
        f.write(f"Dir trees     : {split_info['n_trees']}\n")
        f.write(f"Val acc       : {acc_va:.4f}\n\n")
        f.write(f"BASE EQUITY   : {BASE_EQUITY:.2f}\n")
        f.write(f"FINAL EQUITY  : {equity:.2f}\n")
        f.write(f"NET PROFIT    : {net_profit:.2f}\n")
        f.write(f"NET PCT       : {net_pct:.4f}%\n\n")
        f.write(f"SIZING MODEL  : ratio-based  B4={BAND4_PCT:.1%}  "
                f"B3={BAND3_PCT:.1%}  B2={BAND2_PCT:.1%}  "
                f"max={MAX_POSITION_PCT:.0%}/trade\n")
        f.write(f"EOD close     : ON\n")
        f.write(f"Concurrent    : ON (max_deployed={MAX_DEPLOYED_PCT:.0%})\n")
        f.write(f"Fill          : next-tick (i+1)\n\n")
        if len(combined) > 0:
            f.write(f"COMBINED TEST PERFORMANCE\n" + "-"*55 + "\n")
            f.write(f"  Trades        : {len(combined)}\n")
            f.write(f"  Total PnL     : {combined['pnl'].sum():.2f}\n")
            f.write(f"  Win rate      : {(combined['pnl']>0).mean():.4f}\n")
            f.write(f"  Profit factor : {gw/(gl+1e-8):.4f}\n")
            f.write(f"  Sharpe        : {combined['pnl'].mean()/(combined['pnl'].std()+1e-8):.4f}\n")
            f.write(f"  Max drawdown  : {dd:.2f}\n\n")
        f.write(f"DAY-BY-DAY EQUITY\n" + "-"*55 + "\n")
        f.write(f"  {'Day':>3}  {'Date':12}  {'Start':>12}  {'Scalar':>6}  "
                f"{'n':>6}  {'PnL':>10}  {'%':>7}  {'End':>12}  {'Acc':>5}\n")
        f.write(f"  {'-'*80}\n")
        for _, r in equity_df.iterrows():
            f.write(f"  {int(r.day_num):>3}  {r.date_tag:12}  "
                    f"${r.equity_start:>11,.2f}  {r.equity_scalar:>6.3f}  "
                    f"{int(r.n_trades):>6,}  ${r.day_pnl:>+9,.2f}  "
                    f"{r.day_pct:>+6.2f}%  ${r.equity_end:>11,.2f}  "
                    f"{r.dir_acc:>5.1%}\n")
        f.write(f"\nPER-BAND THRESHOLDS\n" + "-"*55 + "\n")
        for b in CONFIDENCE_BANDS:
            f.write(f"  {b['name']:22}: SL={b['sl_pct']}%  TP={b['tp_pct']}%  "
                    f"pos_pct={b['pos_pct']:.1%}\n")

    print(f"\nStep 5 complete → {OUTPUT_DIR}")
    print(f"  sim_trades.csv  sim_trades_day1-{n_test}.csv")
    print(f"  equity_curve.csv  model_report.txt")
    print(f"\nRun step6_model_analysis.py for full analysis.")

    return {
        "test_days": test_days, "n_test": n_test,
        "base_equity": BASE_EQUITY, "final_equity": equity,
        "net_profit": net_profit, "net_pct": net_pct,
        "equity_log": equity_log,
        "total_pnl": combined["pnl"].sum() if len(combined) > 0 else 0,
        "win_rate":  (combined["pnl"] > 0).mean() if len(combined) > 0 else 0,
        "profit_factor": gw / (gl + 1e-8) if len(combined) > 0 else 0,
        "sharpe": combined["pnl"].mean() / (combined["pnl"].std() + 1e-8) if len(combined) > 0 else 0,
        "n_trades": len(combined),
    }


if __name__ == "__main__":
    run()