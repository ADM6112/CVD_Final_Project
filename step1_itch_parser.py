"""
Step 1 — ITCH Parser
====================
Parses raw NASDAQ TotalView-ITCH 5.0 .gz files into trade CSVs
with full Level 5 order book depth snapshots at each trade.

GZ FILE ORGANISATION:
  Place .gz files into subfolders under the base ITCH directory:

    Tick GZ primary data files\\
      Train\\       ← training day(s) — minimum 1
      Validation\\  ← validation / early-stopping day(s) — minimum 1
      Test\\        ← unseen test days — minimum 1, drives simulation count

  The pipeline auto-detects how many files are in each folder and
  adjusts N_TRAIN / N_VAL / N_TEST in step4 + step5 accordingly.
  You control the split entirely by where you put the files — no
  config changes needed when you add more data.

Handles both filename formats:
  Old format : S071321-v50.txt.gz    → date tag S071321
  New format : 01302019.NASDAQ_ITCH50.gz → date tag S013019

Output per file:
  E:\\CVD data\\Data\\Tick\\raw_csv\\<SPLIT>_<DATE_TAG>.csv
  e.g.  train_S081321.csv
        val_S083019.csv
        test_S101819.csv

Usage (called from run_pipeline.py or standalone):
  python step1_itch_parser.py
"""

import csv
import gzip
import re
import struct
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count

# =========================================================
# USER SETTINGS
# =========================================================
from config import GZ_BASE_DIR, RAW_CSV_DIR
BASE_ITCH_DIR = GZ_BASE_DIR
OUTPUT_DIR    = RAW_CSV_DIR

RANDOM_SEED       = 42
NUM_RANDOM_STOCKS = 20

# Subfolder names — must match what you created on disk
SPLIT_FOLDERS = {
    "train":      "Train",
    "validation": "Validation",
    "test":       "Test",
}

# Curated high-volume NASDAQ universe — 30 stocks selected for:
#   - Deep order books (high tick count, tight spreads)
#   - Pure NASDAQ listing (present in ITCH files)
#   - Price > $10 (liquidity gate, matches MIN_STOCK_PRICE in step3/step5)
#   - Consistent intraday order flow signal for CVD/pressure features
# Do NOT add low-volume or NYSE-primary stocks — they won't appear in ITCH data.
TARGET_STOCKS = {
    # Mega-cap tech — highest tick counts, deepest books
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "META", "INTC", "CSCO", "NFLX",
    # High-volume NASDAQ growth
    "AMD", "PYPL", "ADBE", "CMCSA", "COST", "SBUX", "QCOM", "TXN", "AVGO", "MU",
    # Mid-large NASDAQ with consistent volume
    "AMAT", "LRCX", "KLAC", "MCHP", "SNPS", "CDNS", "IDXX", "ILMN", "BIIB", "REGN",
}


# =========================================================
# DATE TAG EXTRACTION
# =========================================================
def extract_date_tag(filename: str) -> str:
    m = re.search(r'(S\d{6})', filename)
    if m:
        return m.group(1)
    m = re.search(r'(\d{8})\.NASDAQ', filename)
    if m:
        d = m.group(1)
        return f"S{d[0:2]}{d[2:4]}{d[6:8]}"
    return "SUNKNOWN"


# =========================================================
# HELPERS
# =========================================================
def parse_ts_6(b):  return int.from_bytes(b, byteorder="big", signed=False)
def decode_alpha(b): return bytes(b).decode("ascii", errors="ignore").strip()
def ns_to_time_str(ns):
    s, nano = divmod(ns, 1_000_000_000)
    h, rem  = divmod(s, 3600)
    m, sec  = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}.{nano:09d}"
def safe_div(n, d): return None if d == 0 else n / d

def top_n_bids(book, n=5):
    lvls = sorted([(px, sz) for px, sz in book.items() if sz > 0],
                  key=lambda x: x[0], reverse=True)
    return lvls[:n]

def top_n_asks(book, n=5):
    lvls = sorted([(px, sz) for px, sz in book.items() if sz > 0],
                  key=lambda x: x[0])
    return lvls[:n]


# =========================================================
# PASS 1 — READ STOCK DIRECTORY
# =========================================================
def get_all_symbols(itch_path):
    symbols = set()
    unpack_H = struct.Struct(">H").unpack
    with gzip.open(itch_path, "rb") as f:
        while True:
            lb = f.read(2)
            if not lb: break
            msg_len = unpack_H(lb)[0]
            msg = f.read(msg_len)
            if len(msg) != msg_len: break
            mv = memoryview(msg)
            if mv[0:1] == b"R":
                symbols.add(decode_alpha(mv[11:19]))
    return symbols


def choose_symbols(all_symbols):
    """Return the intersection of TARGET_STOCKS and what's actually in this ITCH file."""
    matched = TARGET_STOCKS & set(all_symbols)
    missing = TARGET_STOCKS - set(all_symbols)
    print(f"  Symbols matched: {len(matched)} / {len(TARGET_STOCKS)}")
    if missing:
        print(f"  Not in file:     {sorted(missing)}")
    return matched


# =========================================================
# ORDER BOOK
# =========================================================
def make_book():
    return {"orders": {}, "bids": defaultdict(int), "asks": defaultdict(int)}

def book_add(bk, order_ref, symbol, side, shares, price, target):
    bk["orders"][order_ref] = {"symbol": symbol, "side": side, "price": price, "shares": shares}
    if symbol in target:
        (bk["bids"] if side == "B" else bk["asks"])[price] += shares

def book_reduce(bk, order_ref, remove, target):
    o = bk["orders"].get(order_ref)
    if not o: return None
    amt = min(remove, o["shares"])
    if o["symbol"] in target:
        side_book = bk["bids"] if o["side"] == "B" else bk["asks"]
        side_book[o["price"]] -= amt
        if side_book[o["price"]] <= 0:
            side_book.pop(o["price"], None)
    o["shares"] -= amt
    if o["shares"] <= 0:
        bk["orders"].pop(order_ref)
    return o

def book_delete(bk, order_ref, target):
    o = bk["orders"].pop(order_ref, None)
    if not o: return
    if o["symbol"] in target:
        side_book = bk["bids"] if o["side"] == "B" else bk["asks"]
        side_book[o["price"]] -= o["shares"]
        if side_book[o["price"]] <= 0:
            side_book.pop(o["price"], None)

def book_replace(bk, old_ref, new_ref, new_shares, new_price, target):
    o = bk["orders"].pop(old_ref, None)
    if not o: return
    if o["symbol"] in target:
        side_book = bk["bids"] if o["side"] == "B" else bk["asks"]
        side_book[o["price"]] -= o["shares"]
        if side_book[o["price"]] <= 0:
            side_book.pop(o["price"], None)
    book_add(bk, new_ref, o["symbol"], o["side"], new_shares, new_price, target)


# =========================================================
# LOB DEPTH SNAPSHOT
# =========================================================
def get_depth(bk, symbol):
    bids_l = top_n_bids(bk["bids"], 5)
    asks_l = top_n_asks(bk["asks"], 5)
    while len(bids_l) < 5: bids_l.append((None, 0))
    while len(asks_l) < 5: asks_l.append((None, 0))
    bid_px = [px / 10000.0 if px else None for px, _ in bids_l]
    bid_sz = [sz for _, sz in bids_l]
    ask_px = [px / 10000.0 if px else None for px, _ in asks_l]
    ask_sz = [sz for _, sz in asks_l]
    w = (1.0, 0.5, 1/3, 0.25, 0.2)
    bid_w = sum(w[i] * bid_sz[i] for i in range(5))
    ask_w = sum(w[i] * ask_sz[i] for i in range(5))
    return {
        "bid_px": bid_px, "bid_sz": bid_sz,
        "ask_px": ask_px, "ask_sz": ask_sz,
        "depth_l1_avg":          (bid_sz[0] + ask_sz[0]) / 2.0,
        "bid_depth_l5":          sum(bid_sz),
        "ask_depth_l5":          sum(ask_sz),
        "depth_l5_total":        sum(bid_sz) + sum(ask_sz),
        "depth_l5_avg":          (sum(bid_sz) + sum(ask_sz)) / 10.0,
        "bid_depth_l5_weighted": bid_w,
        "ask_depth_l5_weighted": ask_w,
        "depth_l5_weighted":     bid_w + ask_w,
        "l1_imbalance":          safe_div(bid_sz[0] - ask_sz[0], bid_sz[0] + ask_sz[0]),
        "l5_imbalance":          safe_div(sum(bid_sz) - sum(ask_sz), sum(bid_sz) + sum(ask_sz)),
        "l5_weighted_imbalance": safe_div(bid_w - ask_w, bid_w + ask_w),
    }


HEADER = [
    "timestamp_ns", "time_str", "symbol", "msg_type",
    "trade_price", "trade_shares", "match_number",
    "aggressor_side", "stock_locate", "order_reference",
    "bid_px_1", "bid_sz_1", "bid_px_2", "bid_sz_2",
    "bid_px_3", "bid_sz_3", "bid_px_4", "bid_sz_4",
    "bid_px_5", "bid_sz_5",
    "ask_px_1", "ask_sz_1", "ask_px_2", "ask_sz_2",
    "ask_px_3", "ask_sz_3", "ask_px_4", "ask_sz_4",
    "ask_px_5", "ask_sz_5",
    "depth_l1_avg", "bid_depth_l5", "ask_depth_l5",
    "depth_l5_total", "depth_l5_avg",
    "bid_depth_l5_weighted", "ask_depth_l5_weighted",
    "depth_l5_weighted",
    "l1_imbalance", "l5_imbalance", "l5_weighted_imbalance",
]


def write_row(writer, ts, symbol, msg_type, price, shares, match, side, locate, ref, depth):
    d = depth
    writer.writerow([
        ts, ns_to_time_str(ts), symbol, msg_type,
        price, shares, match, side, locate, ref,
        d["bid_px"][0], d["bid_sz"][0], d["bid_px"][1], d["bid_sz"][1],
        d["bid_px"][2], d["bid_sz"][2], d["bid_px"][3], d["bid_sz"][3],
        d["bid_px"][4], d["bid_sz"][4],
        d["ask_px"][0], d["ask_sz"][0], d["ask_px"][1], d["ask_sz"][1],
        d["ask_px"][2], d["ask_sz"][2], d["ask_px"][3], d["ask_sz"][3],
        d["ask_px"][4], d["ask_sz"][4],
        d["depth_l1_avg"], d["bid_depth_l5"], d["ask_depth_l5"],
        d["depth_l5_total"], d["depth_l5_avg"],
        d["bid_depth_l5_weighted"], d["ask_depth_l5_weighted"],
        d["depth_l5_weighted"],
        d["l1_imbalance"], d["l5_imbalance"], d["l5_weighted_imbalance"],
    ])


# =========================================================
# PASS 2 — RECONSTRUCT LOB + CAPTURE TRADES
# =========================================================
def parse_itch_file(itch_path: str, out_csv: str, target_symbols: set):
    """
    Parse one ITCH gz file into a trade CSV.

    Performance optimisations vs original:
      1. order_to_sym dict — O(1) order lookup instead of scanning all 35 books.
         Every Add message registers ref → symbol. Cancel/Delete/Execute look
         up the symbol instantly. This is the single biggest speedup.
      2. Large gzip read buffer (64 MB) — reduces decompression overhead.
      3. Large CSV write buffer (4 MB) — fewer write syscalls.
    """
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    books        = {sym: make_book() for sym in target_symbols}
    order_to_sym = {}   # order_ref (int) → symbol (str)  — O(1) lookup

    unpack_H = struct.Struct(">H").unpack
    unpack_I = struct.Struct(">I").unpack
    unpack_Q = struct.Struct(">Q").unpack
    msg_count = trade_rows = 0

    with gzip.open(itch_path, "rb") as f, \
         open(out_csv, "w", newline="", encoding="utf-8", buffering=1 << 22) as out_f:
        writer = csv.writer(out_f)
        writer.writerow(HEADER)
        read = f.read   # local alias avoids attribute lookup in tight loop

        while True:
            lb = read(2)
            if not lb: break
            msg_len = unpack_H(lb)[0]
            msg = read(msg_len)
            if len(msg) != msg_len: break
            msg_count += 1
            mv    = memoryview(msg)
            mtype = mv[0:1]

            if mtype == b"R":
                continue

            elif mtype in (b"A", b"F"):
                # Add Order — register in book AND in order_to_sym
                ref    = unpack_Q(mv[11:19])[0]
                side   = decode_alpha(mv[19:20])
                shares = unpack_I(mv[20:24])[0]
                symbol = decode_alpha(mv[24:32])
                price  = unpack_I(mv[32:36])[0]
                if symbol in target_symbols:
                    book_add(books[symbol], ref, symbol, side, shares, price, target_symbols)
                    order_to_sym[ref] = symbol   # O(1) registration

            elif mtype == b"X":
                # Order Cancel — partial cancel, look up book directly
                ref    = unpack_Q(mv[11:19])[0]
                cancel = unpack_I(mv[19:23])[0]
                sym    = order_to_sym.get(ref)
                if sym:
                    book_reduce(books[sym], ref, cancel, target_symbols)

            elif mtype == b"D":
                # Order Delete — full delete
                ref = unpack_Q(mv[11:19])[0]
                sym = order_to_sym.pop(ref, None)   # remove from lookup
                if sym:
                    book_delete(books[sym], ref, target_symbols)

            elif mtype == b"U":
                # Order Replace — old ref gone, new ref registered
                old_ref    = unpack_Q(mv[11:19])[0]
                new_ref    = unpack_Q(mv[19:27])[0]
                new_shares = unpack_I(mv[27:31])[0]
                new_price  = unpack_I(mv[31:35])[0]
                sym        = order_to_sym.pop(old_ref, None)
                if sym:
                    book_replace(books[sym], old_ref, new_ref, new_shares, new_price, target_symbols)
                    order_to_sym[new_ref] = sym   # register replacement

            elif mtype == b"E":
                # Order Executed — trade event, emit row
                ts     = parse_ts_6(mv[5:11])
                ref    = unpack_Q(mv[11:19])[0]
                shares = unpack_I(mv[19:23])[0]
                match  = unpack_Q(mv[23:31])[0]
                sym    = order_to_sym.get(ref)
                if sym:
                    bk = books[sym]
                    o  = book_reduce(bk, ref, shares, target_symbols)
                    if o:
                        write_row(writer, ts, sym, "E", o["price"] / 10000.0, shares, match,
                                  "B" if o["side"] == "S" else "S", 0, ref, get_depth(bk, sym))
                        trade_rows += 1

            elif mtype == b"C":
                # Order Executed with Price (non-standard)
                locate = unpack_H(mv[1:3])[0]
                ts     = parse_ts_6(mv[5:11])
                ref    = unpack_Q(mv[11:19])[0]
                shares = unpack_I(mv[19:23])[0]
                match  = unpack_Q(mv[23:31])[0]
                price  = unpack_I(mv[32:36])[0]
                sym    = order_to_sym.get(ref)
                if sym:
                    bk = books[sym]
                    o  = book_reduce(bk, ref, shares, target_symbols)
                    if o:
                        write_row(writer, ts, sym, "C", price / 10000.0, shares, match,
                                  "B" if o["side"] == "S" else "S", locate, ref, get_depth(bk, sym))
                        trade_rows += 1

            elif mtype == b"P":
                # Non-cross Trade — direct trade print, no book reference
                ts     = parse_ts_6(mv[5:11])
                ref    = unpack_Q(mv[11:19])[0]
                side   = decode_alpha(mv[19:20])
                shares = unpack_I(mv[20:24])[0]
                symbol = decode_alpha(mv[24:32])
                price  = unpack_I(mv[32:36])[0]
                match  = unpack_Q(mv[36:44])[0]
                if symbol in target_symbols:
                    write_row(writer, ts, symbol, "P", price / 10000.0, shares, match,
                              side, 0, ref, get_depth(books[symbol], symbol))
                    trade_rows += 1

            elif mtype == b"Q":
                # Cross Trade
                locate = unpack_H(mv[1:3])[0]
                ts     = parse_ts_6(mv[5:11])
                shares = unpack_Q(mv[11:19])[0]
                symbol = decode_alpha(mv[19:27])
                price  = unpack_I(mv[27:31])[0]
                match  = unpack_Q(mv[31:39])[0]
                ctype  = decode_alpha(mv[39:40])
                if symbol in target_symbols:
                    write_row(writer, ts, symbol, "Q", price / 10000.0, shares, match,
                              ctype, locate, "", get_depth(books[symbol], symbol))
                    trade_rows += 1

            if msg_count % 5_000_000 == 0:
                print(f"  {msg_count:,} messages | {trade_rows:,} trade rows")

    print(f"  Done: {msg_count:,} messages → {trade_rows:,} trades → {out_csv}")
    return trade_rows


# =========================================================
# SCAN SPLIT FOLDERS AND DETECT FILE COUNTS
# =========================================================
def scan_split_folders():
    """
    Returns dict: { split_name: [sorted list of gz Paths] }
    Also prints a summary showing how many files are in each folder
    and what N_TRAIN / N_VAL / N_TEST will be.
    """
    base = Path(BASE_ITCH_DIR)
    split_files = {}
    for split_key, folder_name in SPLIT_FOLDERS.items():
        folder = base / folder_name
        if not folder.exists():
            print(f"  WARNING: {folder} does not exist — creating it")
            folder.mkdir(parents=True, exist_ok=True)
        files = sorted(folder.glob("*.gz"))
        split_files[split_key] = files

    n_train = len(split_files["train"])
    n_val   = len(split_files["validation"])
    n_test  = len(split_files["test"])

    print(f"  Train      folder: {n_train} file(s)")
    print(f"  Validation folder: {n_val} file(s)")
    print(f"  Test       folder: {n_test} file(s)")

    if n_train == 0:
        raise FileNotFoundError(
            f"No .gz files found in Train\\ folder.\n"
            f"Place at least 1 ITCH .gz file there and rerun."
        )
    if n_val == 0:
        raise FileNotFoundError(
            f"No .gz files found in Validation\\ folder.\n"
            f"Place at least 1 ITCH .gz file there and rerun."
        )
    if n_test == 0:
        raise FileNotFoundError(
            f"No .gz files found in Test\\ folder.\n"
            f"Place at least 1 ITCH .gz file there and rerun."
        )

    return split_files, n_train, n_val, n_test


# =========================================================
# PARALLEL WORKER — called by multiprocessing.Pool
# =========================================================
def _parse_worker(args):
    """Top-level function so it can be pickled by multiprocessing."""
    gz_path, out_csv, split_key, date_tag = args
    print(f"\n[{split_key}] Parsing {gz_path.name} → {date_tag}")
    print(f"  Pass 1: reading stock directory...")
    all_syms = get_all_symbols(str(gz_path))
    target   = choose_symbols(all_syms)
    print(f"  Pass 2: reconstructing LOB + capturing trades...")
    parse_itch_file(str(gz_path), out_csv, target)
    return (split_key, date_tag, out_csv)


# =========================================================
# PROCESS ALL FILES ACROSS ALL SPLITS
# =========================================================
def run():
    print("Scanning split folders...")
    split_files, n_train, n_val, n_test = scan_split_folders()
    print(f"  → N_TRAIN={n_train}  N_VAL={n_val}  N_TEST={n_test}\n")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Build work list — skip already-parsed files
    work    = []
    skipped = []
    for split_key, gz_paths in split_files.items():
        for gz_path in gz_paths:
            date_tag = extract_date_tag(gz_path.name)
            out_csv  = str(Path(OUTPUT_DIR) / f"{split_key}_{date_tag}.csv")
            if Path(out_csv).exists():
                print(f"  [{split_key}] {gz_path.name} → already parsed, skipping")
                skipped.append((split_key, date_tag, out_csv))
            else:
                work.append((gz_path, out_csv, split_key, date_tag))

    results = list(skipped)

    if work:
        # Use up to min(n_files, cpu_count) workers — each file is independent
        n_workers = min(len(work), cpu_count())
        print(f"\nParsing {len(work)} file(s) in parallel using {n_workers} worker(s)...")
        print("(Each file uses one CPU core — total time ≈ slowest single file)\n")

        if n_workers == 1:
            # Single file or single core — run in-process (easier debugging)
            for args in work:
                results.append(_parse_worker(args))
        else:
            with Pool(processes=n_workers) as pool:
                for result in pool.imap_unordered(_parse_worker, work):
                    results.append(result)

    print(f"\n{'='*50}")
    print(f"Step 1 complete — {len(results)} days parsed")
    for split, tag, path in sorted(results):
        print(f"  [{split:10}] {tag} → {Path(path).name}")

    return results, n_train, n_val, n_test


if __name__ == "__main__":
    run()
