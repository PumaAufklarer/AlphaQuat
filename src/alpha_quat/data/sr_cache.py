"""Alpha360 cache builder — pre-compute 6 raw features + SR labels per day.

Output: data/alpha360/YYYYMMDD.parquet — one file per date with columns:
  ts_code, open, high, low, close, volume, vwap,
  resistance_5d_price, resistance_5d_dist,
  resistance_20d_price, resistance_20d_dist,
  resistance_60d_price, resistance_60d_dist,
  support_5d_price, support_5d_dist,
  support_20d_price, support_20d_dist,
  support_60d_price, support_60d_dist
"""

import logging
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

_OUTPUT_DIR = "alpha360"

_FEATURE_COLS = ["open", "high", "low", "close", "volume", "vwap"]
_SR_SUFFIXES = ["5d", "20d", "60d"]
_SR_PRICE_COLS = [
    f"{side}_{s}_price" for side in ("resistance", "support") for s in _SR_SUFFIXES
]
_SR_DIST_COLS = [
    f"{side}_{s}_dist" for side in ("resistance", "support") for s in _SR_SUFFIXES
]

_HORIZONS = [
    ("5d", 2, 5),
    ("20d", 10, 20),
    ("60d", 30, 60),
]


def _load_all_daily(data_dir: Path) -> pd.DataFrame:
    daily_dir = data_dir / "daily"
    files = sorted(daily_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No daily parquet files found in {daily_dir}")

    chunks = []
    for f in files:
        df = pd.read_parquet(f)
        df["trade_date"] = f.stem.replace("_", "")
        chunks.append(df)
    all_df = pd.concat(chunks, ignore_index=True)

    if "volume" not in all_df.columns and "vol" in all_df.columns:
        all_df = all_df.rename(columns={"vol": "volume"})
    all_df["vwap"] = all_df["amount"] / all_df["volume"].replace(0, np.nan)
    return all_df


def _nearest_peak_and_dist(
    prices: np.ndarray,
    is_verified: np.ndarray,
    lookahead: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each day d, find nearest verified peak at > d within lookahead.

    Returns (price_array, distance_in_days_array).
    """
    T = len(prices)
    price_out = np.full(T, np.nan)
    dist_out = np.full(T, np.nan)
    queue: deque[tuple[int, float]] = deque()

    for d in range(T - 1, -1, -1):
        while queue and queue[0][0] - d > lookahead:
            queue.popleft()
        if queue and queue[0][0] > d:
            price_out[d] = queue[0][1]
            dist_out[d] = queue[0][0] - d
        if is_verified[d]:
            queue.appendleft((d, prices[d]))

    return price_out, dist_out


def _process_one_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SR labels for one stock. Returns DataFrame with price+dist columns."""
    df = df.sort_values("trade_date").reset_index(drop=True)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    out = df[
        ["ts_code", "trade_date", "open", "high", "low", "close", "volume", "vwap"]
    ].copy()

    for suffix, neighborhood, lookahead in _HORIZONS:
        # --- Resistance ---
        is_peak = np.zeros(len(high), dtype=bool)
        for d in range(len(high)):
            lo = max(0, d - neighborhood)
            hi = min(len(high), d + neighborhood + 1)
            if high[d] == high[lo:hi].max():
                is_peak[d] = True

        verified_res = np.zeros(len(high), dtype=bool)
        for d in range(len(high) - 2):
            if is_peak[d]:
                decline = high[d] - high[d + 1 : min(d + 4, len(high))].min()
                if decline / high[d] >= 0.005:
                    verified_res[d] = True

        prices, dists = _nearest_peak_and_dist(high, verified_res, lookahead)
        out[f"resistance_{suffix}_price"] = prices
        out[f"resistance_{suffix}_dist"] = dists

        # --- Support ---
        is_trough = np.zeros(len(high), dtype=bool)
        for d in range(len(high)):
            lo = max(0, d - neighborhood)
            hi = min(len(high), d + neighborhood + 1)
            if low[d] == low[lo:hi].min():
                is_trough[d] = True

        verified_sup = np.zeros(len(high), dtype=bool)
        for d in range(len(high) - 2):
            if is_trough[d]:
                bounce = low[d + 1 : min(d + 4, len(high))].max() - low[d]
                if bounce / low[d] >= 0.005:
                    verified_sup[d] = True

        prices, dists = _nearest_peak_and_dist(low, verified_sup, lookahead)
        out[f"support_{suffix}_price"] = prices
        out[f"support_{suffix}_dist"] = dists

    return out


def build_cache(data_dir: Path) -> int:
    """Build alpha360 cache using multiprocessing. Returns number of date files written."""
    all_df = _load_all_daily(data_dir)
    ts_codes = all_df["ts_code"].unique()
    logger.info("Loaded %d daily rows across %d stocks", len(all_df), len(ts_codes))

    # Stock-level parallel processing
    stock_dfs = [all_df[all_df["ts_code"] == code].copy() for code in ts_codes]
    results: list[pd.DataFrame] = []
    n_workers = max(1, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_one_stock, sd): i for i, sd in enumerate(stock_dfs)
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing stocks"
        ):
            results.append(future.result())

    out_df = pd.concat(results, ignore_index=True)

    all_cols = (
        ["ts_code", "trade_date"] + _FEATURE_COLS + _SR_PRICE_COLS + _SR_DIST_COLS
    )
    out_df = out_df[[c for c in all_cols if c in out_df.columns]]

    output_dir = data_dir / _OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for trade_date, group in tqdm(
        out_df.groupby("trade_date"), desc="Writing date files"
    ):
        path = output_dir / f"{trade_date}.parquet"
        group = group.drop(columns=["trade_date"])
        group.to_parquet(path, index=False)
        written += 1

    logger.info("Wrote %d alpha360 cache files to %s", written, output_dir)
    return written
