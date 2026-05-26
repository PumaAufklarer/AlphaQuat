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
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

_OUTPUT_DIR = "alpha360"

_FEATURE_COLS = ["open", "high", "low", "close", "volume", "vwap"]
_SR_SUFFIXES = ["5d", "20d", "60d"]

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


def _find_local_peaks(prices: np.ndarray, neighborhood: int) -> np.ndarray:
    """Vectorized local peak detection via rolling max."""
    s = pd.Series(prices)
    rolling = s.rolling(2 * neighborhood + 1, center=True, min_periods=1)
    return (prices == rolling.max().values).astype(bool)


def _find_local_troughs(prices: np.ndarray, neighborhood: int) -> np.ndarray:
    """Vectorized local trough detection via rolling min."""
    s = pd.Series(prices)
    rolling = s.rolling(2 * neighborhood + 1, center=True, min_periods=1)
    return (prices == rolling.min().values).astype(bool)


def _verify_extreme(
    values: np.ndarray,
    is_extreme: np.ndarray,
    is_resistance: bool,
) -> np.ndarray:
    """Vectorized verification: for extremes, check rejection/bounce within 3 days."""
    rev = values[::-1]
    fwd_window = pd.Series(rev).rolling(3, min_periods=1)
    if is_resistance:
        peak_vals = values
        fwd_min = fwd_window.min().values[::-1]  # min(values[d:d+3])
        fwd_min_shifted = np.roll(fwd_min, -1)
        fwd_min_shifted[-1] = np.nan
        movement = peak_vals - fwd_min_shifted
    else:
        trough_vals = values
        fwd_max = fwd_window.max().values[::-1]
        fwd_max_shifted = np.roll(fwd_max, -1)
        fwd_max_shifted[-1] = np.nan
        movement = fwd_max_shifted - trough_vals

    verified = is_extreme.copy()
    valid = is_extreme & ~np.isnan(movement)
    verified[valid] = (movement[valid] / values[valid]) >= 0.005
    return verified


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
    """Compute SR labels for one stock with vectorized operations."""
    df = df.sort_values("trade_date").reset_index(drop=True)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    out = df[
        ["ts_code", "trade_date", "open", "high", "low", "close", "volume", "vwap"]
    ].copy()

    for suffix, neighborhood, lookahead in _HORIZONS:
        # Resistance: local peak + verification + nearest fill
        is_peak = _find_local_peaks(high, neighborhood)
        verified_res = _verify_extreme(high, is_peak, is_resistance=True)
        prices, dists = _nearest_peak_and_dist(high, verified_res, lookahead)
        out[f"resistance_{suffix}_price"] = prices
        out[f"resistance_{suffix}_dist"] = dists

        # Support: local trough + verification + nearest fill
        is_trough = _find_local_troughs(low, neighborhood)
        verified_sup = _verify_extreme(low, is_trough, is_resistance=False)
        prices, dists = _nearest_peak_and_dist(low, verified_sup, lookahead)
        out[f"support_{suffix}_price"] = prices
        out[f"support_{suffix}_dist"] = dists

    return out


def build_cache(data_dir: Path) -> int:
    """Build alpha360 cache. Returns number of date files written."""
    all_df = _load_all_daily(data_dir)
    logger.info(
        "Loaded %d daily rows across %d stocks",
        len(all_df),
        all_df["ts_code"].nunique(),
    )

    results: list[pd.DataFrame] = []
    for ts_code, stock_df in tqdm(
        all_df.groupby("ts_code", sort=False), desc="Processing stocks"
    ):
        processed = _process_one_stock(stock_df)
        results.append(processed)

    out_df = pd.concat(results, ignore_index=True)

    sr_cols = [
        f"{side}_{s}_{attr}"
        for side in ("resistance", "support")
        for s in _SR_SUFFIXES
        for attr in ("price", "dist")
    ]
    all_cols = ["ts_code", "trade_date"] + _FEATURE_COLS + sr_cols
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
