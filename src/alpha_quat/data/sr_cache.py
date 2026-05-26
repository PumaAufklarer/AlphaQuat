"""Alpha360 cache builder — pre-compute 6 raw features + SR labels per day.

Output: data/alpha360/YYYYMMDD.parquet — one file per date with columns:
  ts_code, open, high, low, close, volume, vwap,
  resistance_5d, resistance_20d, resistance_60d,
  support_5d, support_20d, support_60d
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
_SR_COLS = [
    "resistance_5d",
    "resistance_20d",
    "resistance_60d",
    "support_5d",
    "support_20d",
    "support_60d",
]

# Each horizon: (neighborhood, max_lookahead)
_HORIZONS = [
    ("5d", 2, 5),
    ("20d", 10, 20),
    ("60d", 30, 60),
]


def _load_all_daily(data_dir: Path) -> pd.DataFrame:
    """Read all daily parquet files, compute vwap."""
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


def _find_sr_for_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized SR label computation for one stock's sorted daily data."""
    df = df.sort_values("trade_date").reset_index(drop=True)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    out = df.copy()
    for col in _SR_COLS:
        out[col] = np.nan

    for suffix, neighborhood, lookahead in _HORIZONS:
        _compute_one_horizon(out, high, low, suffix, neighborhood, lookahead)

    return out


def _compute_one_horizon(
    out: pd.DataFrame,
    high: np.ndarray,
    low: np.ndarray,
    suffix: str,
    neighborhood: int,
    lookahead: int,
):
    """Compute resistance/support for one horizon using rolling + reverse fill."""
    T = len(high)

    # --- Resistance: find local peaks with rejection ---
    is_peak = np.zeros(T, dtype=bool)
    for d in range(0, T):
        lo = max(0, d - neighborhood)
        hi = min(T, d + neighborhood + 1)
        if high[d] == high[lo:hi].max():
            is_peak[d] = True

    # Verify rejection: decline within 3 days after peak
    verified_resistance = np.zeros(T, dtype=bool)
    for d in range(0, T - 2):
        if is_peak[d]:
            decline = high[d] - high[d + 1 : min(d + 4, T)].min()
            if decline / high[d] >= 0.005:
                verified_resistance[d] = True

    # Reverse fill: nearest verified peak within lookahead
    resistance_prices = _nearest_peak_price(high, verified_resistance, lookahead)
    out[f"resistance_{suffix}"] = resistance_prices

    # --- Support: find local troughs with bounce ---
    is_trough = np.zeros(T, dtype=bool)
    for d in range(0, T):
        lo = max(0, d - neighborhood)
        hi = min(T, d + neighborhood + 1)
        if low[d] == low[lo:hi].min():
            is_trough[d] = True

    # Verify bounce: rise within 3 days after trough
    verified_support = np.zeros(T, dtype=bool)
    for d in range(0, T - 2):
        if is_trough[d]:
            bounce = low[d + 1 : min(d + 4, T)].max() - low[d]
            if bounce / low[d] >= 0.005:
                verified_support[d] = True

    support_prices = _nearest_peak_price(low, verified_support, lookahead)
    out[f"support_{suffix}"] = support_prices


def _nearest_peak_price(
    prices: np.ndarray,
    is_verified: np.ndarray,
    lookahead: int,
) -> np.ndarray:
    """For each day d, find nearest verified price at > d within lookahead days.

    Uses reverse scan with deque (O(n) per horizon).
    """
    T = len(prices)
    result = np.full(T, np.nan)
    queue: deque[tuple[int, float]] = deque()

    for d in range(T - 1, -1, -1):
        # Clean peaks too far ahead
        while queue and queue[0][0] - d > lookahead:
            queue.popleft()

        # Verify the nearest peak is strictly in the future (idx > d)
        if queue and queue[0][0] > d:
            result[d] = queue[0][1]

        # Add current day's peak for earlier days to use
        if is_verified[d]:
            queue.appendleft((d, prices[d]))

    return result


def build_cache(data_dir: Path) -> int:
    """Build / update alpha360 cache. Returns number of date files written."""
    all_df = _load_all_daily(data_dir)
    ts_codes = all_df["ts_code"].unique()
    logger.info("Loaded %d daily rows across %d stocks", len(all_df), len(ts_codes))

    results = []
    for ts_code in tqdm(ts_codes, desc="Processing stocks"):
        stock_df = all_df[all_df["ts_code"] == ts_code].copy()
        processed = _find_sr_for_stock(stock_df)
        results.append(processed)

    out_df = pd.concat(results, ignore_index=True)

    all_cols = ["ts_code", "trade_date"] + _FEATURE_COLS + _SR_COLS
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
