"""Alpha360 cache builder — pre-compute 6 raw features + SR labels per day.

Output: data/alpha360/YYYYMMDD.parquet — one file per date with columns:
  ts_code, open, high, low, close, volume, vwap,
  resistance_5d, resistance_20d, resistance_60d,
  support_5d, support_20d, support_60d
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OUTPUT_DIR = "alpha360"

# Default per-date fields (matches tushare daily API)
_FEATURE_COLS = ["open", "high", "low", "close", "volume", "vwap"]
_SR_COLS = [
    "resistance_5d",
    "resistance_20d",
    "resistance_60d",
    "support_5d",
    "support_20d",
    "support_60d",
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
        df["trade_date"] = f.stem.replace("_", "")  # YYYY_MM_DD → YYYYMMDD
        chunks.append(df)
    all_df = pd.concat(chunks, ignore_index=True)

    # Rename vol → volume
    if "volume" not in all_df.columns and "vol" in all_df.columns:
        all_df = all_df.rename(columns={"vol": "volume"})

    # Compute vwap
    all_df["vwap"] = all_df["amount"] / all_df["volume"].replace(0, np.nan)
    return all_df


def _find_resistance_levels(high: np.ndarray, n: int, horizon: int) -> float:
    """Find nearest local peak (resistance) from n+1 within horizon days.

    A local peak = day d where high[d] is max of [d-2:d+3],
    with rejection (price declines %%) after the peak.
    """
    end = min(n + horizon + 1, len(high))
    for d in range(n + 1, end):
        lo = max(0, d - 2)
        hi = min(len(high), d + 3)
        if high[d] == high[lo:hi].max():
            decline_end = min(d + 3, len(high))
            if decline_end > d + 1:
                decline = high[d] - high[d + 1 : decline_end].min()
                if decline / high[d] >= 0.005:
                    return high[d]
    return np.nan


def _find_support_levels(low: np.ndarray, n: int, horizon: int) -> float:
    """Find nearest local trough (support) from n+1 within horizon days.

    A local trough = day d where low[d] is min of [d-2:d+3],
    with bounce (price rises %%) after the trough.
    """
    end = min(n + horizon + 1, len(low))
    for d in range(n + 1, end):
        lo = max(0, d - 2)
        hi = min(len(low), d + 3)
        if low[d] == low[lo:hi].min():
            bounce_end = min(d + 3, len(low))
            if bounce_end > d + 1:
                bounce = low[d + 1 : bounce_end].max() - low[d]
                if bounce / low[d] >= 0.005:
                    return low[d]
    return np.nan


def _process_one_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SR labels for one stock's sorted daily data."""
    df = df.sort_values("trade_date").reset_index(drop=True)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)

    out = df.copy()
    out["resistance_5d"] = np.nan
    out["resistance_20d"] = np.nan
    out["resistance_60d"] = np.nan
    out["support_5d"] = np.nan
    out["support_20d"] = np.nan
    out["support_60d"] = np.nan

    for i in range(n - 1):
        out.loc[i, "resistance_5d"] = _find_resistance_levels(high, i, 5)
        out.loc[i, "resistance_20d"] = _find_resistance_levels(high, i, 20)
        out.loc[i, "resistance_60d"] = _find_resistance_levels(high, i, 60)
        out.loc[i, "support_5d"] = _find_support_levels(low, i, 5)
        out.loc[i, "support_20d"] = _find_support_levels(low, i, 20)
        out.loc[i, "support_60d"] = _find_support_levels(low, i, 60)

    return out


def build_cache(data_dir: Path) -> int:
    """Build / update alpha360 cache. Returns number of date files written."""
    all_df = _load_all_daily(data_dir)
    ts_codes = all_df["ts_code"].unique()
    logger.info("Loaded %d daily rows across %d stocks", len(all_df), len(ts_codes))

    results = []
    for ts_code in ts_codes:
        stock_df = all_df[all_df["ts_code"] == ts_code].copy()
        processed = _process_one_stock(stock_df)
        results.append(processed)
        logger.debug("Processed %s: %d rows", ts_code, len(processed))

    out_df = pd.concat(results, ignore_index=True)

    # Filter to required columns only
    all_cols = ["ts_code", "trade_date"] + _FEATURE_COLS + _SR_COLS
    out_df = out_df[[c for c in all_cols if c in out_df.columns]]

    # Write one file per date
    output_dir = data_dir / _OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for trade_date, group in out_df.groupby("trade_date"):
        path = output_dir / f"{trade_date}.parquet"
        group = group.drop(columns=["trade_date"])
        group.to_parquet(path, index=False)
        written += 1

    logger.info("Wrote %d alpha360 cache files to %s", written, output_dir)
    return written
