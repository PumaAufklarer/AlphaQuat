"""Alpha360 cache builder — pre-compute features + SR labels per day.

Output: data/alpha360/YYYYMMDD.parquet — one file per date with columns:
  ts_code, open, high, low, close, volume, vwap,
  volume_ratio, turnover_rate, hl_ratio, ret_5d, close_ma20, atr_ratio,
  vol_change, amt_change,
  keltner_pos, keltner_width, keltner_above_ema,
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

_FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
]
_NEW_FEATURE_COLS = [
    "volume_ratio",
    "turnover_rate",
    "hl_ratio",
    "ret_5d",
    "close_ma20",
    "atr_ratio",
    "vol_change",
    "amt_change",
]
_KELTNER_FEATURE_COLS = [
    "keltner_pos",
    "keltner_width",
    "keltner_above_ema",
]
_ALL_FEATURE_COLS = _FEATURE_COLS + _NEW_FEATURE_COLS + _KELTNER_FEATURE_COLS
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


def _load_all_daily_basic(data_dir: Path) -> pd.DataFrame:
    basic_dir = data_dir / "daily_basic"
    files = sorted(basic_dir.glob("*.parquet"))
    if not files:
        logger.warning("No daily_basic parquet files found in %s", basic_dir)
        return pd.DataFrame(columns=["ts_code", "trade_date", "turnover_rate"])

    chunks = []
    for f in files:
        df = pd.read_parquet(f)
        df["trade_date"] = f.stem.replace("_", "")
        chunks.append(df)
    all_df = pd.concat(chunks, ignore_index=True)

    if "turnover_rate" not in all_df.columns:
        logger.warning("daily_basic missing 'turnover_rate' column")
        all_df["turnover_rate"] = 0.0
    return all_df[["ts_code", "trade_date", "turnover_rate"]]


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


def _compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stock time-series features. Assumes df is sorted by trade_date."""
    close = pd.to_numeric(df["close"], errors="coerce").replace(0, np.nan)
    vol = pd.to_numeric(df["volume"], errors="coerce").replace(0, np.nan)
    amount = pd.to_numeric(df["amount"], errors="coerce").replace(0, np.nan)

    # Volume ratio: vol / MA(vol, 20)
    df["volume_ratio"] = vol / vol.rolling(20, min_periods=1).mean()
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

    # turnover_rate is already in df from daily_basic merge, fill missing with 0
    df["turnover_rate"] = df.get(
        "turnover_rate", pd.Series(0.0, index=df.index)
    ).fillna(0.0)

    # HL ratio: (high - low) / close
    df["hl_ratio"] = (df["high"] - df["low"]) / close
    df["hl_ratio"] = df["hl_ratio"].fillna(0.0)

    # Ret 5d: close / close[5] - 1
    df["ret_5d"] = df["close"] / df["close"].shift(5) - 1
    df["ret_5d"] = df["ret_5d"].fillna(0.0)

    # Close MA20: close / MA(close, 20) - 1
    df["close_ma20"] = df["close"] / close.rolling(20, min_periods=1).mean() - 1
    df["close_ma20"] = df["close_ma20"].fillna(0.0)

    # ATR(14): max(high-low, high-prev_close, low-prev_close), smoothed
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_ratio"] = tr.rolling(14, min_periods=1).mean() / close
    df["atr_ratio"] = df["atr_ratio"].fillna(0.0)

    # Vol change: log(vol / vol[1])
    df["vol_change"] = np.log(vol / vol.shift(1))
    df["vol_change"] = df["vol_change"].fillna(0.0)

    # Amount change: log(amount / amount[1])
    df["amt_change"] = np.log(amount / amount.shift(1))
    df["amt_change"] = df["amt_change"].fillna(0.0)

    # ── Keltner Channel features ──
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr20 = tr.rolling(20, min_periods=1).mean()

    df["keltner_pos"] = (df["close"] - ema20) / (atr20 * 2.0).replace(0, np.nan)
    df["keltner_width"] = atr20 / ema20.replace(0, np.nan)
    df["keltner_above_ema"] = df["close"] / ema20.replace(0, np.nan) - 1.0

    for col in _KELTNER_FEATURE_COLS:
        df[col] = df[col].fillna(0.0)

    return df


def _process_one_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SR labels + derived features for one stock."""
    df = df.sort_values("trade_date").reset_index(drop=True)
    df = _compute_derived_features(df)

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    out = df[["ts_code", "trade_date"] + _ALL_FEATURE_COLS].copy()

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

    # Load and merge daily_basic for turnover_rate
    basic_df = _load_all_daily_basic(data_dir)
    if not basic_df.empty:
        all_df = all_df.merge(
            basic_df[["ts_code", "trade_date", "turnover_rate"]],
            on=["ts_code", "trade_date"],
            how="left",
        )
    else:
        all_df["turnover_rate"] = 0.0

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
    all_cols = ["ts_code", "trade_date"] + _ALL_FEATURE_COLS + sr_cols
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
