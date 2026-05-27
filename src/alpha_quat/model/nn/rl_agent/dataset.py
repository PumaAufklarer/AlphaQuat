"""State builder and episode data loader for RL agent."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "volume_ratio",
    "turnover_rate",
    "hl_ratio",
    "ret_5d",
    "close_ma20",
    "atr_ratio",
    "vol_change",
    "amt_change",
]

_COMMISSION = 0.0005
_REWARD_HORIZON = 5


def _normalize_market(x: np.ndarray) -> np.ndarray:
    """Per-sequence z-score normalization for (60, 14) market data."""
    out = x.copy().astype(np.float32)
    close_last = out[-1, 3]
    if close_last <= 0:
        close_last = 1.0

    out[:, 0] = out[:, 0] / close_last - 1
    out[:, 1] = out[:, 1] / close_last - 1
    out[:, 2] = out[:, 2] / close_last - 1
    out[:, 3] = out[:, 3] / close_last - 1
    out[:, 5] = out[:, 5] / close_last - 1
    out[:, 4] = np.log1p(out[:, 4])

    for j in range(6, x.shape[1]):
        col = out[:, j]
        mean = col.mean()
        std = col.std()
        if std > 1e-8:
            out[:, j] = (col - mean) / std
        else:
            out[:, j] = 0.0
        out[:, j] = np.clip(out[:, j], -5.0, 5.0)

    return out


def _load_stock_data(data_dir: Path, start: str, end: str) -> dict[str, pd.DataFrame]:
    """Load alpha360 data for date range, return dict of {ts_code: df}."""
    cache_dir = data_dir / "alpha360"
    all_dates = sorted(
        d.stem for d in cache_dir.glob("*.parquet") if start <= d.stem <= end
    )
    if not all_dates:
        raise FileNotFoundError(f"No alpha360 cache found for {start}~{end}")

    chunks = []
    for d in all_dates:
        df = pd.read_parquet(cache_dir / f"{d}.parquet")
        df["trade_date"] = d
        chunks.append(df)
    full = pd.concat(chunks, ignore_index=True)
    logger.info("Loaded %d rows from %s to %s", len(full), start, end)

    result = {}
    for code, grp in full.groupby("ts_code"):
        result[code] = grp.sort_values("trade_date").reset_index(drop=True)
    logger.info("Loaded %d stocks", len(result))
    return result


def build_state(
    market_vals: np.ndarray,
    position: float,
    days_held: int,
    seq_length: int = 60,
) -> np.ndarray:
    """Build (seq_length, 16) state from normalized market data + agent state.

    market_vals: (seq_length, 14) raw market data
    position: current position [-1, 1]
    days_held: days since last position change
    Returns: (seq_length, 16) float32 array
    """
    market = _normalize_market(market_vals)
    pos_signal = np.full((seq_length, 1), position, dtype=np.float32)
    dh = min(days_held, 30) / 30.0
    days_signal = np.full((seq_length, 1), dh, dtype=np.float32)
    return np.concatenate([market, pos_signal, days_signal], axis=1)


def compute_reward(
    close_current: float,
    close_future: float,
    position: float,
    prev_position: float,
    atr_ratio: float,
    commission: float = _COMMISSION,
) -> float:
    """Compute vol-normalized profit for taking position at current close.

    ret = close_future / close_current - 1
    reward = position * (ret / atr_ratio) - trade_cost
    """
    ret = close_future / close_current - 1.0
    vol = max(atr_ratio, 1e-6)
    trade_cost = abs(position - prev_position) * commission
    profit = position * (ret / vol)
    return float(profit - trade_cost)


def select_stocks(
    stock_data: dict[str, pd.DataFrame],
    max_stocks: int = 50,
    min_days: int = 120,
) -> list[str]:
    """Select stocks with enough data for training episodes.

    Skips stocks with fewer than min_days of data.
    If more than max_stocks qualify, randomly sample.
    """
    eligible = [k for k, v in stock_data.items() if len(v) >= min_days]
    logger.info("Eligible stocks: %d (min %d days)", len(eligible), min_days)
    if len(eligible) > max_stocks:
        rng = np.random.default_rng(42)
        eligible = list(rng.choice(eligible, size=max_stocks, replace=False))
    return sorted(eligible)
