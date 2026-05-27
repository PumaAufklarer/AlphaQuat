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


def _load_universe(data_dir: Path) -> tuple[set[str], set[str]]:
    """Load stock_basic to get main-board codes and industry info.

    Returns (main_board_codes, all_codes_with_industry).
    """
    sb = pd.read_parquet(data_dir / "stock_basic.parquet")
    codes = set(sb["ts_code"])
    main_board = set(sb.loc[sb["market"] == "主板", "ts_code"])
    return main_board, codes


def _load_st_dates(data_dir: Path, start: str, end: str) -> set[tuple[str, str]]:
    """Load stock_st data for date range, return {(ts_code, trade_date), ...}."""
    st_dir = data_dir / "stock_st"
    st_set: set[tuple[str, str]] = set()
    for f in st_dir.glob("*.parquet"):
        ds = f.stem.replace("_", "")
        if start <= ds <= end:
            st = pd.read_parquet(f)
            st["trade_date"] = ds
            for _, row in st.iterrows():
                st_set.add((str(row["ts_code"]), ds))
    if st_set:
        logger.info("Loaded %d ST entries from %s-%s", len(st_set), start, end)
    return st_set


def _load_stock_data(
    data_dir: Path,
    start: str,
    end: str,
    filter_universe: bool = True,
    circ_mv_percentile: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Load alpha360 data, filtered to main-board non-ST universe.

    Args:
        filter_universe: Only keep 主板 stocks, remove ST/*ST entries.
        circ_mv_percentile: If set, filter out stocks below this circ_mv percentile
                            (e.g., 0.2 = drop bottom 20% by market cap).
    """
    cache_dir = data_dir / "alpha360"
    all_dates = sorted(
        d.stem for d in cache_dir.glob("*.parquet") if start <= d.stem <= end
    )
    if not all_dates:
        raise FileNotFoundError(f"No alpha360 cache found for {start}~{end}")

    # Universe: main board codes
    main_board, _ = _load_universe(data_dir) if filter_universe else (set(), set())

    # ST set
    st_set = (
        _load_st_dates(data_dir, all_dates[0], all_dates[-1])
        if filter_universe
        else set()
    )

    # Market cap percentile filter
    mv_filter: set[str] | None = None
    if circ_mv_percentile is not None and filter_universe:
        daily_basic_dir = data_dir / "daily_basic"
        mv_codes: dict[str, float] = {}
        for f in daily_basic_dir.glob("*.parquet"):
            ds = f.stem.replace("_", "")
            if start <= ds <= end:
                db = pd.read_parquet(f)
                for _, row in db.iterrows():
                    code = str(row["ts_code"])
                    mv = float(row.get("circ_mv", 0) or 0)
                    mv_codes[code] = max(mv_codes.get(code, 0), mv)
        if mv_codes:
            vals = sorted(mv_codes.values())
            thresh = vals[int(len(vals) * circ_mv_percentile)]
            mv_filter = {c for c, v in mv_codes.items() if v >= thresh}
            logger.info(
                "Market cap filter: %d stocks >= %.0e (p%.0f)",
                len(mv_filter),
                thresh,
                circ_mv_percentile * 100,
            )

    # Load alpha360
    chunks = []
    for d in all_dates:
        df = pd.read_parquet(cache_dir / f"{d}.parquet")
        df["trade_date"] = d
        if filter_universe:
            before = len(df)
            df = df[df["ts_code"].isin(main_board)]
            if st_set:
                st_mask = df.apply(
                    lambda r: (r["ts_code"], r["trade_date"]) in st_set, axis=1
                )
                df = df[~st_mask]
            if mv_filter is not None:
                df = df[df["ts_code"].isin(mv_filter)]
            after = len(df)
            if after < before:
                logger.debug("  %s: filtered %d→%d rows", d, before, after)
        chunks.append(df)
    full = pd.concat(chunks, ignore_index=True)
    logger.info(
        "Loaded %d rows from %s to %s (filter=%s)",
        len(full),
        start,
        end,
        filter_universe,
    )

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
