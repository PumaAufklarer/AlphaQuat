"""Dataset builder for Keltner regime prediction — labels from future channel position."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
    "keltner_pos",
    "keltner_width",
    "keltner_above_ema",
]

_HORIZONS = [5, 20, 60]

_REGIME_NAMES = [
    "ranging",
    "support_test",
    "resistance_test",
    "breakout_up",
    "breakout_down",
]


def _keltner_regime(k_pos: float) -> int:
    """Map Keltner position to regime ID (mutually exclusive by priority)."""
    if k_pos > 1.0:
        return 3  # breakout up
    if k_pos < -1.0:
        return 4  # breakout down
    if k_pos >= 0.5:
        return 2  # resistance test
    if k_pos <= -0.5:
        return 1  # support test
    return 0  # ranging


def _load_alpha360_range(data_dir: Path, start: str, end: str) -> pd.DataFrame:
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
    return pd.concat(chunks, ignore_index=True)


def _normalize_sequence(x: np.ndarray) -> np.ndarray:
    """Per-sequence z-score normalization for all features."""
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


def _build_sequences(
    df: pd.DataFrame,
    seq_length: int,
    stride: int,
):
    """Build (X, y) sequences from per-stock Keltner data.

    X: (seq_length, n_features) — normalized
    y: (n_horizons,) int64 — regime ID per horizon (5d, 20d, 60d)
    """
    features, labels = [], []
    max_horizon = max(_HORIZONS)

    for ts_code, stock_df in df.groupby("ts_code"):
        stock_df = stock_df.sort_values("trade_date").reset_index(drop=True)
        vals = stock_df[_FEATURE_COLS].to_numpy(dtype=np.float32)
        kpos = stock_df["keltner_pos"].to_numpy(dtype=np.float64)

        for i in range(0, len(stock_df) - seq_length - max_horizon, stride):
            x = vals[i : i + seq_length].copy()
            if np.isnan(x).any():
                continue

            x = _normalize_sequence(x)

            y = np.zeros(len(_HORIZONS), dtype=np.int64)
            for h_idx, horizon in enumerate(_HORIZONS):
                future_pos = kpos[i + seq_length + horizon - 1]
                y[h_idx] = _keltner_regime(future_pos)

            features.append(x)
            labels.append(y)

    if not features:
        raise ValueError("No valid sequences built")

    return np.stack(features), np.stack(labels)


class KeltnerRegimeDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def build_datasets(
    data_dir: Path,
    seq_length: int = 60,
    stride: int = 10,
    train_start: str = "20200101",
    train_end: str = "20231231",
    val_start: str = "20240101",
    val_end: str = "20240630",
) -> tuple[KeltnerRegimeDataset, KeltnerRegimeDataset]:
    pad = 120  # extra trading days for lookahead (max horizon 60 + buffer)

    def _pad_end(d: str) -> str:
        from datetime import datetime, timedelta

        dt = datetime.strptime(d, "%Y%m%d") + timedelta(days=pad)
        return dt.strftime("%Y%m%d")

    logger.info("Loading alpha360 cache for train range %s-%s", train_start, train_end)
    train_df = _load_alpha360_range(data_dir, train_start, _pad_end(train_end))
    logger.info("Train data: %d rows", len(train_df))

    val_df = _load_alpha360_range(data_dir, val_start, _pad_end(val_end))
    logger.info("Val data: %d rows", len(val_df))

    logger.info("Building train sequences (seq=%d, stride=%d)...", seq_length, stride)
    X_tr, Y_tr = _build_sequences(train_df, seq_length, stride)
    logger.info("Train: %d sequences", len(X_tr))

    logger.info("Building val sequences...")
    X_val, Y_val = _build_sequences(val_df, seq_length, stride)
    logger.info("Val: %d sequences", len(X_val))

    return KeltnerRegimeDataset(X_tr, Y_tr), KeltnerRegimeDataset(X_val, Y_val)
