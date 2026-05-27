"""Dataset builder — per-sequence normalization, class-index labels."""

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
]
_SR_PRICE_COLS = [
    f"{side}_{s}_price"
    for side in ("resistance", "support")
    for s in ("5d", "20d", "60d")
]
_SR_DIST_COLS = [
    f"{side}_{s}_dist"
    for side in ("resistance", "support")
    for s in ("5d", "20d", "60d")
]


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


def _compute_log_vol_stats(df: pd.DataFrame) -> tuple[float, float]:
    """Compute mean/std of log(1+volume) across training set."""
    vol = df["volume"].dropna().values
    log_vol = np.log1p(vol)
    return float(log_vol.mean()), float(log_vol.std() + 1e-8)


def _normalize_sequence(
    x: np.ndarray, log_vol_mean: float, log_vol_std: float
) -> np.ndarray:
    """Normalize a (60, N) sequence: price ratios + log vol + per-seq z-score."""
    out = x.copy().astype(np.float32)
    close_last = out[-1, 3]
    if close_last <= 0:
        close_last = 1.0

    # Original price features (indices 0,1,2,3,5) → relative to last close
    out[:, 0] = out[:, 0] / close_last - 1  # open
    out[:, 1] = out[:, 1] / close_last - 1  # high
    out[:, 2] = out[:, 2] / close_last - 1  # low
    out[:, 3] = out[:, 3] / close_last - 1  # close
    out[:, 5] = out[:, 5] / close_last - 1  # vwap

    # Volume (index 4) → log transform + z-score
    out[:, 4] = (np.log1p(out[:, 4]) - log_vol_mean) / log_vol_std

    # New features (indices 6+): per-sequence z-score
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
    n_bins: int,
    price_range: float,
    log_vol_mean: float,
    log_vol_std: float,
):
    """Build (X, y, weight) sequences from per-stock data.

    X: (seq_length, n_features) — per-sequence normalized
    y: (6,) int64 — correct bin index per horizon
    weight: (6,) float32 — distance-decayed weight, 0 for invalid
    """
    features, labels, weights = [], [], []

    for ts_code, stock_df in df.groupby("ts_code"):
        stock_df = stock_df.sort_values("trade_date").reset_index(drop=True)
        vals = stock_df[_FEATURE_COLS].to_numpy(dtype=np.float32)
        sr_prices = stock_df[_SR_PRICE_COLS].to_numpy(dtype=np.float32)
        sr_dists = stock_df[_SR_DIST_COLS].to_numpy(dtype=np.float32)

        for i in range(0, len(stock_df) - seq_length, stride):
            x = vals[i : i + seq_length].copy()
            prices = sr_prices[i + seq_length - 1]
            dists = sr_dists[i + seq_length - 1]

            if np.isnan(x).any():
                continue

            x = _normalize_sequence(x, log_vol_mean, log_vol_std)

            has_peak = ~np.isnan(prices)
            y = np.zeros(6, dtype=np.int64)
            w = np.zeros(6, dtype=np.float32)

            for j in range(6):
                if has_peak[j]:
                    close_last = vals[i + seq_length - 1, 3]
                    ratio = (prices[j] - close_last) / close_last
                    n_price_bins = n_bins - 1
                    bin_idx = int((ratio / price_range + 1) * n_price_bins / 2)
                    bin_idx = max(0, min(n_price_bins - 1, bin_idx))
                    y[j] = bin_idx
                    w[j] = 1.0 / (1.0 + dists[j] / 5.0)
                else:
                    y[j] = n_bins - 1
                    w[j] = 1.0

            features.append(x)
            labels.append(y)
            weights.append(w)

    if not features:
        raise ValueError("No valid sequences built")

    return np.stack(features), np.stack(labels), np.stack(weights)


class SRSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, weight: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()
        self.weight = torch.from_numpy(weight).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.weight[idx]


def build_datasets(
    data_dir: Path,
    config,
) -> tuple[SRSequenceDataset, SRSequenceDataset, dict]:
    logger.info(
        "Loading alpha360 cache for train range %s-%s",
        config.train_start,
        config.train_end,
    )
    train_df = _load_alpha360_range(data_dir, config.train_start, config.train_end)
    logger.info("Train data: %d rows", len(train_df))

    val_df = _load_alpha360_range(data_dir, config.val_start, config.val_end)
    logger.info("Val data: %d rows", len(val_df))

    # Log-volume stats from training set only
    log_vol_mean, log_vol_std = _compute_log_vol_stats(train_df)
    norm_params = {"log_vol_mean": log_vol_mean, "log_vol_std": log_vol_std}

    logger.info(
        "Building train sequences (seq=%d, stride=%d)...",
        config.seq_length,
        config.stride,
    )
    X_tr, Y_tr, W_tr = _build_sequences(
        train_df,
        config.seq_length,
        config.stride,
        config.n_bins,
        config.price_range,
        log_vol_mean,
        log_vol_std,
    )
    logger.info("Train: %d sequences", len(X_tr))

    logger.info("Building val sequences...")
    X_val, Y_val, W_val = _build_sequences(
        val_df,
        config.seq_length,
        config.stride,
        config.n_bins,
        config.price_range,
        log_vol_mean,
        log_vol_std,
    )
    logger.info("Val: %d sequences", len(X_val))

    return (
        SRSequenceDataset(X_tr, Y_tr, W_tr),
        SRSequenceDataset(X_val, Y_val, W_val),
        norm_params,
    )
