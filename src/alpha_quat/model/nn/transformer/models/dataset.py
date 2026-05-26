import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_FEATURE_COLS = ["open", "high", "low", "close", "volume", "vwap"]
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


def _compute_norm_params(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    params = {}
    for col in _FEATURE_COLS:
        vals = df[col].dropna().values
        if len(vals) == 0:
            params[col] = (0.0, 1.0)
        else:
            params[col] = (float(vals.mean()), float(vals.std() + 1e-8))
    for col in _SR_PRICE_COLS:
        vals = df[col].dropna().values
        if len(vals) == 0:
            params[col] = (0.0, 1.0)
        else:
            params[col] = (float(vals.mean()), float(vals.std() + 1e-8))
    return params


def _normalize(
    df: pd.DataFrame, params: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    df = df.copy()
    for col, (mean, std) in params.items():
        if col in df.columns:
            df[col] = (df[col] - mean) / std
    return df


def _build_sequences(
    df: pd.DataFrame, seq_length: int, stride: int, n_bins: int, price_range: float
):
    """Build (X, y, weight) sequences from per-stock data.

    X: (seq_length, 6) normalized features
    y: (6,) int64 — correct bin index per horizon, 0 for invalid
    weight: (6,) float32 — distance-decayed weight, 0 for invalid
    """
    features = []
    labels = []
    weights = []

    for ts_code, stock_df in df.groupby("ts_code"):
        stock_df = stock_df.sort_values("trade_date").reset_index(drop=True)
        vals = stock_df[_FEATURE_COLS].to_numpy(dtype=np.float32)
        sr_prices = stock_df[_SR_PRICE_COLS].to_numpy(dtype=np.float32)
        sr_dists = stock_df[_SR_DIST_COLS].to_numpy(dtype=np.float32)

        for i in range(0, len(stock_df) - seq_length, stride):
            x = vals[i : i + seq_length]
            prices = sr_prices[i + seq_length - 1]
            dists = sr_dists[i + seq_length - 1]

            if np.isnan(x).any():
                continue
            if np.isnan(prices).all():
                continue

            valid = ~np.isnan(prices)
            y = np.zeros(6, dtype=np.int64)
            w = np.zeros(6, dtype=np.float32)

            for j in range(6):
                if valid[j]:
                    close_last = vals[i + seq_length - 1, 3]
                    ratio = (prices[j] - close_last) / close_last
                    bin_idx = int((ratio / price_range + 1) * n_bins / 2)
                    bin_idx = max(0, min(n_bins - 1, bin_idx))
                    y[j] = bin_idx
                    # Weight: nearer peaks matter more, half-life ~5 trading days
                    w[j] = 1.0 / (1.0 + dists[j] / 5.0)

            features.append(x)
            labels.append(y)
            weights.append(w)

    if not features:
        raise ValueError("No valid sequences built")

    X = np.stack(features)
    Y = np.stack(labels)
    W = np.stack(weights)
    return X, Y, W


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

    combined = pd.concat([train_df, val_df], ignore_index=True)
    norm_params = _compute_norm_params(combined)

    train_norm = _normalize(train_df, norm_params)
    val_norm = _normalize(val_df, norm_params)

    logger.info(
        "Building train sequences (seq=%d, stride=%d)...",
        config.seq_length,
        config.stride,
    )
    X_tr, Y_tr, W_tr = _build_sequences(
        train_norm, config.seq_length, config.stride, config.n_bins, config.price_range
    )
    logger.info("Train: %d sequences", len(X_tr))

    logger.info("Building val sequences...")
    X_val, Y_val, W_val = _build_sequences(
        val_norm, config.seq_length, config.stride, config.n_bins, config.price_range
    )
    logger.info("Val: %d sequences", len(X_val))

    return (
        SRSequenceDataset(X_tr, Y_tr, W_tr),
        SRSequenceDataset(X_val, Y_val, W_val),
        norm_params,
    )
