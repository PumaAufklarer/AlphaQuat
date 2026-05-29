"""Tests for SR dataset — per-sequence normalization."""

import numpy as np
import pandas as pd

from alpha_quat.model.nn.transformer.models.dataset import (
    _compute_log_vol_stats,
    _normalize_sequence,
    _build_sequences,
)


def _make_synthetic_alpha360(n_stocks=2, n_days=100):
    rows = []
    rng = np.random.RandomState(42)
    for stock in [f"00000{i}.SZ" for i in range(1, n_stocks + 1)]:
        price = 10.0
        for d in range(n_days):
            price *= 1 + rng.randn() * 0.01
            rows.append(
                {
                    "ts_code": stock,
                    "trade_date": f"2024{d // 12 + 1:02d}{d % 12 + 1:02d}",
                    "open": price * (1 + rng.randn() * 0.005),
                    "high": price * (1 + abs(rng.randn()) * 0.01),
                    "low": price * (1 - abs(rng.randn()) * 0.01),
                    "close": price,
                    "volume": rng.randint(100000, 1000000),
                    "vwap": price * (1 + rng.randn() * 0.003),
                    "volume_ratio": 1.0 + rng.randn() * 0.05,
                    "turnover_rate": abs(rng.randn()) * 0.5,
                    "hl_ratio": 0.02 + abs(rng.randn()) * 0.005,
                    "ret_5d": rng.randn() * 0.02,
                    "close_ma20": rng.randn() * 0.02,
                    "atr_ratio": 0.03 + abs(rng.randn()) * 0.005,
                    "vol_change": rng.randn() * 0.1,
                    "amt_change": rng.randn() * 0.1,
                    "resistance_5d_price": price * 1.05 if d < n_days - 5 else np.nan,
                    "resistance_20d_price": price * 1.10 if d < n_days - 20 else np.nan,
                    "resistance_60d_price": price * 1.20 if d < n_days - 60 else np.nan,
                    "support_5d_price": price * 0.95 if d < n_days - 5 else np.nan,
                    "support_20d_price": price * 0.90 if d < n_days - 20 else np.nan,
                    "support_60d_price": price * 0.80 if d < n_days - 60 else np.nan,
                    "resistance_5d_dist": 3.0 if d < n_days - 5 else np.nan,
                    "resistance_20d_dist": 10.0 if d < n_days - 20 else np.nan,
                    "resistance_60d_dist": 30.0 if d < n_days - 60 else np.nan,
                    "support_5d_dist": 3.0 if d < n_days - 5 else np.nan,
                    "support_20d_dist": 10.0 if d < n_days - 20 else np.nan,
                    "support_60d_dist": 30.0 if d < n_days - 60 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_log_vol_stats():
    df = _make_synthetic_alpha360(n_stocks=1, n_days=50)
    mean, std = _compute_log_vol_stats(df)
    assert mean > 0
    assert std > 0


def test_normalize_sequence_shape():
    rng = np.random.RandomState(42)
    seq = rng.uniform(10, 50, (60, 6)).astype(np.float32)
    result = _normalize_sequence(seq, log_vol_mean=12.0, log_vol_std=1.0)
    assert result.shape == (60, 6)
    assert result.dtype == np.float32
    # Close at last position should be 0 (normalized relative to itself)
    assert abs(result[-1, 3]) < 1e-5
    assert np.isfinite(result).all()


def test_build_sequences_shape():
    df = _make_synthetic_alpha360(n_stocks=2, n_days=100)
    mean, std = _compute_log_vol_stats(df)
    X, Y, W = _build_sequences(
        df,
        seq_length=20,
        stride=10,
        n_bins=10,
        price_range=0.10,
        log_vol_mean=mean,
        log_vol_std=std,
    )
    assert X.ndim == 3
    assert X.shape[1] == 20
    assert X.shape[2] == 14
    assert Y.ndim == 2
    assert Y.shape[1] == 6
    assert Y.dtype == np.int64
    assert W.shape == (Y.shape[0], 6)
    assert W.dtype == np.float32
    assert np.isfinite(X).all()
    for i in range(Y.shape[0]):
        for j in range(6):
            if W[i, j] > 0:
                assert 0 <= Y[i, j] < 10
                assert 0 < W[i, j] <= 1.0
                # Close at last day of each sequence should be ~0
                assert abs(X[i, -1, 3]) < 1e-5


def test_normalize_price_ratio():
    """Open 10, last close 50 → normalized open = 10/50 - 1 = -0.8."""
    rng = np.random.RandomState(42)
    seq = rng.uniform(10, 50, (60, 6)).astype(np.float32)
    seq[:, 3] = np.linspace(10, 50, 60)  # ramp up
    result = _normalize_sequence(seq, log_vol_mean=12.0, log_vol_std=1.0)
    # First day close = 10, last close = 50 → normalized = 10/50 - 1 = -0.8
    assert abs(result[0, 3] - (10.0 / 50.0 - 1.0)) < 1e-5
    # Last day close = 50, last close = 50 → 0
    assert abs(result[-1, 3]) < 1e-5
