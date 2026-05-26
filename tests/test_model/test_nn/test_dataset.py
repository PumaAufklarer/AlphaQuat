"""Tests for SR dataset building."""

import numpy as np
import pandas as pd

from alpha_quat.model.nn.transformer.models.dataset import (
    _compute_norm_params,
    _normalize,
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
                    "resistance_5d": price * 1.05 if d < n_days - 5 else np.nan,
                    "resistance_20d": price * 1.10 if d < n_days - 20 else np.nan,
                    "resistance_60d": price * 1.20 if d < n_days - 60 else np.nan,
                    "support_5d": price * 0.95 if d < n_days - 5 else np.nan,
                    "support_20d": price * 0.90 if d < n_days - 20 else np.nan,
                    "support_60d": price * 0.80 if d < n_days - 60 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_compute_norm_params():
    df = _make_synthetic_alpha360(n_stocks=2, n_days=50)
    params = _compute_norm_params(df)
    for col in ["open", "high", "low", "close", "volume", "vwap"]:
        assert col in params
        assert len(params[col]) == 2


def test_normalize():
    df = _make_synthetic_alpha360(n_stocks=1, n_days=50)
    params = _compute_norm_params(df)
    normed = _normalize(df, params)
    for col in ["open", "close"]:
        assert abs(normed[col].mean()) < 0.1  # approximately zero mean


def test_build_sequences_shape():
    df = _make_synthetic_alpha360(n_stocks=2, n_days=100)
    params = _compute_norm_params(df)
    normed = _normalize(df, params)
    X, Y, M = _build_sequences(
        normed, seq_length=20, stride=10, n_bins=10, price_range=0.10
    )
    assert X.ndim == 3
    assert X.shape[1] == 20
    assert X.shape[2] == 6
    assert Y.ndim == 2
    assert Y.shape[1] == 6
    assert Y.dtype == np.int64
    assert M.shape == (Y.shape[0], 6)
    assert M.dtype == bool
    assert np.isfinite(X).all()
    for i in range(Y.shape[0]):
        for j in range(6):
            if M[i, j]:
                assert 0 <= Y[i, j] < 10
