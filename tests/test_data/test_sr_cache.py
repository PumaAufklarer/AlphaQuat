"""Tests for SR cache module."""

import numpy as np
import pandas as pd

from alpha_quat.data.sr_cache import (
    _find_resistance_levels,
    _find_support_levels,
    _process_one_stock,
)


def test_find_resistance_local_peak():
    high = np.array([10.0, 11.0, 12.0, 13.0, 12.5, 12.0, 11.5, 11.0, 10.5, 10.0])
    level = _find_resistance_levels(high, n=0, horizon=5)
    # Day 1-5: 11, 12, 13, 12.5, 12. Day 3 (index 3) = 13 is local max in [1,5]
    # It has rejection (12.5, 12 after it)
    assert level == 13.0


def test_find_resistance_no_peak():
    high = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 12.5])
    level = _find_resistance_levels(high, n=0, horizon=5)
    # Monotonic uptrend — no local peak (because no decline after)
    assert np.isnan(level)


def test_find_support_local_trough():
    low = np.array([10.0, 9.5, 9.0, 8.5, 9.0, 9.5, 10.0, 10.5])
    level = _find_support_levels(low, n=0, horizon=5)
    # Day 3 = 8.5 is local min, bounce follows
    assert level == 8.5


def test_find_support_no_trough():
    low = np.array([10.0, 9.5, 9.0, 8.5, 8.0, 7.5])
    level = _find_support_levels(low, n=0, horizon=5)
    # Monotonic downtrend — no bounce
    assert np.isnan(level)


def test_process_one_stock():
    dates = [f"2024{i:02d}{d:02d}" for i in range(1, 5) for d in range(1, 8)][:28]
    n = len(dates)
    rng = np.random.RandomState(42)

    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * n,
            "trade_date": dates,
            "open": rng.uniform(10, 12, n),
            "high": rng.uniform(11, 14, n),
            "low": rng.uniform(9, 11, n),
            "close": rng.uniform(10, 13, n),
            "volume": rng.randint(100000, 1000000, n),
            "vwap": rng.uniform(10, 13, n),
        }
    )

    result = _process_one_stock(df)

    assert result is not None
    assert len(result) == n
    assert "resistance_5d" in result.columns
    assert "resistance_20d" in result.columns
    assert "resistance_60d" in result.columns
    assert "support_5d" in result.columns
    assert "support_20d" in result.columns
    assert "support_60d" in result.columns

    # Last day has no future data → all NaN
    last = result.iloc[-1]
    assert np.isnan(last["resistance_5d"])
    assert np.isnan(last["support_5d"])
