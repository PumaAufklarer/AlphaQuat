"""Tests for SR cache module — vectorized + multiprocess."""

import numpy as np
import pandas as pd

from alpha_quat.data.sr_cache import (
    _nearest_peak_and_dist,
    _process_one_stock,
    build_cache,
)


def test_nearest_peak_and_dist_basic():
    prices = np.array([10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0])
    verified = np.array([False, False, False, True, False, False, False])
    price_out, dist_out = _nearest_peak_and_dist(prices, verified, lookahead=5)
    assert np.isnan(price_out[3])  # peak day itself: future of itself not included
    assert price_out[0] == 13.0
    assert dist_out[0] == 3.0  # peak at idx 3, current 0 → 3 days ahead
    assert price_out[1] == 13.0
    assert dist_out[1] == 2.0
    assert price_out[2] == 13.0
    assert dist_out[2] == 1.0


def test_nearest_peak_lookahead_limit():
    prices = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 13.0, 12.5, 12.0])
    verified = np.array([False, False, False, False, False, True, False, False])
    price_out, dist_out = _nearest_peak_and_dist(prices, verified, lookahead=2)
    assert np.isnan(price_out[0])
    assert np.isnan(price_out[1])
    assert np.isnan(price_out[2])
    assert price_out[3] == 13.0
    assert dist_out[3] == 2.0
    assert price_out[4] == 13.0
    assert dist_out[4] == 1.0
    assert np.isnan(price_out[5])


def test_nearest_peak_no_peak():
    prices = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
    verified = np.zeros(5, dtype=bool)
    price_out, dist_out = _nearest_peak_and_dist(prices, verified, lookahead=5)
    assert np.all(np.isnan(price_out))
    assert np.all(np.isnan(dist_out))


def test_process_one_stock_column_names():
    n = 30
    rng = np.random.RandomState(42)
    base = 10.0 * (1 + 0.02 * np.sin(np.linspace(0, 4 * np.pi, n)))

    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * n,
            "trade_date": [f"2024{i + 1:02d}01" for i in range(n)],
            "open": base * (1 + rng.randn(n) * 0.005),
            "high": base * (1 + abs(rng.randn(n)) * 0.01 + 0.01),
            "low": base * (1 - abs(rng.randn(n)) * 0.01 - 0.01),
            "close": base,
            "volume": rng.randint(100000, 1000000, n),
            "amount": rng.randint(1000000, 10000000, n),
            "vwap": base,
        }
    )

    result = _process_one_stock(df)
    expected_cols = [
        "resistance_5d_price",
        "resistance_5d_dist",
        "resistance_20d_price",
        "resistance_20d_dist",
        "resistance_60d_price",
        "resistance_60d_dist",
        "support_5d_price",
        "support_5d_dist",
        "support_20d_price",
        "support_20d_dist",
        "support_60d_price",
        "support_60d_dist",
    ]
    for col in expected_cols:
        assert col in result.columns

    assert np.isnan(result.iloc[-1]["resistance_5d_price"])
    assert np.isnan(result.iloc[-1]["resistance_5d_dist"])


def test_build_cache_synthetic_data(tmp_path):
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir(parents=True)

    ts_codes = ["000001.SZ", "000002.SZ"]
    rng = np.random.RandomState(42)

    for d in range(1, 30):
        date_str = f"2024_01_{d:02d}"
        rows = []
        for code in ts_codes:
            rows.append(
                {
                    "ts_code": code,
                    "open": 10.0 + rng.randn() * 0.1,
                    "high": 10.5 + rng.randn() * 0.1,
                    "low": 9.5 + rng.randn() * 0.1,
                    "close": 10.0 + rng.randn() * 0.1,
                    "vol": rng.randint(100000, 1000000),
                    "amount": rng.randint(1000000, 10000000),
                }
            )
        pd.DataFrame(rows).to_parquet(daily_dir / f"{date_str}.parquet")

    written = build_cache(tmp_path)
    assert written > 0

    cache_dir = tmp_path / "alpha360"
    assert cache_dir.exists()
    files = list(cache_dir.glob("*.parquet"))
    assert len(files) == written

    sample = pd.read_parquet(files[0])
    assert "resistance_5d_price" in sample.columns
    assert "resistance_5d_dist" in sample.columns
