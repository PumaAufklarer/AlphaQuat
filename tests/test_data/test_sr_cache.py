"""Tests for SR cache module — vectorized implementation."""

import numpy as np
import pandas as pd

from alpha_quat.data.sr_cache import (
    _nearest_peak_price,
    _find_sr_for_stock,
    build_cache,
)


def test_nearest_peak_price_basic():
    prices = np.array([10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0])
    verified = np.array([False, False, False, True, False, False, False])
    result = _nearest_peak_price(prices, verified, lookahead=5)
    # Day 3 is the only peak (13.0)
    # Day 0,1,2 look forward → see 13.0 at position 3 (within 5)
    # Day 3+ sees nothing after
    assert np.isnan(result[3])
    assert result[0] == 13.0
    assert result[1] == 13.0
    assert result[2] == 13.0
    assert np.isnan(result[4])
    assert np.isnan(result[5])


def test_nearest_peak_lookahead_limit():
    prices = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 13.0, 12.5, 12.0])
    verified = np.array([False, False, False, False, False, True, False, False])
    result = _nearest_peak_price(prices, verified, lookahead=2)
    # Peak at position 5, lookahead=2 means only days 3,4 see it
    assert np.isnan(result[0])  # too far
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    assert result[3] == 13.0  # position 5 is 2 ahead, within lookahead
    assert result[4] == 13.0  # position 5 is 1 ahead
    assert np.isnan(result[5])


def test_nearest_peak_no_peak():
    prices = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
    verified = np.zeros(5, dtype=bool)
    result = _nearest_peak_price(prices, verified, lookahead=5)
    assert np.all(np.isnan(result))


def test_find_sr_for_stock_known_pattern():
    """Test with a known ZigZag pattern: clear local peaks and troughs."""
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
            "vwap": base,
        }
    )

    result = _find_sr_for_stock(df)
    assert len(result) == n
    for col in [
        "resistance_5d",
        "resistance_20d",
        "resistance_60d",
        "support_5d",
        "support_20d",
        "support_60d",
    ]:
        assert col in result.columns

    # Last day has NaN (no forward data)
    assert np.isnan(result.iloc[-1]["resistance_5d"])
    assert np.isnan(result.iloc[-1]["support_5d"])

    # Some middle days should have values
    mids = result.iloc[5 : n - 5]
    assert mids["resistance_5d"].notna().any()
    assert mids["support_5d"].notna().any()
    assert mids["resistance_20d"].notna().any()
    assert mids["resistance_60d"].notna().any()


def test_resistance_above_current():
    """Resistance levels should be above current close."""
    n = 30
    rng = np.random.RandomState(42)
    base = 10.0 * (1 + 0.02 * np.sin(np.linspace(0, 4 * np.pi, n)))

    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * n,
            "trade_date": [f"2024{i + 1:02d}01" for i in range(n)],
            "open": base,
            "high": base * 1.02,
            "low": base * 0.98,
            "close": base,
            "volume": rng.randint(100000, 1000000, n),
            "vwap": base,
        }
    )

    result = _find_sr_for_stock(df)
    res_5d = result["resistance_5d"].dropna()
    if len(res_5d) > 0:
        close_vals = result.loc[res_5d.index, "close"].values
        for i, idx in enumerate(res_5d.index):
            assert res_5d.iloc[i] >= close_vals[i] * 0.99


def test_build_cache_synthetic_data(tmp_path):
    """Integration test: write synthetic daily data, build cache."""
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
