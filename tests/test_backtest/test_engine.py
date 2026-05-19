import pytest
import tempfile
from pathlib import Path

import pandas as pd

from alpha_quat.backtest.config import BacktestConfig
from alpha_quat.backtest.engine import BacktestEngine


def _make_minimal_data(data_dir, dates):
    daily_dir = data_dir / "daily"
    daily_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()
    st_dir = data_dir / "stock_st"
    st_dir.mkdir(parents=True)
    for d in dates:
        d_int = int(d)
        d_path = f"{d[:4]}_{d[4:6]}_{d[6:8]}"
        daily = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [d_int, d_int],
                "open": [10.0, 20.0],
                "high": [10.5, 20.5],
                "low": [9.5, 19.5],
                "close": [10.2, 20.2],
                "pre_close": [10.0, 20.0],
                "change": [0.2, 0.2],
                "pct_chg": [2.0, 1.0],
                "vol": [1e5, 2e5],
                "amount": [1.02e6, 4.04e6],
            }
        )
        daily.to_parquet(daily_dir / f"{d_path}.parquet")
        features = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [d, d],
                "KLEN35": [1.00, 1.00],
                "KLEN36": [1.00, 1.00],
            }
        )
        features.to_parquet(features_dir / f"{d}.parquet")
        pd.DataFrame(columns=["ts_code"]).to_parquet(st_dir / f"{d_path}.parquet")
    sb = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "market": ["主板", "主板"],
            "list_status": ["L", "L"],
        }
    )
    sb.to_parquet(data_dir / "stock_basic.parquet")
    cal = pd.DataFrame({"cal_date": dates, "is_open": [1] * len(dates)})
    cal.to_parquet(data_dir / "trade_cal.parquet")


class TestBacktestEngine:
    def test_runs_and_produces_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _make_minimal_data(data_dir, ["20240115", "20240116", "20240117"])
            config = BacktestConfig(
                start_date="20240115",
                end_date="20240117",
                initial_capital=100000,
                monthly_addition=0,
            )
            engine = BacktestEngine(config, data_dir)
            result = engine.run()
            assert len(result["snapshots"]) == 3
            assert result["metrics"]["total_invested"] == 100000
            assert result["metrics"]["final_value"] > 0

    def test_monthly_addition(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _make_minimal_data(data_dir, ["20240115", "20240201"])
            config = BacktestConfig(
                start_date="20240115",
                end_date="20240201",
                initial_capital=100000,
                monthly_addition=5000,
            )
            engine = BacktestEngine(config, data_dir)
            result = engine.run()
            assert result["metrics"]["total_invested"] == 105000
            assert len(result["snapshots"]) == 2

    def test_missing_trade_cal_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            engine = BacktestEngine(BacktestConfig(), data_dir)
            with pytest.raises(FileNotFoundError, match="trade_cal"):
                engine.run()
