import tempfile
from pathlib import Path

import pandas as pd

from alpha_quat.backtest.filters import build_universe


class TestBuildUniverse:
    def test_excludes_st_stocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            sb = pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                    "market": ["主板", "主板", "主板"],
                    "list_status": ["L", "L", "L"],
                }
            )
            sb.to_parquet(data_dir / "stock_basic.parquet")
            st_dir = data_dir / "stock_st"
            st_dir.mkdir()
            st = pd.DataFrame({"ts_code": ["000002.SZ"], "trade_date": ["20240115"]})
            st.to_parquet(st_dir / "2024_01_15.parquet")
            universe = build_universe("20240115", data_dir)
            assert "000001.SZ" in universe
            assert "000002.SZ" not in universe
            assert "000003.SZ" in universe

    def test_excludes_non_main_board(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            sb = pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "300001.SZ", "688001.SH"],
                    "market": ["主板", "创业板", "科创板"],
                    "list_status": ["L", "L", "L"],
                }
            )
            sb.to_parquet(data_dir / "stock_basic.parquet")
            (data_dir / "stock_st").mkdir()
            universe = build_universe("20240115", data_dir)
            assert universe == {"000001.SZ"}

    def test_no_st_data_returns_main_board(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            sb = pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "market": ["主板", "主板"],
                    "list_status": ["L", "L"],
                }
            )
            sb.to_parquet(data_dir / "stock_basic.parquet")
            (data_dir / "stock_st").mkdir()
            universe = build_universe("20240115", data_dir)
            assert universe == {"000001.SZ", "000002.SZ"}
