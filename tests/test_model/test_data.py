# tests/test_model/test_data.py
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_quat.model.data import DatasetBuilder, DatasetResult


def _generate_dates(start: str, n: int) -> list[str]:
    """Generate n consecutive business dates starting from start (YYYYMMDD)."""
    return [d.strftime("%Y%m%d") for d in pd.bdate_range(start=start, periods=n)]


MARGIN_DATES_70 = _generate_dates("20240112", 80)


def _make_features(data_dir: Path, dates: list[str], ts_codes: list[str]):
    feat_dir = data_dir / "features"
    feat_dir.mkdir()
    for d in dates:
        df = pd.DataFrame(
            {
                "ts_code": ts_codes,
                "trade_date": d,
                "KMID": np.random.randn(len(ts_codes)),
                "KLEN": np.random.randn(len(ts_codes)),
            }
        )
        df.to_parquet(feat_dir / f"{d}.parquet")


def _make_daily(
    data_dir: Path, dates: list[str], ts_codes: list[str], close_col: str = "close"
):
    daily_dir = data_dir / "daily"
    daily_dir.mkdir()
    rng = np.random.RandomState(42)
    for d in dates:
        close_vals = rng.uniform(5, 50, len(ts_codes))
        df = pd.DataFrame(
            {
                "ts_code": ts_codes,
                "trade_date": d,
                "open": close_vals * rng.uniform(0.98, 1.02, len(ts_codes)),
                "high": close_vals + rng.uniform(0, 2, len(ts_codes)),
                "low": close_vals - rng.uniform(0, 2, len(ts_codes)),
                close_col: close_vals,
            }
        )
        path = f"{d[:4]}_{d[4:6]}_{d[6:8]}.parquet"
        df.to_parquet(daily_dir / path)


def _make_stock_basic(data_dir: Path, ts_codes: list[str]):
    df = pd.DataFrame(
        {
            "ts_code": ts_codes,
            "market": ["主板"] * len(ts_codes),
            "list_status": ["L"] * len(ts_codes),
            "industry": ["银行"] * len(ts_codes),
        }
    )
    df.to_parquet(data_dir / "stock_basic.parquet")


def _make_trade_cal(data_dir: Path, dates: list[str]):
    df = pd.DataFrame(
        {
            "cal_date": dates,
            "is_open": [1] * len(dates),
        }
    )
    df.to_parquet(data_dir / "trade_cal.parquet")


class TestDatasetBuilder:
    def test_build_returns_correct_split_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            ts_codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"]

            train_dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]
            val_dates = [
                "20240109",
                "20240110",
                "20240111",
                "20240116",
                "20240117",
                "20240118",
            ]
            margin_dates = MARGIN_DATES_70
            all_feat_dates = train_dates + val_dates + margin_dates
            all_daily_dates = train_dates + val_dates + margin_dates

            _make_features(data_dir, all_feat_dates, ts_codes)
            _make_daily(data_dir, all_daily_dates, ts_codes)
            _make_stock_basic(data_dir, ts_codes)
            _make_trade_cal(data_dir, all_daily_dates)

            builder = DatasetBuilder(data_dir)
            result = builder.build("20240102", "20240108", "20240109", "20240118")

            assert isinstance(result, DatasetResult)
            assert result.X_train.shape == (5, 2)
            assert result.X_val.shape == (15, 2)  # 5 stocks * 3 dates
            assert len(result.y_train_5) == 5
            assert len(result.y_val_5) == 15
            assert len(result.y_train_20) == 5
            assert len(result.y_val_20) == 15

    def test_drops_nan_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            ts_codes = ["000001.SZ", "000002.SZ"]

            train_dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]
            margin_dates = MARGIN_DATES_70
            all_feat_dates = train_dates + margin_dates
            all_daily_dates = train_dates + margin_dates

            _make_features(data_dir, all_feat_dates, ts_codes)
            _make_daily(data_dir, all_daily_dates, ts_codes)
            _make_stock_basic(data_dir, ts_codes)
            _make_trade_cal(data_dir, all_daily_dates)

            builder = DatasetBuilder(data_dir)
            result = builder.build("20240102", "20240108", "20240108", "20240108")

            assert not result.X_train.isna().any().any()
            assert not result.y_train_5.isna().any()
            assert not result.y_train_20.isna().any()

    def test_feature_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            ts_codes = ["000001.SZ", "000002.SZ"]

            train_dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]
            val_dates = ["20240109"]
            margin_dates = MARGIN_DATES_70
            all_feat_dates = train_dates + val_dates + margin_dates
            all_daily_dates = train_dates + val_dates + margin_dates

            _make_features(data_dir, all_feat_dates, ts_codes)
            _make_daily(data_dir, all_daily_dates, ts_codes)
            _make_stock_basic(data_dir, ts_codes)
            _make_trade_cal(data_dir, all_daily_dates)

            builder = DatasetBuilder(data_dir)
            result = builder.build(
                "20240102", "20240108", "20240109", "20240109", feature_names=["KMID"]
            )

            assert list(result.X_train.columns) == ["KMID"]
            assert list(result.X_val.columns) == ["KMID"]

    def test_excludes_st_and_non_main_board(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            ts_codes = ["000001.SZ", "000002.SZ", "300001.SZ"]

            train_dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]
            margin_dates = MARGIN_DATES_70
            all_feat_dates = train_dates + margin_dates
            all_daily_dates = train_dates + margin_dates

            _make_features(data_dir, all_feat_dates, ts_codes)
            _make_daily(data_dir, all_daily_dates, ts_codes)
            sb = pd.DataFrame(
                {
                    "ts_code": ts_codes,
                    "market": ["主板", "主板", "创业板"],
                    "list_status": ["L", "L", "L"],
                    "industry": ["银行", "银行", "科技"],
                }
            )
            sb.to_parquet(data_dir / "stock_basic.parquet")
            st_dir = data_dir / "stock_st"
            st_dir.mkdir()
            for d in train_dates + margin_dates:
                st = pd.DataFrame({"ts_code": ["000002.SZ"], "trade_date": [d]})
                st_path = st_dir / f"{d[:4]}_{d[4:6]}_{d[6:8]}.parquet"
                st.to_parquet(st_path)
            _make_trade_cal(data_dir, all_daily_dates)

            builder = DatasetBuilder(data_dir)
            result = builder.build("20240102", "20240108", "20240108", "20240108")

            codes_in_train = set(result.train_codes)
            assert "300001.SZ" not in codes_in_train
            assert "000002.SZ" not in codes_in_train
            assert "000001.SZ" in codes_in_train
