"""Tests for DailySource."""

from pathlib import Path
from alpha_quat.data.sources.daily import DailySource


def test_daily_api_name():
    src = DailySource()
    assert src.api_name == "daily"


def test_daily_partition_by():
    src = DailySource()
    assert src.partition_by == "date"


def test_daily_get_params():
    src = DailySource()
    params = src.get_params(trade_date="20260115")
    assert params == {"trade_date": "20260115"}


def test_daily_path_for():
    src = DailySource()
    path = src.path_for(data_dir=Path("/data"), trade_date="2026_05_18")
    assert path == Path("/data/daily/2026_05_18.parquet")
