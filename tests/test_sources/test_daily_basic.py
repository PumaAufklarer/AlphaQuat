"""Tests for DailyBasicSource."""
from pathlib import Path
from alpha_quat.data.sources.daily_basic import DailyBasicSource

def test_daily_basic_api_name():
    src = DailyBasicSource()
    assert src.api_name == "daily_basic"

def test_daily_basic_partition_by():
    src = DailyBasicSource()
    assert src.partition_by == "date"

def test_daily_basic_get_params():
    src = DailyBasicSource()
    params = src.get_params(trade_date="20260115")
    assert params == {"trade_date": "20260115"}

def test_daily_basic_path_for():
    src = DailyBasicSource()
    path = src.path_for(data_dir=Path("/data"), trade_date="2026_05_18")
    assert path == Path("/data/daily_basic/2026_05_18.parquet")
