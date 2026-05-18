"""Tests for StockStSource."""

from pathlib import Path
from alpha_quat.data.sources.stock_st import StockStSource


def test_stock_st_api_name():
    src = StockStSource()
    assert src.api_name == "stock_st"


def test_stock_st_partition_by():
    src = StockStSource()
    assert src.partition_by == "date"


def test_stock_st_get_params():
    src = StockStSource()
    params = src.get_params(trade_date="20260115")
    assert params == {"trade_date": "20260115"}


def test_stock_st_path_for():
    src = StockStSource()
    path = src.path_for(data_dir=Path("/data"), trade_date="2026_05_18")
    assert path == Path("/data/stock_st/2026_05_18.parquet")
