"""Tests for StockBasicSource."""

from pathlib import Path
from alpha_quat.data.sources.stock_basic import StockBasicSource


def test_stock_basic_api_name():
    src = StockBasicSource()
    assert src.api_name == "stock_basic"


def test_stock_basic_partition_by():
    src = StockBasicSource()
    assert src.partition_by == "none"


def test_stock_basic_get_params():
    src = StockBasicSource()
    params = src.get_params()
    assert params == {"list_status": "L"}


def test_stock_basic_path_for():
    src = StockBasicSource()
    path = src.path_for(data_dir=Path("/data"))
    assert path == Path("/data/stock_basic.parquet")
