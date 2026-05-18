"""Tests for TradeCalSource."""

from pathlib import Path
from alpha_quat.data.sources.trade_cal import TradeCalSource


def test_trade_cal_api_name():
    src = TradeCalSource()
    assert src.api_name == "trade_cal"


def test_trade_cal_partition_by():
    src = TradeCalSource()
    assert src.partition_by == "none"


def test_trade_cal_get_params():
    src = TradeCalSource()
    params = src.get_params()
    assert params == {"exchange": "SSE"}


def test_trade_cal_path_for():
    src = TradeCalSource()
    path = src.path_for(data_dir=Path("/data"))
    assert path == Path("/data/trade_cal.parquet")
