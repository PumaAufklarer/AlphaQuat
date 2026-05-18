"""Tests for StkStSource."""

from pathlib import Path
from alpha_quat.data.sources.stk_st import StkStSource


def test_stk_st_api_name():
    src = StkStSource()
    assert src.api_name == "stk_st"


def test_stk_st_partition_by():
    src = StkStSource()
    assert src.partition_by == "date"


def test_stk_st_get_params():
    src = StkStSource()
    params = src.get_params(trade_date="20260115")
    assert params == {"trade_date": "20260115"}


def test_stk_st_path_for():
    src = StkStSource()
    path = src.path_for(data_dir=Path("/data"), trade_date="2026_05_18")
    assert path == Path("/data/stk_st/2026_05_18.parquet")
