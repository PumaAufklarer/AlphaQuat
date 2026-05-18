"""Tests for DataSource base class."""

from pathlib import Path
import pytest
from alpha_quat.data.source import DataSource


class FakeFullSource(DataSource):
    api_name = "full_test"
    partition_by = "none"
    fields = "ts_code,name"
    def get_params(self, trade_date=None):
        return {"list_status": "L"}


class FakeDateSource(DataSource):
    api_name = "date_test"
    partition_by = "date"
    fields = "ts_code,close"
    def get_params(self, trade_date):
        return {"trade_date": trade_date}


def test_full_source_path_for():
    src = FakeFullSource()
    path = src.path_for(data_dir=Path("/data"))
    assert path == Path("/data/full_test.parquet")


def test_date_source_path_for():
    src = FakeDateSource()
    path = src.path_for(data_dir=Path("/data"), trade_date="2026_05_18")
    assert path == Path("/data/date_test/2026_05_18.parquet")


def test_full_source_path_for_ignores_trade_date():
    src = FakeFullSource()
    path = src.path_for(data_dir=Path("/data"), trade_date="2026_05_18")
    assert path == Path("/data/full_test.parquet")


def test_date_source_path_for_no_date_returns_dir():
    src = FakeDateSource()
    path = src.path_for(data_dir=Path("/data"))
    assert path == Path("/data/date_test")


def test_datasource_is_abstract():
    with pytest.raises(TypeError):
        DataSource()  # type: ignore
