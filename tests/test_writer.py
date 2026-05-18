"""Tests for Parquet writer module."""

import pandas as pd
from alpha_quat.data.writer import ParquetWriter


def test_overwrite_writes_parquet_file(tmp_path):
    writer = ParquetWriter()
    df = pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"], "name": ["平安银行", "万科A"]})
    path = tmp_path / "stock_basic.parquet"
    writer.overwrite(df, path)
    assert path.exists()
    result = pd.read_parquet(path)
    assert len(result) == 2
    assert list(result["ts_code"]) == ["000001.SZ", "000002.SZ"]


def test_overwrite_replaces_existing_file(tmp_path):
    writer = ParquetWriter()
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2, 3]})
    path = tmp_path / "test.parquet"
    writer.overwrite(df1, path)
    writer.overwrite(df2, path)
    result = pd.read_parquet(path)
    assert len(result) == 2


def test_write_creates_date_partitioned_file(tmp_path):
    writer = ParquetWriter()
    df = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.5]})
    base_dir = tmp_path / "daily"
    writer.write(df, base_dir, trade_date="2026_05_18")
    expected = base_dir / "2026_05_18.parquet"
    assert expected.exists()
    result = pd.read_parquet(expected)
    assert len(result) == 1
    assert result.iloc[0]["close"] == 10.5


def test_write_creates_parent_directory(tmp_path):
    writer = ParquetWriter()
    df = pd.DataFrame({"a": [1]})
    base_dir = tmp_path / "nested" / "deep" / "daily"
    writer.write(df, base_dir, trade_date="2026_05_18")
    expected = base_dir / "2026_05_18.parquet"
    assert expected.exists()
