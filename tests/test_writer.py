"""Tests for Parquet writer module."""

import pandas as pd
from alpha_quat.data.writer import ParquetWriter


def test_overwrite_writes_parquet_file(tmp_path):
    writer = ParquetWriter()
    df = pd.DataFrame(
        {"ts_code": ["000001.SZ", "000002.SZ"], "name": ["平安银行", "万科A"]}
    )
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


class TestMerge:
    def test_merge_writes_new_file(self, tmp_path):
        writer = ParquetWriter()
        df = pd.DataFrame({"ts_code": ["A", "B"], "f_001": [1.0, 2.0]})
        path = tmp_path / "output.parquet"
        writer.merge(df, path)
        assert path.exists()
        result = pd.read_parquet(path)
        assert list(result.columns) == ["ts_code", "f_001"]
        assert len(result) == 2

    def test_merge_joins_with_existing_file(self, tmp_path):
        writer = ParquetWriter()
        path = tmp_path / "output.parquet"

        existing = pd.DataFrame({"ts_code": ["A", "B"], "f_001": [1.0, 2.0]})
        existing.to_parquet(path, index=False)

        new = pd.DataFrame({"ts_code": ["A", "B"], "f_010": [3.0, 4.0]})
        writer.merge(new, path)

        result = pd.read_parquet(path)
        assert "f_001" in result.columns
        assert "f_010" in result.columns
        assert list(result["f_001"]) == [1.0, 2.0]
        assert list(result["f_010"]) == [3.0, 4.0]

    def test_merge_preserves_ts_code_mismatch(self, tmp_path):
        writer = ParquetWriter()
        path = tmp_path / "output.parquet"

        existing = pd.DataFrame({"ts_code": ["A", "B", "C"], "f_001": [1.0, 2.0, 3.0]})
        existing.to_parquet(path, index=False)

        new = pd.DataFrame({"ts_code": ["B", "C", "D"], "f_010": [4.0, 5.0, 6.0]})
        writer.merge(new, path)

        result = pd.read_parquet(path)
        assert len(result) == 4  # A, B, C, D
        assert set(result["ts_code"]) == {"A", "B", "C", "D"}
