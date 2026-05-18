"""Tests for Pipeline module."""

import pandas as pd

from alpha_quat.data.pipeline import Pipeline
from alpha_quat.data.metadata import MetadataManager
from alpha_quat.data.writer import ParquetWriter
from alpha_quat.data.sources.stock_basic import StockBasicSource
from alpha_quat.data.sources.daily import DailySource
from alpha_quat.data.sources.trade_cal import TradeCalSource


class FakeFetcher:
    def __init__(self, calls=None):
        self.calls = calls or []
        self._idx = 0
        self.query_log = []

    def query(self, api_name, **params):
        if self._idx < len(self.calls):
            df = self.calls[self._idx]
        else:
            df = pd.DataFrame()
        self.query_log.append((api_name, params))
        self._idx += 1
        return df


def test_pipeline_runs_full_source(tmp_path):
    data_dir = tmp_path / "data"
    db_path = str(tmp_path / "registry.db")

    fetcher = FakeFetcher(
        calls=[
            pd.DataFrame({"ts_code": ["000001.SZ"], "name": ["平安银行"]}),
        ]
    )
    metadata = MetadataManager(db_path)
    writer = ParquetWriter()
    pipeline = Pipeline(
        data_dir=data_dir, fetcher=fetcher, metadata=metadata, writer=writer
    )

    pipeline.run_full_source(StockBasicSource())

    parquet_path = data_dir / "stock_basic.parquet"
    assert parquet_path.exists()
    result = pd.read_parquet(parquet_path)
    assert len(result) == 1
    assert result.iloc[0]["ts_code"] == "000001.SZ"


def test_pipeline_run_incremental_source_no_prior_data(tmp_path):
    data_dir = tmp_path / "data"
    db_path = str(tmp_path / "registry.db")

    trade_cal_path = data_dir / "trade_cal.parquet"
    trade_cal_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "cal_date": ["20260112", "20260113", "20260114", "20260115"],
            "is_open": [1, 1, 1, 0],
        }
    ).to_parquet(trade_cal_path)

    fetcher = FakeFetcher(
        calls=[
            pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]}),
            pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.5]}),
            pd.DataFrame({"ts_code": ["000001.SZ"], "close": [11.0]}),
        ]
    )
    metadata = MetadataManager(db_path)
    writer = ParquetWriter()
    pipeline = Pipeline(
        data_dir=data_dir, fetcher=fetcher, metadata=metadata, writer=writer
    )

    results = pipeline.run_incremental_source(DailySource())

    assert results["success"] == 3
    assert results["failed"] == 0
    assert (data_dir / "daily" / "2026_01_12.parquet").exists()
    assert (data_dir / "daily" / "2026_01_13.parquet").exists()
    assert (data_dir / "daily" / "2026_01_14.parquet").exists()
    # 20260115 is not open, should be skipped
    assert not (data_dir / "daily" / "2026_01_15.parquet").exists()


def test_pipeline_run_incremental_source_with_prior_data(tmp_path):
    data_dir = tmp_path / "data"
    db_path = str(tmp_path / "registry.db")

    trade_cal_path = data_dir / "trade_cal.parquet"
    trade_cal_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "cal_date": ["20260112", "20260113", "20260114"],
            "is_open": [1, 1, 1],
        }
    ).to_parquet(trade_cal_path)

    metadata = MetadataManager(db_path)
    metadata.insert("daily", "2026-01-12", "data/daily/2026_01_12.parquet", 50)

    fetcher = FakeFetcher(
        calls=[
            pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.5]}),
            pd.DataFrame({"ts_code": ["000001.SZ"], "close": [11.0]}),
        ]
    )
    writer = ParquetWriter()
    pipeline = Pipeline(
        data_dir=data_dir, fetcher=fetcher, metadata=metadata, writer=writer
    )

    results = pipeline.run_incremental_source(DailySource())

    # Should skip 20260112 (already pulled), only pull 20260113-14
    assert results["success"] == 2
    assert not (data_dir / "daily" / "2026_01_12.parquet").exists()
    assert (data_dir / "daily" / "2026_01_13.parquet").exists()
    assert (data_dir / "daily" / "2026_01_14.parquet").exists()
    assert len(fetcher.query_log) == 2


def test_pipeline_incremental_skips_when_trade_cal_missing(tmp_path):
    data_dir = tmp_path / "data"
    db_path = str(tmp_path / "registry.db")

    fetcher = FakeFetcher()
    metadata = MetadataManager(db_path)
    writer = ParquetWriter()
    pipeline = Pipeline(
        data_dir=data_dir, fetcher=fetcher, metadata=metadata, writer=writer
    )

    results = pipeline.run_incremental_source(DailySource())

    assert results["success"] == 0
    assert results["failed"] == 0
    assert results["message"] == "trade_cal.parquet not found"
    assert "errors" in results


def test_pipeline_error_handling_continues_after_failure(tmp_path):
    data_dir = tmp_path / "data"
    db_path = str(tmp_path / "registry.db")

    trade_cal_path = data_dir / "trade_cal.parquet"
    trade_cal_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "cal_date": ["20260112", "20260113", "20260114"],
            "is_open": [1, 1, 1],
        }
    ).to_parquet(trade_cal_path)

    class FailingFetcher:
        def __init__(self):
            self.call_count = 0
            self.query_log = []

        def query(self, api_name, **params):
            self.call_count += 1
            self.query_log.append((api_name, params))
            if self.call_count == 2:
                raise RuntimeError("network failure")
            return pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]})

    fetcher = FailingFetcher()
    metadata = MetadataManager(db_path)
    writer = ParquetWriter()
    pipeline = Pipeline(
        data_dir=data_dir, fetcher=fetcher, metadata=metadata, writer=writer
    )

    results = pipeline.run_incremental_source(DailySource())

    assert results["success"] == 2
    assert results["failed"] == 1
    assert len(results["errors"]) == 1
    assert "20260113" in results["errors"][0]


def test_pipeline_run_all(tmp_path):
    data_dir = tmp_path / "data"
    db_path = str(tmp_path / "registry.db")

    trade_cal_path = data_dir / "trade_cal.parquet"
    trade_cal_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "cal_date": ["20260112", "20260113"],
            "is_open": [1, 1],
        }
    ).to_parquet(trade_cal_path)

    calls = [
        pd.DataFrame({"ts_code": ["000001.SZ"]}),
        pd.DataFrame({"cal_date": ["20260112", "20260113"], "is_open": [1, 1]}),
        pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]}),
        pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.5]}),
    ]
    fetcher = FakeFetcher(calls=calls)
    metadata = MetadataManager(db_path)
    writer = ParquetWriter()
    pipeline = Pipeline(
        data_dir=data_dir, fetcher=fetcher, metadata=metadata, writer=writer
    )

    pipeline.run([StockBasicSource(), TradeCalSource(), DailySource()])

    assert (data_dir / "stock_basic.parquet").exists()
    assert (data_dir / "daily" / "2026_01_12.parquet").exists()
    assert (data_dir / "daily" / "2026_01_13.parquet").exists()
