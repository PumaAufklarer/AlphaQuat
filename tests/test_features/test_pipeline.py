import datetime

import pandas as pd

from alpha_quat.features.pipeline import FeaturePipeline
from alpha_quat.features.factor import Factor
from alpha_quat.features.registry import FactorRegistry
from alpha_quat.data.writer import ParquetWriter
from alpha_quat.data.metadata import MetadataManager


class FakeEngine:
    def compute_batch(self, registry, trade_dates):
        return {
            d: pd.DataFrame(
                {
                    "ts_code": [f"{d}_A", f"{d}_B"],
                    "trade_date": [d, d],
                    "f_001": [1.0, 2.0],
                }
            )
            for d in trade_dates
        }


class TestFeaturePipeline:
    def make_trade_cal(self, data_dir, dates):
        path = data_dir / "trade_cal.parquet"
        df = pd.DataFrame(
            {
                "exchange": "SSE",
                "cal_date": dates,
                "is_open": 1,
                "pretrade_date": "",
            }
        )
        df.to_parquet(path, index=False)

    def test_incremental_first_run(self, tmp_path):
        features_dir = tmp_path / "features"
        db_path = str(tmp_path / "registry.db")

        self.make_trade_cal(tmp_path, ["20240102", "20240103", "20240104"])

        engine = FakeEngine()
        writer = ParquetWriter()
        metadata = MetadataManager(db_path)

        reg = FactorRegistry(name="alpha158")
        reg.register(Factor(name="f_001", expression="$close", category="price"))

        pipeline = FeaturePipeline(
            data_dir=tmp_path,
            output_dir=features_dir,
            engine=engine,
            writer=writer,
            metadata=metadata,
        )
        result = pipeline.run(reg)

        assert result["success"] == 3
        for date in ["20240102", "20240103", "20240104"]:
            assert (features_dir / f"{date}.parquet").exists()
        assert metadata.get_last_date("alpha158") == datetime.date(2024, 1, 4)

    def test_incremental_skips_completed_dates(self, tmp_path):
        features_dir = tmp_path / "features"
        db_path = str(tmp_path / "registry.db")

        self.make_trade_cal(tmp_path, ["20240102", "20240103", "20240104"])

        engine = FakeEngine()
        writer = ParquetWriter()
        metadata = MetadataManager(db_path)

        metadata.insert("alpha158", "2024-01-02", "features/20240102.parquet", 2)
        metadata.insert("alpha158", "2024-01-03", "features/20240103.parquet", 2)

        reg = FactorRegistry(name="alpha158")
        reg.register(Factor(name="f_001", expression="$close", category="price"))

        pipeline = FeaturePipeline(
            data_dir=tmp_path,
            output_dir=features_dir,
            engine=engine,
            writer=writer,
            metadata=metadata,
        )
        result = pipeline.run(reg)

        assert result["success"] == 1

    def test_rebuild_deletes_all(self, tmp_path):
        features_dir = tmp_path / "features"
        db_path = str(tmp_path / "registry.db")

        self.make_trade_cal(tmp_path, ["20240102", "20240103"])

        engine = FakeEngine()
        writer = ParquetWriter()
        metadata = MetadataManager(db_path)

        metadata.insert("alpha158", "2024-01-02", "features/20240102.parquet", 2)

        reg = FactorRegistry(name="alpha158")
        reg.register(Factor(name="f_001", expression="$close", category="price"))

        pipeline = FeaturePipeline(
            data_dir=tmp_path,
            output_dir=features_dir,
            engine=engine,
            writer=writer,
            metadata=metadata,
        )
        result = pipeline.run(reg, rebuild=True)

        assert result["success"] == 2

    def test_error_tolerance_continues(self, tmp_path):
        features_dir = tmp_path / "features"
        db_path = str(tmp_path / "registry.db")

        self.make_trade_cal(tmp_path, ["20240102", "20240103", "20240104"])

        class FailingEngine:
            def compute_batch(self, registry, trade_dates):
                raise RuntimeError("simulated failure")

        engine = FailingEngine()
        writer = ParquetWriter()
        metadata = MetadataManager(db_path)

        reg = FactorRegistry(name="alpha158")
        reg.register(Factor(name="f_001", expression="$close", category="price"))

        pipeline = FeaturePipeline(
            data_dir=tmp_path,
            output_dir=features_dir,
            engine=engine,
            writer=writer,
            metadata=metadata,
        )
        result = pipeline.run(reg)

        assert result["success"] == 0
        assert result["failed"] == 3
        assert len(result["errors"]) == 1

    def test_missing_trade_cal_returns_early(self, tmp_path):
        features_dir = tmp_path / "features"
        db_path = str(tmp_path / "registry.db")

        engine = FakeEngine()
        writer = ParquetWriter()
        metadata = MetadataManager(db_path)

        reg = FactorRegistry(name="alpha158")
        reg.register(Factor(name="f_001", expression="$close", category="price"))

        pipeline = FeaturePipeline(
            data_dir=tmp_path,
            output_dir=features_dir,
            engine=engine,
            writer=writer,
            metadata=metadata,
        )
        result = pipeline.run(reg)

        assert result.get("message") is not None
