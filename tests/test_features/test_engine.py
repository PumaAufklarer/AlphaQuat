import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from alpha_quat.features.engine import FeatureEngine
from alpha_quat.features.factor import Factor
from alpha_quat.features.registry import FactorRegistry


def make_daily_parquet(tmp_path, data: list[dict]):
    """Helper: write synthetic daily parquet files partitioned by trade_date."""
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    df = pd.DataFrame(data)
    df["trade_date"] = df["trade_date"].astype(str)
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, str(daily_dir), partition_cols=["trade_date"])


def make_daily_basic_parquet(tmp_path, data: list[dict]):
    """Helper: write synthetic daily_basic parquet files partitioned by trade_date."""
    db_dir = tmp_path / "daily_basic"
    db_dir.mkdir()
    if not data:
        return
    df = pd.DataFrame(data)
    df["trade_date"] = df["trade_date"].astype(str)
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, str(db_dir), partition_cols=["trade_date"])


class TestFeatureEngine:
    def test_compute_single_factor(self, tmp_path):
        # 10 days of data for 2 stocks
        daily_data = []
        for stock in ["000001.SZ", "000002.SZ"]:
            for i, date in enumerate(
                [
                    "20240102",
                    "20240103",
                    "20240104",
                    "20240105",
                    "20240108",
                    "20240109",
                    "20240110",
                    "20240111",
                    "20240112",
                    "20240115",
                ]
            ):
                daily_data.append(
                    {
                        "ts_code": stock,
                        "trade_date": date,
                        "open": 10.0 + i + (0.1 if stock == "000002.SZ" else 0),
                        "high": 11.0 + i,
                        "low": 9.0 + i,
                        "close": 10.5 + i + (0.1 if stock == "000002.SZ" else 0),
                        "vol": 1000000.0,
                        "amount": 10500000.0,
                    }
                )
        make_daily_parquet(tmp_path, daily_data)
        make_daily_basic_parquet(tmp_path, [])

        reg = FactorRegistry(name="test")
        reg.register(
            Factor(
                name="f_001",
                expression="REF($close, 1) / $close - 1",
                category="momentum",
            )
        )

        engine = FeatureEngine(data_dir=tmp_path)
        result = engine.compute(reg, trade_date="20240115")

        assert "ts_code" in result.columns
        assert "trade_date" in result.columns
        assert "f_001" in result.columns
        assert len(result) == 2  # two stocks
        assert all(result["trade_date"] == "20240115")

    def test_compute_multiple_factors(self, tmp_path):
        daily_data = []
        for stock in ["000001.SZ"]:
            for i, date in enumerate(
                [
                    "20240102",
                    "20240103",
                    "20240104",
                    "20240105",
                    "20240108",
                    "20240109",
                ]
            ):
                daily_data.append(
                    {
                        "ts_code": stock,
                        "trade_date": date,
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.0,
                        "close": 10.0 + i,
                        "vol": 1000000.0,
                        "amount": 10000000.0,
                    }
                )
        make_daily_parquet(tmp_path, daily_data)
        make_daily_basic_parquet(tmp_path, [])

        reg = FactorRegistry(name="test")
        reg.register(Factor(name="f_001", expression="$close", category="price"))
        reg.register(
            Factor(name="f_002", expression="REF(f_001, 1)", category="momentum")
        )
        reg.register(
            Factor(name="f_003", expression="MEAN(f_001, 3)", category="momentum")
        )

        engine = FeatureEngine(data_dir=tmp_path)
        result = engine.compute(reg, trade_date="20240109")

        assert "f_001" in result.columns
        assert "f_002" in result.columns
        assert "f_003" in result.columns
        assert len(result) == 1

    def test_compute_with_daily_basic_join(self, tmp_path):
        daily_data = []
        daily_basic_data = []
        for stock in ["000001.SZ", "000002.SZ"]:
            for i, date in enumerate(["20240102", "20240103", "20240104", "20240105"]):
                daily_data.append(
                    {
                        "ts_code": stock,
                        "trade_date": date,
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.0,
                        "close": 10.0 + i,
                        "vol": 1000000.0,
                        "amount": 10000000.0,
                    }
                )
                daily_basic_data.append(
                    {
                        "ts_code": stock,
                        "trade_date": date,
                        "total_mv": 1e8,
                        "circ_mv": 5e7,
                    }
                )
        make_daily_parquet(tmp_path, daily_data)
        make_daily_basic_parquet(tmp_path, daily_basic_data)

        reg = FactorRegistry(name="test")
        reg.register(Factor(name="f_001", expression="$close", category="price"))

        engine = FeatureEngine(data_dir=tmp_path)
        result = engine.compute(reg, trade_date="20240105")

        assert len(result) == 2
        assert "total_mv" not in result.columns  # raw columns filtered out
