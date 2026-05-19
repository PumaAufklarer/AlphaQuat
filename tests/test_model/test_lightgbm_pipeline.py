import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_quat.model.lightgbm.config import LightGBMConfig
from alpha_quat.model.lightgbm.pipeline import LightGBMPipeline

RNG = np.random.RandomState(42)


def _make_synthetic_data(data_dir: Path):
    ts_codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"]

    train_dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]
    val_dates = ["20240109", "20240110", "20240111"]
    margin_dates = [
        "20240112",
        "20240115",
        "20240116",
        "20240117",
        "20240118",
        "20240119",
        "20240122",
        "20240123",
        "20240124",
        "20240125",
        "20240126",
        "20240129",
        "20240130",
        "20240131",
        "20240201",
        "20240202",
        "20240205",
        "20240206",
        "20240207",
        "20240208",
    ]

    feat_dir = data_dir / "features"
    feat_dir.mkdir()
    all_feat_dates = train_dates + val_dates + margin_dates
    feature_cols = ["KMID", "KLEN", "KMID2", "KLEN2", "KMID3"]
    for d in all_feat_dates:
        data = {"ts_code": ts_codes, "trade_date": d}
        for col in feature_cols:
            data[col] = RNG.randn(len(ts_codes))
        df = pd.DataFrame(data)
        df.to_parquet(feat_dir / f"{d}.parquet")

    daily_dir = data_dir / "daily"
    daily_dir.mkdir()
    all_dates = train_dates + val_dates + margin_dates
    for d in all_dates:
        df = pd.DataFrame(
            {
                "ts_code": ts_codes,
                "trade_date": d,
                "close": RNG.uniform(5, 50, len(ts_codes)),
            }
        )
        path = f"{d[:4]}_{d[4:6]}_{d[6:8]}.parquet"
        df.to_parquet(daily_dir / path)

    sb = pd.DataFrame(
        {
            "ts_code": ts_codes,
            "market": ["主板"] * len(ts_codes),
            "list_status": ["L"] * len(ts_codes),
        }
    )
    sb.to_parquet(data_dir / "stock_basic.parquet")

    (data_dir / "stock_st").mkdir()

    cal = pd.DataFrame(
        {
            "cal_date": all_dates,
            "is_open": [1] * len(all_dates),
        }
    )
    cal.to_parquet(data_dir / "trade_cal.parquet")


class TestLightGBMPipeline:
    def test_pipeline_runs_and_saves_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _make_synthetic_data(data_dir)

            config = LightGBMConfig(
                train_start="20240102",
                train_end="20240108",
                val_start="20240109",
                val_end="20240111",
                tune=False,
                n_estimators=10,
                num_leaves=5,
            )

            pipeline = LightGBMPipeline(data_dir, config)
            results = pipeline.run()

            assert "ret_5d" in results
            assert "ret_20d" in results
            assert results["ret_5d"].mse >= 0
            assert results["ret_20d"].mse >= 0
            assert len(results["ret_5d"].top5_features) == 5

            models_dir = data_dir / "models"
            assert (models_dir / "lightgbm_model_5d.txt").exists()
            assert (models_dir / "lightgbm_model_20d.txt").exists()
            assert (models_dir / "results.json").exists()

            with open(models_dir / "results.json") as f:
                results_json = json.load(f)
            assert results_json["model_type"] == "lightgbm"
            assert "ret_5d" in results_json
            assert "ret_20d" in results_json
            assert "mse" in results_json["ret_5d"]
            assert "mean_ic" in results_json["ret_5d"]
            assert "top5_features" in results_json["ret_5d"]
