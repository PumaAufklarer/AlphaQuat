import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.model.lightgbm.pipeline import run_variant

RNG = np.random.RandomState(42)


def _make_synthetic_data(data_dir: Path):
    ts_codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"]

    train_dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]
    val_dates = ["20240109", "20240110", "20240111", "20240116", "20240117", "20240118"]
    # 70 trading days margin for max_offset=60 forward labels
    margin_dates = [
        f"2024{m:02d}{d:02d}"
        for m, d in [
            (1, 12),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 29),
            (1, 30),
            (1, 31),
            (2, 1),
            (2, 2),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 19),
            (2, 20),
            (2, 21),
            (2, 22),
            (2, 23),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (3, 1),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
            (3, 18),
            (3, 19),
            (3, 20),
            (3, 21),
            (3, 22),
            (3, 25),
            (3, 26),
            (3, 27),
            (3, 28),
            (3, 29),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 15),
            (4, 16),
            (4, 17),
            (4, 18),
            (4, 19),
            (4, 22),
            (4, 23),
            (4, 24),
            (4, 25),
            (4, 26),
            (4, 29),
            (4, 30),
        ]
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
        close_vals = RNG.uniform(5, 50, len(ts_codes))
        df = pd.DataFrame(
            {
                "ts_code": ts_codes,
                "trade_date": d,
                "open": close_vals * RNG.uniform(0.98, 1.02, len(ts_codes)),
                "high": close_vals + RNG.uniform(0, 2, len(ts_codes)),
                "low": close_vals - RNG.uniform(0, 2, len(ts_codes)),
                "close": close_vals,
            }
        )
        path = f"{d[:4]}_{d[4:6]}_{d[6:8]}.parquet"
        df.to_parquet(daily_dir / path)

    sb = pd.DataFrame(
        {
            "ts_code": ts_codes,
            "market": ["主板"] * len(ts_codes),
            "list_status": ["L"] * len(ts_codes),
            "industry": ["银行"] * len(ts_codes),
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

            config = ExperimentConfig(
                name="test_run",
                mode="regression",
                train_start="20240102",
                train_end="20240108",
                val_start="20240109",
                val_end="20240118",
                tune=False,
                n_estimators=10,
                num_leaves=5,
            )

            results = run_variant(data_dir, config)

            assert "5d" in results
            assert "20d" in results
            assert "60d" in results
            assert results["5d"].mse >= 0
            assert results["20d"].mse >= 0
            assert results["60d"].mse >= 0
            assert len(results["5d"].top5_features) == 5

            exp_dir = data_dir / "models" / "experiments" / "test_run"
            assert (exp_dir / "lightgbm_model_5d.txt").exists()
            assert (exp_dir / "lightgbm_model_20d.txt").exists()
            assert (exp_dir / "lightgbm_model_60d.txt").exists()
            assert (exp_dir / "experiment.yaml").exists()
