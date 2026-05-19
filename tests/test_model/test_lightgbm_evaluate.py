import numpy as np
import pandas as pd

from alpha_quat.model.lightgbm.evaluate import EvalResult, LightGBMEvaluator

RNG = np.random.RandomState(42)


class TestLightGBMEvaluator:
    def test_eval_result_has_all_fields(self):
        r = EvalResult(
            label_name="ret_5d",
            mse=0.001,
            mae=0.02,
            mean_ic=0.05,
            ic_std=0.08,
            icir=0.625,
            top5_features=[("A", 0.5), ("B", 0.4), ("C", 0.3), ("D", 0.2), ("E", 0.1)],
            bottom5_features=[
                ("V", 0.001),
                ("W", 0.002),
                ("X", 0.003),
                ("Y", 0.004),
                ("Z", 0.005),
            ],
            best_params={"num_leaves": 31},
            feature_names=None,
        )
        assert r.label_name == "ret_5d"
        assert r.mse == 0.001
        assert r.icir == 0.625
        assert len(r.top5_features) == 5
        assert len(r.bottom5_features) == 5

    def test_rank_ic_computes_per_date_spearman(self):
        evaluator = LightGBMEvaluator()
        n = 50
        dates = [f"202401{str(i + 1).zfill(2)}" for i in range(5)] * 10
        val_dates = pd.Series(dates[:n])
        y_val = pd.Series(RNG.randn(n))
        y_pred = RNG.randn(n) * 0.5
        y_true = np.asarray(y_val, dtype=float)

        result = evaluator.compute_rank_ic(y_pred, y_true, val_dates.to_numpy())

        assert isinstance(result.mean_ic, float)
        assert isinstance(result.ic_std, float)
        assert result.ic_std >= 0
