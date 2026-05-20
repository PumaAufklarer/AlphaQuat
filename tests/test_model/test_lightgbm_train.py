import numpy as np
import pandas as pd

from alpha_quat.model.lightgbm.config import LightGBMConfig
from alpha_quat.model.lightgbm.train import LightGBMTrainer

RNG = np.random.RandomState(42)


class TestLightGBMTrainer:
    def test_train_without_tune(self):
        cfg = LightGBMConfig(tune=False, n_estimators=10, num_leaves=5)
        trainer = LightGBMTrainer(cfg)

        n_samples = 200
        group_size = 40
        n_groups = n_samples // group_size
        X = pd.DataFrame(
            {
                "feat1": RNG.randn(n_samples),
                "feat2": RNG.randn(n_samples),
            }
        )
        y = pd.Series(RNG.randint(0, 5, n_samples))
        groups = [group_size] * n_groups

        model, params = trainer.train(X, y, "test_label", groups=groups)

        assert params["num_leaves"] == 5
        assert params["n_estimators"] == 10
        assert model is not None
        assert model.params["objective"] == "lambdarank"
        assert "ndcg_eval_at" in model.params

    def test_train_with_tune_small_search(self):
        cfg = LightGBMConfig(tune=True, n_trials=3, n_estimators=10, num_leaves=5)
        trainer = LightGBMTrainer(cfg)

        n_samples = 200
        group_size = 40
        n_groups = n_samples // group_size
        X = pd.DataFrame(
            {
                "feat1": RNG.randn(n_samples),
                "feat2": RNG.randn(n_samples),
            }
        )
        y = pd.Series(RNG.randint(0, 5, n_samples))
        groups = [group_size] * n_groups

        model, params = trainer.train(X, y, "test_label_tune", groups=groups)

        assert model is not None
        assert "num_leaves" in params
        assert "learning_rate" in params
        assert model.params["objective"] == "lambdarank"
