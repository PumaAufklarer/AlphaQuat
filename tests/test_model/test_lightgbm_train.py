import numpy as np
import pandas as pd

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.model.lightgbm.train import LightGBMTrainer

RNG = np.random.RandomState(42)


class TestLightGBMTrainer:
    def test_train_without_tune(self):
        cfg = ExperimentConfig(
            name="test_train",
            mode="regression",
            tune=False,
            n_estimators=10,
            num_leaves=5,
        )
        trainer = LightGBMTrainer(cfg)

        n_samples = 200
        X = pd.DataFrame(
            {
                "feat1": RNG.randn(n_samples),
                "feat2": RNG.randn(n_samples),
            }
        )
        y = pd.Series(RNG.randn(n_samples))

        model, params = trainer.train(X, y, "test_label")

        assert params["num_leaves"] == 5
        assert params["n_estimators"] == 10
        assert model is not None
        assert model.params["objective"] == "regression"

    def test_train_with_tune_small_search(self):
        cfg = ExperimentConfig(
            name="test_train_tune",
            mode="regression",
            tune=True,
            n_trials=3,
            n_estimators=10,
            num_leaves=5,
        )
        trainer = LightGBMTrainer(cfg)

        n_samples = 200
        X = pd.DataFrame(
            {
                "feat1": RNG.randn(n_samples),
                "feat2": RNG.randn(n_samples),
            }
        )
        y = pd.Series(RNG.randn(n_samples))

        model, params = trainer.train(X, y, "test_label_tune")

        assert model is not None
        assert "num_leaves" in params
        assert "learning_rate" in params
        assert model.params["objective"] == "regression"
