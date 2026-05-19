# tests/test_model/test_lightgbm_config.py
from alpha_quat.model.lightgbm.config import LightGBMConfig


class TestLightGBMConfig:
    def test_default_values(self):
        cfg = LightGBMConfig()
        assert cfg.train_start == "20240401"
        assert cfg.train_end == "20250430"
        assert cfg.val_start == "20250501"
        assert cfg.val_end == "20260430"
        assert cfg.num_leaves == 31
        assert cfg.learning_rate == 0.05
        assert cfg.n_estimators == 200
        assert cfg.feature_fraction == 0.8
        assert cfg.bagging_fraction == 0.8
        assert cfg.early_stopping_rounds == 20
        assert cfg.random_state == 42
        assert cfg.n_jobs == -1
        assert cfg.verbosity == -1
        assert cfg.n_trials == 50
        assert cfg.tune is True
        assert cfg.feature_names is None

    def test_custom_values(self):
        cfg = LightGBMConfig(
            train_start="20230101",
            train_end="20231231",
            num_leaves=63,
            learning_rate=0.1,
            n_trials=100,
            tune=False,
            feature_names=["KLEN35", "KMID5"],
        )
        assert cfg.train_start == "20230101"
        assert cfg.train_end == "20231231"
        assert cfg.num_leaves == 63
        assert cfg.learning_rate == 0.1
        assert cfg.n_trials == 100
        assert cfg.tune is False
        assert cfg.feature_names == ["KLEN35", "KMID5"]
