import pytest
import yaml
from alpha_quat.experiment.config import ExperimentConfig


class TestExperimentConfig:
    def test_experiment_config_defaults(self):
        cfg = ExperimentConfig(name="test", mode="regression")
        assert cfg.name == "test"
        assert cfg.mode == "regression"
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
        assert cfg.quantile_alphas is None

        assert isinstance(cfg.created_at, str)

    def test_experiment_config_yaml_roundtrip(self, tmp_path):
        path = tmp_path / "experiment.yaml"
        cfg = ExperimentConfig(
            name="my_exp",
            mode="quantile",
            train_start="20230101",
            train_end="20231231",
            val_start="20240101",
            val_end="20241231",
            num_leaves=63,
            learning_rate=0.1,
            n_estimators=500,
            feature_fraction=0.7,
            bagging_fraction=0.9,
            early_stopping_rounds=50,
            random_state=99,
            n_jobs=4,
            verbosity=1,
            n_trials=100,
            tune=False,
            feature_names=["KLEN35", "KMID5"],
            quantile_alphas=[0.1, 0.5, 0.9],
            created_at="2025-01-01T00:00:00",
        )
        cfg.save(path)

        assert path.exists()
        with open(path) as f:
            raw = yaml.safe_load(f)
        assert raw["name"] == "my_exp"
        assert raw["mode"] == "quantile"
        assert raw["num_leaves"] == 63

        loaded = ExperimentConfig.load(path)
        for key in (
            "name",
            "mode",
            "train_start",
            "train_end",
            "val_start",
            "val_end",
            "num_leaves",
            "learning_rate",
            "n_estimators",
            "feature_fraction",
            "bagging_fraction",
            "early_stopping_rounds",
            "random_state",
            "n_jobs",
            "verbosity",
            "n_trials",
            "tune",
            "feature_names",
            "quantile_alphas",
            "created_at",
        ):
            assert getattr(loaded, key) == getattr(cfg, key), f"Mismatch for {key}"

    def test_experiment_config_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ExperimentConfig.load("/nonexistent/path.yaml")
