from alpha_quat.experiment.config import ExperimentConfig


class TestExperimentConfigDefaults:
    def test_default_values(self):
        cfg = ExperimentConfig(name="test", mode="regression")
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
        assert cfg.n_tile == 10
        assert cfg.label_gain is None

    def test_custom_values(self):
        cfg = ExperimentConfig(
            name="test_custom",
            mode="regression",
            num_leaves=63,
            learning_rate=0.1,
            n_trials=100,
            tune=False,
            feature_names=["KLEN35", "KMID5"],
        )
        assert cfg.num_leaves == 63
        assert cfg.learning_rate == 0.1
        assert cfg.n_trials == 100
        assert cfg.tune is False
        assert cfg.feature_names == ["KLEN35", "KMID5"]
