
from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.experiment.registry import ExperimentRegistry


class TestExperimentRegistry:
    def test_registry_empty(self, tmp_path):
        reg = ExperimentRegistry(tmp_path)
        assert reg.list_experiments() == []
        assert reg.latest() is None

    def test_registry_register(self, tmp_path):
        reg = ExperimentRegistry(tmp_path)
        cfg = ExperimentConfig(name="exp1", mode="regression")
        reg.register(cfg)
        entries = reg.list_experiments()
        assert len(entries) == 1
        assert entries[0]["name"] == "exp1"
        assert entries[0]["mode"] == "regression"
        assert "created" in entries[0]

    def test_registry_register_multiple(self, tmp_path):
        reg = ExperimentRegistry(tmp_path)
        reg.register(ExperimentConfig(name="a", mode="regression"))
        reg.register(ExperimentConfig(name="b", mode="quantile"))
        reg.register(ExperimentConfig(name="c", mode="lambdarank"))
        entries = reg.list_experiments()
        assert len(entries) == 3
        assert [e["name"] for e in entries] == ["a", "b", "c"]

    def test_registry_latest(self, tmp_path):
        reg = ExperimentRegistry(tmp_path)
        assert reg.latest() is None
        reg.register(ExperimentConfig(name="first", mode="regression"))
        reg.register(ExperimentConfig(name="second", mode="quantile"))
        latest = reg.latest()
        assert latest is not None
        assert latest["name"] == "second"
        assert latest["mode"] == "quantile"

    def test_registry_find(self, tmp_path):
        reg = ExperimentRegistry(tmp_path)
        reg.register(ExperimentConfig(name="alpha", mode="regression"))
        reg.register(ExperimentConfig(name="beta", mode="quantile"))
        found = reg.find("alpha")
        assert found is not None
        assert found["name"] == "alpha"
        assert found["mode"] == "regression"
        assert reg.find("gamma") is None

    def test_registry_persists_across_instances(self, tmp_path):
        reg1 = ExperimentRegistry(tmp_path)
        reg1.register(ExperimentConfig(name="persist", mode="meta"))
        reg2 = ExperimentRegistry(tmp_path)
        entries = reg2.list_experiments()
        assert len(entries) == 1
        assert entries[0]["name"] == "persist"
