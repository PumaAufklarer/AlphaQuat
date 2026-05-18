import pytest
from alpha_quat.features.factor import Factor
from alpha_quat.features.registry import FactorRegistry


class TestFactorRegistry:
    def test_register_and_list(self):
        reg = FactorRegistry(name="test")
        f1 = Factor(
            name="f_001", expression="REF($close, 1) / $close - 1", category="momentum"
        )
        reg.register(f1)
        assert "f_001" in reg.factors
        assert reg.factors["f_001"] is f1

    def test_topological_order_simple(self):
        reg = FactorRegistry(name="test")
        f1 = Factor(name="f_001", expression="$close", category="price")
        f2 = Factor(name="f_002", expression="REF(f_001, 1)", category="momentum")
        f3 = Factor(name="f_003", expression="STD(f_002, 5)", category="volatility")
        reg.register(f1)
        reg.register(f2)
        reg.register(f3)
        order = reg.topological_order()
        names = [f.name for f in order]
        assert names.index("f_001") < names.index("f_002")
        assert names.index("f_002") < names.index("f_003")

    def test_topological_order_no_deps(self):
        reg = FactorRegistry(name="test")
        f1 = Factor(name="f_001", expression="$close", category="price")
        f2 = Factor(name="f_002", expression="$volume", category="volume")
        reg.register(f1)
        reg.register(f2)
        order = reg.topological_order()
        names = [f.name for f in order]
        assert len(names) == 2

    def test_cycle_detection(self):
        reg = FactorRegistry(name="test")
        f1 = Factor(name="f_001", expression="f_002", category="test")
        f2 = Factor(name="f_002", expression="f_001", category="test")
        reg.register(f1)
        reg.register(f2)
        with pytest.raises(ValueError, match="cycle"):
            reg.topological_order()

    def test_min_lookback(self):
        reg = FactorRegistry(name="test")
        f1 = Factor(name="f_001", expression="REF($close, 5)", category="momentum")
        f2 = Factor(name="f_002", expression="MEAN($volume, 20)", category="volume")
        f3 = Factor(name="f_003", expression="STD($close, 10)", category="volatility")
        reg.register(f1)
        reg.register(f2)
        reg.register(f3)
        assert reg.min_lookback() == 20

    def test_min_lookback_no_operators(self):
        reg = FactorRegistry(name="test")
        reg.register(Factor(name="f_001", expression="$close", category="price"))
        assert reg.min_lookback() == 0
