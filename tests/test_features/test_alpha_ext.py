import pytest
from alpha_quat.features.alphasets.alpha_ext import build_alpha_ext


class TestAlphaExt:
    def test_all_factors_registered(self):
        reg = build_alpha_ext()
        assert reg.name == "alpha_ext"
        assert len(reg.factors) == 7

    def test_all_factors_compile(self):
        from alpha_quat.features.factor import compile

        reg = build_alpha_ext()
        for name, factor in reg.factors.items():
            try:
                result = compile(factor.expression)
                assert result, f"compile({name}) returned empty"
            except Exception as e:
                pytest.fail(f"compile({name}) failed: {e}")

    def test_no_cycles(self):
        reg = build_alpha_ext()
        ordered = reg.topological_order()
        assert len(ordered) == len(reg.factors)

    def test_all_deps_exist(self):
        reg = build_alpha_ext()
        factor_names = set(reg.factors.keys())
        raw_fields = {"$open", "$high", "$low", "$close", "$volume", "$amount", "$vwap"}
        for factor in reg.factors.values():
            for dep in factor.depends_on:
                if dep.startswith("$"):
                    assert dep in raw_fields, (
                        f"{factor.name} depends on unknown raw field {dep}"
                    )
                else:
                    assert dep in factor_names, (
                        f"{factor.name} depends on unknown factor {dep}"
                    )

    def test_min_lookback_consistent(self):
        reg = build_alpha_ext()
        lookback = reg.min_lookback()
        assert lookback >= 0
        assert lookback <= 60
