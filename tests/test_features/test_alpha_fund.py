import pytest
from alpha_quat.features.alphasets.alpha_fund import build_alpha_fund


class TestAlphaFund:
    def test_all_factors_registered(self):
        reg = build_alpha_fund()
        assert reg.name == "alpha_fund"
        assert len(reg.factors) == 8

    def test_all_factors_compile(self):
        from alpha_quat.features.factor import compile

        reg = build_alpha_fund()
        for name, factor in reg.factors.items():
            try:
                result = compile(factor.expression)
                assert result, f"compile({name}) returned empty"
            except Exception as e:
                pytest.fail(f"compile({name}) failed: {e}")

    def test_no_cycles(self):
        reg = build_alpha_fund()
        ordered = reg.topological_order()
        assert len(ordered) == len(reg.factors)

    def test_all_deps_exist(self):
        reg = build_alpha_fund()
        factor_names = set(reg.factors.keys())
        raw_fields = {
            "$open",
            "$high",
            "$low",
            "$close",
            "$volume",
            "$amount",
            "$vwap",
            "$pe_ttm",
            "$pb",
            "$total_mv",
            "$turnover_rate",
            "$volume_ratio",
        }
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
        reg = build_alpha_fund()
        lookback = reg.min_lookback()
        assert lookback >= 0
        assert lookback <= 750
