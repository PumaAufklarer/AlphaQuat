from alpha_quat.features.factor import Factor, compile


class TestFactor:
    def test_basic_fields(self):
        f = Factor(
            name="KMID", expression="REF($close, 5) / $close - 1", category="momentum"
        )
        assert f.name == "KMID"
        assert f.expression == "REF($close, 5) / $close - 1"
        assert f.category == "momentum"

    def test_depends_on_parsed_from_expression(self):
        f = Factor(
            name="f_001", expression="REF($close, 5) / $close - 1", category="momentum"
        )
        assert "$close" in f.depends_on

    def test_depends_on_other_factor(self):
        f = Factor(name="f_010", expression="STD(f_001, 20)", category="volatility")
        assert "f_001" in f.depends_on

    def test_depends_on_multiple(self):
        f = Factor(
            name="f_050", expression="CORR($close, $volume, 10)", category="correlation"
        )
        assert "$close" in f.depends_on
        assert "$volume" in f.depends_on
        assert len(f.depends_on) == 2


class TestCompile:
    def test_ref(self):
        assert compile("REF($close, 5)") == "LAG(close, 5) OVER w_time"

    def test_ref_default_lookback(self):
        assert compile("REF($close, 1)") == "LAG(close, 1) OVER w_time"

    def test_mean(self):
        result = compile("MEAN($close, 20)")
        assert "AVG(close) OVER (" in result
        assert "w_time" in result
        assert "ROWS BETWEEN 19 PRECEDING AND CURRENT ROW" in result

    def test_std(self):
        result = compile("STD($close, 10)")
        assert "STDDEV_SAMP(close) OVER (" in result
        assert "ROWS BETWEEN 9 PRECEDING AND CURRENT ROW" in result

    def test_sum(self):
        result = compile("SUM($volume, 5)")
        assert "SUM(volume) OVER (" in result
        assert "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW" in result

    def test_max_min(self):
        assert "MAX(close) OVER (" in compile("MAX($close, 10)")
        assert "MIN(close) OVER (" in compile("MIN($close, 10)")

    def test_corr(self):
        result = compile("CORR($close, $volume, 10)")
        assert "CORR(close, volume) OVER (" in result
        assert "ROWS BETWEEN 9 PRECEDING AND CURRENT ROW" in result

    def test_delta(self):
        result = compile("DELTA($close, 5)")
        assert "close - LAG(close, 5) OVER w_time" in result

    def test_rank(self):
        result = compile("RANK(f_001)")
        assert "RANK() OVER (PARTITION BY trade_date ORDER BY f_001)" in result

    def test_rank_with_compound_argument(self):
        result = compile("RANK(REF($close, 1) / $close - 1)")
        assert "RANK() OVER (PARTITION BY trade_date ORDER BY" in result
        assert "LAG(close, 1) OVER w_time / close - 1" in result

    def test_rank_with_nested_parens(self):
        result = compile("RANK(CORR($close, $volume, 10))")
        assert "RANK() OVER (PARTITION BY trade_date ORDER BY" in result
        assert "CORR(close, volume) OVER" in result

    def test_quantile(self):
        result = compile("QUANTILE(f_001, 10)")
        assert "NTILE(10) OVER (PARTITION BY trade_date ORDER BY f_001)" in result

    def test_quantile_with_compound_argument(self):
        result = compile("QUANTILE(MEAN($close, 5) / $close, 10)")
        assert "NTILE(10) OVER (PARTITION BY trade_date ORDER BY" in result
        assert "AVG(close) OVER" in result

    def test_vwap(self):
        result = compile("$vwap")
        assert result == "vwap"

    def test_arithmetic(self):
        result = compile("REF($close, 1) / $close - 1")
        assert "LAG(close, 1) OVER w_time / close - 1" == result

    def test_raw_field_passthrough(self):
        assert compile("$open") == "open"
        assert compile("$high") == "high"
        assert compile("$low") == "low"
        assert compile("$volume") == "volume"
        assert compile("$amount") == "amount"

    def test_factor_reference_passthrough(self):
        assert compile("f_001") == "f_001"
        assert compile("f_050") == "f_050"

    def test_ema(self):
        result = compile("EMA($close, 5)")
        assert "SUM(close * POW" in result
        assert "OVER (w_time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)" in result
        assert "__p5" in result

    def test_reg_slope(self):
        result = compile("REG_SLOPE($close, 10)")
        assert "REGR_SLOPE(close, CAST(__rn AS DOUBLE))" in result
        assert "OVER (w_time ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)" in result

    def test_rsi(self):
        result = compile("RSI($close, 14)")
        assert "100.0 - 100.0" in result
        assert "__diff > 0" in result
        assert "OVER (w_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)" in result
