from alpha_quat.strategy.signals.variants import VARIANTS


def test_quantile_is_registered():
    assert "quantile" in VARIANTS


def test_quantile_instantiate():
    cls = VARIANTS["quantile"]
    assert cls.mode == "quantile"
