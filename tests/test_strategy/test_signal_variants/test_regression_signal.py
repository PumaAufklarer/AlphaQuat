from alpha_quat.strategy.signals.variants import VARIANTS


def test_regression_is_registered():
    assert "regression" in VARIANTS


def test_regression_instantiate():
    cls = VARIANTS["regression"]
    assert cls.mode == "regression"
