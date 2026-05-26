from alpha_quat.strategy.signals.variants import VARIANTS


def test_lambdarank_is_registered():
    assert "lambdarank" in VARIANTS


def test_lambdarank_instantiate():
    cls = VARIANTS["lambdarank"]
    assert cls.mode == "lambdarank"
