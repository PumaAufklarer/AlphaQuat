from alpha_quat.strategy.signals.variants import VARIANTS


def test_meta_is_registered():
    assert "meta" in VARIANTS


def test_meta_instantiate():
    cls = VARIANTS["meta"]
    assert cls.mode == "meta"
