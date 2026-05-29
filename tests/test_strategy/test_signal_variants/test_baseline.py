import pytest

from alpha_quat.model.constants import ZERO_GAIN_FEATURES
from alpha_quat.strategy.signals.variants.baseline import BaseMLSignal
from alpha_quat.strategy.signals.variants import VARIANTS


def test_base_signal_is_abstract():
    with pytest.raises(TypeError):
        BaseMLSignal()


def test_register():
    assert "regression" in VARIANTS
    assert "quantile" in VARIANTS
    assert "lambdarank" in VARIANTS


def test_zero_gain_is_set():
    assert len(ZERO_GAIN_FEATURES) == 30
