from alpha_quat.strategy.signals.variants.baseline import BaseMLSignal

VARIANTS: dict[str, type[BaseMLSignal]] = {}


def register(cls):
    VARIANTS[cls.mode] = cls
    return cls


from alpha_quat.strategy.signals.variants import regression_signal  # noqa: F401, E402
from alpha_quat.strategy.signals.variants import quantile_signal  # noqa: F401, E402
from alpha_quat.strategy.signals.variants import lambdarank_signal  # noqa: F401, E402
