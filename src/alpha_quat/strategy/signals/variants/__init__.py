from alpha_quat.strategy.signals.variants.baseline import BaseMLSignal
from alpha_quat.model.variant_registry import VariantRegistry

_REGISTRY = VariantRegistry[BaseMLSignal]("signal")
VARIANTS = _REGISTRY.as_dict()

register = _REGISTRY.register


from alpha_quat.strategy.signals.variants import regression_signal  # noqa: F401, E402
from alpha_quat.strategy.signals.variants import quantile_signal  # noqa: F401, E402
from alpha_quat.strategy.signals.variants import lambdarank_signal  # noqa: F401, E402
from alpha_quat.strategy.signals.variants import transformer_sr_signal  # noqa: F401, E402
