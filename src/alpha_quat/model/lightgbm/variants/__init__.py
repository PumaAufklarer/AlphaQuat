from alpha_quat.model.lightgbm.variants.baseline import LightGBMBasePipeline
from alpha_quat.model.variant_registry import VariantRegistry

_REGISTRY = VariantRegistry[LightGBMBasePipeline]("LightGBM")
VARIANTS = _REGISTRY.as_dict()

register = _REGISTRY.register


# Import concrete variant submodules to populate VARIANTS via @register
from alpha_quat.model.lightgbm.variants import lambdarank  # noqa: F401, E402

from alpha_quat.model.lightgbm.variants import quantile  # noqa: F401, E402
from alpha_quat.model.lightgbm.variants import regression  # noqa: F401, E402
