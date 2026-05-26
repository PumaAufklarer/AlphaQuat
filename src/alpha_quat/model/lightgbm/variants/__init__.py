from alpha_quat.model.lightgbm.variants.baseline import LightGBMBasePipeline

VARIANTS: dict[str, type[LightGBMBasePipeline]] = {}


def register(cls):
    VARIANTS[cls.mode] = cls
    return cls


# Import concrete variant submodules to populate VARIANTS via @register
from alpha_quat.model.lightgbm.variants import lambdarank  # noqa: F401, E402

from alpha_quat.model.lightgbm.variants import quantile  # noqa: F401, E402
from alpha_quat.model.lightgbm.variants import regression  # noqa: F401, E402
