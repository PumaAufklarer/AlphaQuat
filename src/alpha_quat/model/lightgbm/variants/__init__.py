from alpha_quat.model.lightgbm.variants.baseline import LightGBMBasePipeline

VARIANTS: dict[str, type[LightGBMBasePipeline]] = {}


def register(cls):
    VARIANTS[cls.mode] = cls
    return cls
