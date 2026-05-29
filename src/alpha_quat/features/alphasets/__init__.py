"""Combined factor registry — merges all factor sets for training."""

from alpha_quat.features.alphasets.alpha158 import build_alpha158
from alpha_quat.features.alphasets.alpha_ext import build_alpha_ext
from alpha_quat.features.alphasets.alpha_fund import build_alpha_fund
from alpha_quat.features.registry import FactorRegistry


def build_alpha_combined() -> FactorRegistry:
    reg = build_alpha158()
    for f in build_alpha_ext().factors.values():
        reg.register(f)
    for f in build_alpha_fund().factors.values():
        reg.register(f)
    return reg


__all__ = [
    "build_alpha158",
    "build_alpha_ext",
    "build_alpha_fund",
    "build_alpha_combined",
    "FactorRegistry",
]
