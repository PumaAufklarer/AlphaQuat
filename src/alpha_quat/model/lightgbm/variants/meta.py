import lightgbm as lgb

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.model.lightgbm.variants import register
from alpha_quat.model.lightgbm.variants.quantile import QuantilePipeline


@register
class MetaPipeline(QuantilePipeline):
    mode = "meta"

    def _train(self, data, config: ExperimentConfig) -> dict[str, lgb.Booster]:
        base_models = super()._train(data, config)
        return base_models
