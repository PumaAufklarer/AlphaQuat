import lightgbm as lgb

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.model.lightgbm.train import LightGBMTrainer
from alpha_quat.model.lightgbm.variants import register
from alpha_quat.model.lightgbm.variants.baseline import LightGBMBasePipeline


@register
class QuantilePipeline(LightGBMBasePipeline):
    mode = "quantile"

    def _train(self, data, config: ExperimentConfig) -> dict[str, lgb.Booster]:
        trainer = LightGBMTrainer.from_config(config)
        X = data.X_train
        alphas = config.quantile_alphas or [0.1, 0.5, 0.9]
        models = {}
        for h_name in ("5d", "20d", "60d"):
            y = getattr(data, f"y_train_{h_name}")
            for alpha in alphas:
                suffix = f"{h_name}_alpha_{alpha}"
                model, _ = trainer.train(X, y, label_name=suffix, quantile_alpha=alpha)
                models[suffix] = model
        return models
