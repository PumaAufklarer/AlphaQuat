import lightgbm as lgb

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.model.lightgbm.train import LightGBMTrainer
from alpha_quat.model.lightgbm.variants import register
from alpha_quat.model.lightgbm.variants.baseline import LightGBMBasePipeline


@register
class RegressionPipeline(LightGBMBasePipeline):
    mode = "regression"

    def _train(self, data, config: ExperimentConfig) -> dict[str, lgb.Booster]:
        trainer = LightGBMTrainer.from_config(config)
        X = data.X_train
        models = {}
        for h_name, h in [("5d", 5), ("20d", 20), ("60d", 60)]:
            y = getattr(data, f"y_train_{h}")
            model, _ = trainer.train(X, y, label_name=h_name)
            models[h_name] = model
        return models
