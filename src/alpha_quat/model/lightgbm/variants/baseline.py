from abc import ABC, abstractmethod
from pathlib import Path

import lightgbm as lgb

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.experiment.registry import ExperimentRegistry
from alpha_quat.model.data import DatasetBuilder
from alpha_quat.model.lightgbm.evaluate import LightGBMEvaluator

_H = {"5d": "5", "20d": "20", "60d": "60"}


class LightGBMBasePipeline(ABC):
    mode: str = ""

    def __init__(self) -> None:
        self.evaluator = LightGBMEvaluator()

    @abstractmethod
    def _train(self, data, config: ExperimentConfig) -> dict[str, lgb.Booster]: ...

    def run(self, data_dir: Path, config: ExperimentConfig) -> dict:
        builder = DatasetBuilder(data_dir)
        data = builder.build(
            config.train_start,
            config.train_end,
            config.val_start,
            config.val_end,
            feature_names=config.feature_names,
            lambdarank=(config.mode == "lambdarank"),
        )

        models = self._train(data, config)

        exp_dir = data_dir / "models" / "experiments" / config.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        config.save(exp_dir / "experiment.yaml")

        results = {}
        for suffix, model in models.items():
            model.save_model(str(exp_dir / f"lightgbm_model_{suffix}.txt"))
            eval_result = self.evaluator.evaluate(
                model=model,
                X_val=data.X_val,
                y_val=getattr(data, f"y_val_{_H.get(suffix, suffix)}"),
                val_dates=data.val_dates,
                val_codes=data.val_codes,
                best_params={},
                feature_names=config.feature_names,
                label_name=f"ret_{suffix}",
            )
            results[suffix] = eval_result

        ExperimentRegistry(data_dir).register(config)

        return results
