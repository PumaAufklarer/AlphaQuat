import json
import logging
from pathlib import Path

from alpha_quat.model.data import DatasetBuilder
from alpha_quat.model.lightgbm.config import LightGBMConfig
from alpha_quat.model.lightgbm.evaluate import LightGBMEvaluator
from alpha_quat.model.lightgbm.train import LightGBMTrainer

logger = logging.getLogger(__name__)


class LightGBMPipeline:
    def __init__(self, data_dir: Path, config: LightGBMConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        self.builder = DatasetBuilder(self.data_dir)
        self.trainer = LightGBMTrainer(config)
        self.evaluator = LightGBMEvaluator()

    def run(self) -> dict[str, object]:
        logger.info("Building dataset...")
        data = self.builder.build(
            self.config.train_start,
            self.config.train_end,
            self.config.val_start,
            self.config.val_end,
            feature_names=self.config.feature_names,
        )

        logger.info(
            "Training data: %d samples, Validation data: %d samples",
            len(data.X_train),
            len(data.X_val),
        )

        logger.info("Training model_5d...")
        model_5d, params_5d = self.trainer.train(data.X_train, data.y_train_5, "ret_5d")

        logger.info("Training model_20d...")
        model_20d, params_20d = self.trainer.train(
            data.X_train, data.y_train_20, "ret_20d"
        )

        logger.info("Evaluating model_5d...")
        result_5d = self.evaluator.evaluate(
            model_5d,
            data.X_val,
            data.y_val_5,
            data.val_dates,
            data.val_codes,
            params_5d,
            self.config.feature_names,
            "ret_5d",
        )

        logger.info("Evaluating model_20d...")
        result_20d = self.evaluator.evaluate(
            model_20d,
            data.X_val,
            data.y_val_20,
            data.val_dates,
            data.val_codes,
            params_20d,
            self.config.feature_names,
            "ret_20d",
        )

        self._save_models(model_5d, model_20d)
        self._save_results(result_5d, result_20d)
        self._print_summary(result_5d, result_20d)

        return {"ret_5d": result_5d, "ret_20d": result_20d}

    def _save_models(self, model_5d, model_20d):
        output_dir = self.data_dir / "models"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_5d.save_model(str(output_dir / "lightgbm_model_5d.txt"))
        model_20d.save_model(str(output_dir / "lightgbm_model_20d.txt"))
        logger.info("Models saved to %s", output_dir)

    def _save_results(self, result_5d, result_20d):
        output_dir = self.data_dir / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        def _make_json(result):
            return {
                "mse": result.mse,
                "mae": result.mae,
                "mean_ic": result.mean_ic,
                "ic_std": result.ic_std,
                "icir": result.icir,
                "top5_features": result.top5_features,
                "bottom5_features": result.bottom5_features,
                "best_params": result.best_params,
                "feature_names": result.feature_names,
            }

        output = {
            "model_type": "lightgbm",
            "ret_5d": _make_json(result_5d),
            "ret_20d": _make_json(result_20d),
            "config": {
                "train_start": self.config.train_start,
                "train_end": self.config.train_end,
                "val_start": self.config.val_start,
                "val_end": self.config.val_end,
            },
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Results saved to %s", output_dir / "results.json")

    def _print_summary(self, result_5d, result_20d):
        print()
        print("=" * 60)
        print("  LIGHTGBM MODEL EVALUATION")
        print("=" * 60)

        for label, result in [("ret_5d", result_5d), ("ret_20d", result_20d)]:
            print(f"\n  --- {label} ---")
            print(f"  MSE:      {result.mse:.6f}")
            print(f"  MAE:      {result.mae:.6f}")
            print(f"  Mean IC:  {result.mean_ic:.4f}")
            print(f"  IC Std:   {result.ic_std:.4f}")
            print(f"  ICIR:     {result.icir:.4f}")
            print(f"  Top 5 features (gain):")
            for name, val in result.top5_features:
                print(f"    {name}: {val:.4f}")
            print(f"  Bottom 5 features (gain):")
            for name, val in result.bottom5_features:
                print(f"    {name}: {val:.4f}")

        print()
        print("=" * 60)
