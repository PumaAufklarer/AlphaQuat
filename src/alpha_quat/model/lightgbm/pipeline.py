import json
import logging
from pathlib import Path

from alpha_quat.model.data import DatasetBuilder
from alpha_quat.model.lightgbm.config import LightGBMConfig
from alpha_quat.model.lightgbm.evaluate import EvalResult, LightGBMEvaluator
from alpha_quat.model.lightgbm.train import LightGBMTrainer, pinball_loss

logger = logging.getLogger(__name__)


class LightGBMPipeline:
    def __init__(self, data_dir: Path, config: LightGBMConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        self.builder = DatasetBuilder(self.data_dir)
        self.trainer = LightGBMTrainer(config)
        self.evaluator = LightGBMEvaluator()

    def _suffix(self, horizon: str, alpha: float | None) -> str:
        if alpha is not None:
            return f"{horizon}_alpha_{alpha}"
        return horizon

    def _model_path(self, suffix: str) -> Path:
        return self.data_dir / "models" / f"lightgbm_model_{suffix}.txt"

    def run(self) -> dict[str, dict[str, EvalResult]]:
        logger.info("Building dataset...")
        data = self.builder.build(
            self.config.train_start,
            self.config.train_end,
            self.config.val_start,
            self.config.val_end,
            feature_names=self.config.feature_names,
            lambdarank=self.config.lambdarank,
        )

        logger.info(
            "Training data: %d samples, Validation data: %d samples",
            len(data.X_train),
            len(data.X_val),
        )

        horizons = [
            ("5d", data.y_train_5, data.y_val_5),
            ("20d", data.y_train_20, data.y_val_20),
            ("60d", data.y_train_60, data.y_val_60),
        ]
        alphas = self.config.quantile_alphas or [None]
        results: dict[str, dict[str, EvalResult]] = {}
        output_dir = self.data_dir / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        for h_name, y_tr, y_val in horizons:
            results[h_name] = {}
            lr = self.config.lambdarank

            if lr:
                label = f"{h_name}_lambdarank"
                logger.info("Training lambdarank %s...", label)
                model, params = self.trainer.train(
                    data.X_train,
                    y_tr,
                    label,
                    lambdarank=True,
                    groups=data.train_groups,
                )
                logger.info("Evaluating %s...", label)
                result = self.evaluator.evaluate(
                    model,
                    data.X_val,
                    y_val,
                    data.val_dates,
                    data.val_codes,
                    params,
                    self.config.feature_names,
                    label,
                )
                results[h_name][label] = result
                output_dir = self.data_dir / "models"
                output_dir.mkdir(parents=True, exist_ok=True)
                model.save_model(str(output_dir / f"lightgbm_model_{label}.txt"))
                continue
            for alpha in alphas:
                suffix = self._suffix(h_name, alpha)
                label = suffix

                logger.info("Training %s...", label)
                model, params = self.trainer.train(
                    data.X_train, y_tr, label, quantile_alpha=alpha
                )

                logger.info("Evaluating %s...", label)
                result = self.evaluator.evaluate(
                    model,
                    data.X_val,
                    y_val,
                    data.val_dates,
                    data.val_codes,
                    params,
                    self.config.feature_names,
                    label,
                    quantile_alpha=alpha,
                )
                results[h_name][suffix] = result

                output_dir = self.data_dir / "models"
                output_dir.mkdir(parents=True, exist_ok=True)
                model.save_model(str(output_dir / f"lightgbm_model_{suffix}.txt"))

        self._save_results(results)
        self._print_summary(results)

        # Meta model training (stacking layer on top of base models)
        if self.config.quantile_alphas and self.config.meta_start:
            logger.info("Training meta models (stacking layer)...")
            from alpha_quat.model.meta import (
                build_meta_features,
                train_meta_model,
            )

            assert self.config.meta_start and self.config.meta_end
            meta_train_df, meta_val_df = build_meta_features(
                data_dir=self.data_dir,
                model_dir=output_dir,
                train_start=self.config.meta_start,
                train_end=self.config.meta_end,
                val_start=self.config.val_start,
                val_end=self.config.val_end,
            )

            for h in ["5d", "20d", "60d"]:
                train_meta_model(
                    meta_train_df,
                    meta_val_df,
                    horizon=h,
                    output_dir=output_dir,
                    tune=False,
                )
            logger.info("Meta training complete")

        return results

    def _save_results(self, results: dict[str, dict[str, EvalResult]]):
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
            "config": {
                "train_start": self.config.train_start,
                "train_end": self.config.train_end,
                "val_start": self.config.val_start,
                "val_end": self.config.val_end,
            },
        }
        for h_name, alphas_dict in results.items():
            for suffix, result in alphas_dict.items():
                output[suffix] = _make_json(result)

        with open(output_dir / "results.json", "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Results saved to %s", output_dir / "results.json")

    def _print_summary(self, results: dict[str, dict[str, EvalResult]]):
        print()
        print("=" * 60)
        print("  LIGHTGBM MODEL EVALUATION")
        print("=" * 60)

        for h_name, alphas_dict in results.items():
            for suffix, result in alphas_dict.items():
                metric = f"Pinball" if "_alpha_" in suffix else "MSE"
                print(f"\n  --- {suffix} ---")
                print(
                    f"  {metric}: {result.mse:.6f}"
                    if "_alpha_" in suffix
                    else f"  MSE:      {result.mse:.6f}"
                )
                if "_alpha_0.5" in suffix or "_alpha_" not in suffix:
                    print(f"  Mean IC:  {result.mean_ic:.4f}")
                    print(f"  IC Std:   {result.ic_std:.4f}")
                    print(f"  ICIR:     {result.icir:.4f}")
                print("  Top 5 features (gain):")
                for name, val in result.top5_features:
                    print(f"    {name}: {val:.4f}")
                print("  Bottom 5 features (gain):")
                for name, val in result.bottom5_features:
                    print(f"    {name}: {val:.4f}")

        print()
        print("=" * 60)
