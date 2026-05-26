import json
import logging
from pathlib import Path

import torch

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.experiment.registry import ExperimentRegistry
from alpha_quat.model.nn.config import TransformerConfig
from alpha_quat.model.nn.transformer.evaluate import evaluate
from alpha_quat.model.nn.transformer.models.dataset import build_datasets
from alpha_quat.model.nn.transformer.models.transformer import StockTransformer
from alpha_quat.model.nn.transformer.train import train

logger = logging.getLogger(__name__)


def run_sr_transformer(data_dir: Path, config: ExperimentConfig) -> dict:
    """Run SR Transformer training pipeline."""
    tr_cfg = TransformerConfig(
        train_start=config.train_start,
        train_end=config.train_end,
        val_start=config.val_start,
        val_end=config.val_end,
    )

    logger.info("Building datasets...")
    train_ds, val_ds, norm_params = build_datasets(data_dir, tr_cfg)
    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    model = StockTransformer(
        n_features=tr_cfg.n_features,
        d_model=tr_cfg.d_model,
        nhead=tr_cfg.nhead,
        n_layers=tr_cfg.n_layers,
        dim_feed=tr_cfg.dim_feed,
        dropout=tr_cfg.dropout,
        n_bins=tr_cfg.n_bins,
    )

    exp_dir = data_dir / "models" / "experiments" / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config.save(exp_dir / "experiment.yaml")

    logger.info("Training Transformer...")
    model = train(model, train_ds, val_ds, tr_cfg, exp_dir)

    # Save final model
    torch.save(model.state_dict(), exp_dir / "model.pt")

    # Save config for inference
    with open(exp_dir / "transformer_config.json", "w") as f:
        import dataclasses

        json.dump(dataclasses.asdict(tr_cfg), f, indent=2)

    # Save norm params for inference
    norm_save = {k: list(v) for k, v in norm_params.items()}
    with open(exp_dir / "norm_params.json", "w") as f:
        json.dump(norm_save, f, indent=2)

    # Evaluate
    metrics = evaluate(model, val_ds, tr_cfg)
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Register experiment
    ExperimentRegistry(data_dir).register(config)

    logger.info("SR Transformer experiment '%s' complete.", config.name)
    return metrics
