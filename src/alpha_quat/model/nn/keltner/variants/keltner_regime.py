import json
import logging
from pathlib import Path

import torch

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.experiment.registry import ExperimentRegistry
from alpha_quat.model.nn.keltner.evaluate import evaluate
from alpha_quat.model.nn.keltner.models.dataset import build_datasets
from alpha_quat.model.nn.keltner.models.keltner_transformer import (
    KeltnerRegimeTransformer,
)
from alpha_quat.model.nn.keltner.train import train

logger = logging.getLogger(__name__)


def run_keltner_regime(data_dir: Path, config: ExperimentConfig) -> dict:
    """Run Keltner Regime Transformer training pipeline."""
    train_ds, val_ds = build_datasets(
        data_dir,
        seq_length=60,
        stride=10,
        train_start=config.train_start,
        train_end=config.train_end,
        val_start=config.val_start,
        val_end=config.val_end,
    )
    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    model = KeltnerRegimeTransformer(
        n_features=17,
        d_model=128,
        nhead=4,
        n_layers=4,
        dim_feed=512,
        dropout=0.1,
        n_heads=3,
        n_regimes=5,
    )

    exp_dir = data_dir / "models" / "experiments" / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    config.save(exp_dir / "experiment.yaml")

    logger.info("Training Keltner Regime Transformer...")
    model = train(model, train_ds, val_ds, exp_dir)

    torch.save(model.state_dict(), exp_dir / "model.pt")

    keltner_cfg = {
        "n_features": 17,
        "n_heads": 3,
        "n_regimes": 5,
        "d_model": 128,
        "nhead": 4,
        "n_layers": 4,
        "dim_feed": 512,
        "dropout": 0.1,
        "seq_length": 60,
    }
    with open(exp_dir / "keltner_config.json", "w") as f:
        json.dump(keltner_cfg, f, indent=2)

    metrics = evaluate(model, val_ds)
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    ExperimentRegistry(data_dir).register(config)

    logger.info("Keltner Regime experiment '%s' complete.", config.name)
    return metrics
