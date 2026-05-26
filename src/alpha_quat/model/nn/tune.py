"""Grid search for Transformer SR hyperparameters."""

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import torch

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.model.nn.config import TransformerConfig
from alpha_quat.model.nn.transformer.models.dataset import (
    build_datasets,
)
from alpha_quat.model.nn.transformer.models.transformer import StockTransformer
from alpha_quat.model.nn.transformer.train import train
from alpha_quat.model.nn.transformer.evaluate import evaluate

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_GRID = {
    "lr": [3e-4, 1e-3, 3e-3],
    "d_model": [64, 128, 256],
    "dropout": [0.1, 0.2],
}

_NHEAD_MAP = {64: 2, 128: 4, 256: 8}


def _make_label(lr, d_model, dropout):
    return (
        f"lr{lr:.0e}_d{d_model}_do{dropout}".replace(".", "p")
        .replace("e", "e")
        .replace("+", "")
    )


def _run_one(exp_name, config, data_dir, train_ds, val_ds):
    nhead = _NHEAD_MAP[config.d_model]
    model = StockTransformer(
        n_features=config.n_features,
        d_model=config.d_model,
        nhead=nhead,
        n_layers=config.n_layers,
        dim_feed=config.dim_feed,
        dropout=config.dropout,
        n_bins=config.n_bins,
    )

    exp_dir = data_dir / "models" / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = train(model, train_ds, val_ds, config, exp_dir)

    metrics = evaluate(model, val_ds, config)

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def grid_search(data_dir: Path, config: ExperimentConfig):
    """Run grid search over hyperparameters."""
    tr_cfg_base = TransformerConfig(
        train_start=config.train_start,
        train_end=config.train_end,
        val_start=config.val_start,
        val_end=config.val_end,
    )

    logger.info("Building datasets once...")
    train_ds, val_ds, norm_params = build_datasets(data_dir, tr_cfg_base)
    logger.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    results = []
    keys = list(_GRID.keys())
    values = list(_GRID.values())
    total = len(list(itertools.product(*values)))

    logger.info("Grid search: %d combinations", total)

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        lr, d_model, dropout = params["lr"], params["d_model"], params["dropout"]
        label = _make_label(lr, d_model, dropout)
        exp_name = f"{config.name}/{label}"
        nhead = _NHEAD_MAP[d_model]

        tr_cfg = TransformerConfig(
            train_start=config.train_start,
            train_end=config.train_end,
            val_start=config.val_start,
            val_end=config.val_end,
            lr=lr,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        logger.info("=" * 60)
        logger.info("Trial: lr=%.0e d_model=%d dropout=%.1f", lr, d_model, dropout)
        logger.info("=" * 60)

        try:
            metrics = _run_one(exp_name, tr_cfg, data_dir, train_ds, val_ds)
            metrics["params"] = {"lr": lr, "d_model": d_model, "dropout": dropout}
            results.append(metrics)
            logger.info(
                "  val_loss=%.4f avg_entropy=%.4f",
                metrics.get("avg_loss", float("nan")),
                metrics.get("avg_entropy", float("nan")),
            )
        except Exception as e:
            logger.error("Trial failed: %s", e)
            results.append(
                {
                    "params": {"lr": lr, "d_model": d_model, "dropout": dropout},
                    "error": str(e),
                }
            )

    # Summary table
    print()
    print("=" * 80)
    print("  GRID SEARCH RESULTS (sorted by val_loss)")
    print("=" * 80)
    print(f"  {'Params':<30} {'val_loss':<10} {'top3_acc':<10} {'entropy':<10}")
    print("  " + "-" * 60)

    sorted_results = sorted(
        [r for r in results if "avg_loss" in r],
        key=lambda r: r.get("avg_loss", float("inf")),
    )
    for r in sorted_results:
        p = r["params"]
        label = f"lr={p['lr']:.0e} d={p['d_model']} do={p['dropout']}"
        avg_loss = r.get("avg_loss", float("nan"))
        top3 = np.mean(
            [
                r.get(f"{h}_top3_acc", 0)
                for h in [
                    "resistance_5d",
                    "resistance_20d",
                    "resistance_60d",
                    "support_5d",
                    "support_20d",
                    "support_60d",
                ]
            ]
        )
        ent = r.get("avg_entropy", float("nan"))
        print(f"  {label:<30} {avg_loss:<10.4f} {top3:<10.3f} {ent:<10.4f}")

    print()

    # Save results
    out_path = data_dir / "models" / "experiments" / config.name / "grid_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "best": {
                    "params": sorted_results[0]["params"],
                    "val_loss": sorted_results[0].get("avg_loss", None),
                    "top3_acc": float(
                        np.mean(
                            [
                                sorted_results[0].get(f"{h}_top3_acc", 0)
                                for h in [
                                    "resistance_5d",
                                    "resistance_20d",
                                    "resistance_60d",
                                    "support_5d",
                                    "support_20d",
                                    "support_60d",
                                ]
                            ]
                        )
                    ),
                },
                "all": sorted_results,
            },
            f,
            indent=2,
            default=str,
        )

    logger.info("Results saved to %s", out_path)
    return sorted_results
