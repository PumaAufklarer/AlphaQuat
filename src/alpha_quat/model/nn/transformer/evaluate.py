import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from alpha_quat.model.nn.config import TransformerConfig
from alpha_quat.model.nn.transformer.models.dataset import SRSequenceDataset
from alpha_quat.model.nn.transformer.models.transformer import StockTransformer

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    model: StockTransformer,
    dataset: SRSequenceDataset,
    config: TransformerConfig,
) -> dict:
    """Evaluate model on dataset, return metrics per horizon."""
    model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Per-horizon metrics
    horizon_names = [
        "resistance_5d",
        "resistance_20d",
        "resistance_60d",
        "support_5d",
        "support_20d",
        "support_60d",
    ]
    horizon_losses = {h: [] for h in horizon_names}
    horizon_top3 = {h: [] for h in horizon_names}
    all_entropies = []

    for x, y in loader:
        x, y = x.to(_DEVICE), y.to(_DEVICE)
        logits = model(x)  # (B, 6, 100)

        probs = torch.softmax(logits, dim=-1)

        for h_idx in range(6):
            h_name = horizon_names[h_idx]
            loss = criterion(logits[:, h_idx], y[:, h_idx])
            horizon_losses[h_name].extend(loss.cpu().numpy().tolist())

            # Top-3 accuracy: is the true bin (argmax of y) in top 3 predicted?
            true_bin = y[:, h_idx].argmax(dim=1)
            _, top3 = probs[:, h_idx].topk(3, dim=1)
            top3_hits = (top3 == true_bin.unsqueeze(1)).any(dim=1).float()
            horizon_top3[h_name].extend(top3_hits.cpu().numpy().tolist())

        # Distribution sharpness (lower entropy = sharper)
        entropies = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # (B, 6)
        all_entropies.extend(entropies.mean(dim=1).cpu().numpy().tolist())

    metrics = {}
    for h_name in horizon_names:
        losses = np.array(horizon_losses[h_name])
        top3 = np.array(horizon_top3[h_name])
        metrics[f"{h_name}_loss"] = float(losses.mean())
        metrics[f"{h_name}_top3_acc"] = float(top3.mean())

    metrics["avg_entropy"] = float(np.mean(all_entropies))
    metrics["avg_loss"] = float(np.mean([metrics[f"{h}_loss"] for h in horizon_names]))

    logger.info("Evaluation results:")
    for h_name in horizon_names:
        logger.info(
            "  %s: loss=%.4f top3_acc=%.3f",
            h_name,
            metrics[f"{h_name}_loss"],
            metrics[f"{h_name}_top3_acc"],
        )
    logger.info("  avg_entropy=%.4f", metrics["avg_entropy"])

    return metrics
