import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alpha_quat.model.nn.config import TransformerConfig
from alpha_quat.model.nn.transformer.models.dataset import SRSequenceDataset
from alpha_quat.model.nn.transformer.models.transformer import StockTransformer

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_HORIZON_NAMES = [
    "resistance_5d",
    "resistance_20d",
    "resistance_60d",
    "support_5d",
    "support_20d",
    "support_60d",
]


@torch.no_grad()
def evaluate(
    model: StockTransformer,
    dataset: SRSequenceDataset,
    config: TransformerConfig,
) -> dict:
    """Evaluate model on dataset, return metrics per horizon."""
    model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    horizon_losses = {h: [] for h in _HORIZON_NAMES}
    horizon_top3 = {h: [] for h in _HORIZON_NAMES}
    all_entropies = []

    for x, y, m in loader:
        x, y, m = x.to(_DEVICE), y.to(_DEVICE), m.to(_DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)

        for h_idx in range(6):
            h_name = _HORIZON_NAMES[h_idx]
            h_mask = m[:, h_idx]

            if h_mask.sum() == 0:
                continue

            per_sample = F.cross_entropy(
                logits[:, h_idx], y[:, h_idx], reduction="none"
            )
            horizon_losses[h_name].extend(per_sample[h_mask].cpu().numpy().tolist())

            true_bin = y[:, h_idx].argmax(dim=1)
            _, top3 = probs[:, h_idx].topk(3, dim=1)
            top3_hit = (top3 == true_bin.unsqueeze(1)).any(dim=1).float()
            horizon_top3[h_name].extend(top3_hit[h_mask].cpu().numpy().tolist())

        entropies = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        all_entropies.extend(entropies.mean(dim=1).cpu().numpy().tolist())

    metrics = {}
    for h_name in _HORIZON_NAMES:
        losses = np.array(horizon_losses[h_name])
        top3 = np.array(horizon_top3[h_name])
        metrics[f"{h_name}_loss"] = (
            float(losses.mean()) if len(losses) > 0 else float("nan")
        )
        metrics[f"{h_name}_top3_acc"] = (
            float(top3.mean()) if len(top3) > 0 else float("nan")
        )

    metrics["avg_entropy"] = float(np.mean(all_entropies))
    valid_losses = [
        metrics[f"{h}_loss"]
        for h in _HORIZON_NAMES
        if not np.isnan(metrics[f"{h}_loss"])
    ]
    metrics["avg_loss"] = float(np.mean(valid_losses)) if valid_losses else float("nan")

    logger.info("Evaluation results:")
    for h_name in _HORIZON_NAMES:
        logger.info(
            "  %s: loss=%.4f top3_acc=%.3f",
            h_name,
            metrics[f"{h_name}_loss"],
            metrics[f"{h_name}_top3_acc"],
        )
    logger.info("  avg_entropy=%.4f", metrics["avg_entropy"])

    return metrics
