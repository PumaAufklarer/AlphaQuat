import logging

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from alpha_quat.model.nn.keltner.models.dataset import (
    KeltnerRegimeDataset,
    _HORIZONS,
)
from alpha_quat.model.nn.keltner.models.keltner_transformer import (
    KeltnerRegimeTransformer,
)

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    model: KeltnerRegimeTransformer,
    dataset: KeltnerRegimeDataset,
    batch_size: int = 128,
) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    targets = []

    for x, y in loader:
        x, y = x.to(_DEVICE), y.to(_DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=-1)
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0).astype(np.int64)
    targets = np.concatenate(targets, axis=0).astype(np.int64)

    metrics = {}
    all_acc = []
    all_f1 = []

    for h_idx, horizon in enumerate(_HORIZONS):
        p = preds[:, h_idx]
        t = targets[:, h_idx]
        acc = (p == t).mean()
        f1 = f1_score(t, p, average="macro")
        all_acc.append(acc)
        all_f1.append(f1)
        metrics[f"{horizon}d_accuracy"] = float(acc)
        metrics[f"{horizon}d_f1"] = float(f1)
        cm = confusion_matrix(t, p, labels=range(5))
        metrics[f"{horizon}d_confusion_matrix"] = cm.tolist()

    metrics["avg_accuracy"] = float(np.mean(all_acc))
    metrics["avg_f1"] = float(np.mean(all_f1))

    logger.info("Evaluation results:")
    for h_idx, horizon in enumerate(_HORIZONS):
        logger.info(
            "  %dd: acc=%.3f f1=%.3f",
            horizon,
            metrics[f"{horizon}d_accuracy"],
            metrics[f"{horizon}d_f1"],
        )
    logger.info("  avg: acc=%.3f f1=%.3f", metrics["avg_accuracy"], metrics["avg_f1"])

    return metrics
