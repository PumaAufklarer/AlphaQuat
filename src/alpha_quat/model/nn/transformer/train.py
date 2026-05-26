import logging
from math import cos, pi
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from alpha_quat.model.nn.config import TransformerConfig
from alpha_quat.model.nn.transformer.models.dataset import SRSequenceDataset
from alpha_quat.model.nn.transformer.models.transformer import StockTransformer

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weighted_cross_entropy(
    logits: torch.Tensor,  # (B, 6, n_bins)
    target: torch.Tensor,  # (B, 6) long — class indices
    weight: torch.Tensor,  # (B, 6) float32 — distance-decayed weights, 0 = invalid
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    B, H, C = logits.shape
    loss = F.cross_entropy(
        logits.view(-1, C),
        target.view(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    )
    loss = loss.view(B, H) * weight
    return loss.sum() / weight.sum().clamp(min=1)


def _validate(model, val_loader):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y, w in val_loader:
            x, y, w = x.to(_DEVICE), y.to(_DEVICE), w.to(_DEVICE)
            logits = model(x)
            loss = weighted_cross_entropy(logits, y, w)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


class _WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self, step: int):
        if step < self.warmup_steps:
            factor = step / max(self.warmup_steps, 1)
        else:
            progress = (step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            factor = 0.5 * (1 + cos(pi * min(progress, 1)))
        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base_lr * factor


def train(
    model: StockTransformer,
    train_dataset: SRSequenceDataset,
    val_dataset: SRSequenceDataset,
    config: TransformerConfig,
    output_dir: Path,
):
    model.to(_DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = _WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0

        for x, y, w in train_loader:
            x, y, w = x.to(_DEVICE), y.to(_DEVICE), w.to(_DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = weighted_cross_entropy(logits, y, w, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(global_step)
            global_step += 1
            train_loss += loss.item()
            train_count += 1

        val_loss = _validate(model, val_loader)

        logger.info(
            "Epoch %2d/%d: train_loss=%.4f val_loss=%.4f lr=%.2e",
            epoch + 1,
            config.epochs,
            train_loss / max(train_count, 1),
            val_loss,
            optimizer.param_groups[0]["lr"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    return model
