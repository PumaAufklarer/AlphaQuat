import logging
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


def masked_cross_entropy(
    logits: torch.Tensor,  # (B, 6, n_bins)
    target: torch.Tensor,  # (B, 6, n_bins)
    mask: torch.Tensor,  # (B, 6) bool
) -> torch.Tensor:
    """Cross-entropy masked to only valid horizons."""
    B, H, C = logits.shape
    loss = F.cross_entropy(logits.view(-1, C), target.view(-1, C), reduction="none")
    loss = loss.view(B, H) * mask
    valid_count = mask.sum()
    return loss.sum() / valid_count.clamp(min=1)


def _validate(model, val_loader):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y, m in val_loader:
            x, y, m = x.to(_DEVICE), y.to(_DEVICE), m.to(_DEVICE)
            logits = model(x)
            loss = masked_cross_entropy(logits, y, m)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0

        for x, y, m in train_loader:
            x, y, m = x.to(_DEVICE), y.to(_DEVICE), m.to(_DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = masked_cross_entropy(logits, y, m)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_count += 1

        scheduler.step()
        val_loss = _validate(model, val_loader)

        logger.info(
            "Epoch %2d/%d: train_loss=%.4f val_loss=%.4f",
            epoch + 1,
            config.epochs,
            train_loss / max(train_count, 1),
            val_loss,
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
