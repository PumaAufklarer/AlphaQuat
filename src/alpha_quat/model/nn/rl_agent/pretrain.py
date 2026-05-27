"""Supervised pre-training — predict direction of 5-day forward return.

Output: (B, 2) logits — [down, up].
Used to warm-start encoder for RL fine-tuning.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from alpha_quat.model.nn.rl_agent.dataset import (
    _FEATURE_COLS,
    _load_stock_data,
    _normalize_market,
    select_stocks,
)

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SEQ_LEN = 60


class DirectionEncoder(nn.Module):
    """Transformer encoder that outputs a market state embedding.

    Shared encoder between supervised pre-training and RL fine-tuning.
    """

    def __init__(
        self,
        n_features: int = 14,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 4,
        dim_feed: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feed, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x.mean(dim=1)  # (B, d_model)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        import math

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class DirectionClassifier(nn.Module):
    """Full pre-training model: encoder + classification head."""

    def __init__(self, n_features: int = 14, d_model: int = 128, **kwargs):
        super().__init__()
        self.encoder = DirectionEncoder(
            n_features=n_features, d_model=d_model, **kwargs
        )
        self.class_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        emb = self.encoder(x)
        return self.class_head(emb)


class _LRDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def _build_dataset(
    stock_data: dict,
    codes: list[str],
    horizon: int = 5,
    max_samples: int = 500_000,
):
    """Build supervised dataset: X = (60, 14) market data, Y = {0, 1} direction.

    Label: 0 if close[t+horizon] < close[t], else 1.
    """
    Xs, Ys = [], []
    for code in codes:
        df = stock_data.get(code)
        if df is None or len(df) < _SEQ_LEN + horizon:
            continue
        vals = df[_FEATURE_COLS].to_numpy(dtype=np.float32)
        close = df["close"].to_numpy(dtype=np.float64)

        for t in range(_SEQ_LEN, len(df) - horizon):
            raw = vals[t - _SEQ_LEN : t].copy()
            if np.isnan(raw).any():
                continue
            x = _normalize_market(raw)
            ret = close[t + horizon] / close[t] - 1.0
            y = 1 if ret >= 0 else 0
            Xs.append(x)
            Ys.append(y)
            if len(Xs) >= max_samples:
                break
        if len(Xs) >= max_samples:
            break

    logger.info("Built dataset: %d samples", len(Xs))
    return np.stack(Xs), np.array(Ys, dtype=np.int64)


def pretrain(
    data_dir: Path,
    output_dir: Path,
    train_start: str = "20200101",
    train_end: str = "20231231",
    val_start: str = "20240101",
    val_end: str = "20240630",
    n_epochs: int = 20,
    batch_size: int = 128,
    lr: float = 3e-4,
    max_samples: int = 500_000,
) -> DirectionEncoder:
    """Run supervised pre-training. Returns trained encoder."""
    logger.info("Loading training data: %s ~ %s", train_start, train_end)
    train_data = _load_stock_data(data_dir, train_start, train_end)
    train_codes = select_stocks(train_data, max_stocks=300)
    logger.info("Training on %d stocks", len(train_codes))

    logger.info("Loading validation data: %s ~ %s", val_start, val_end)
    val_data = _load_stock_data(data_dir, val_start, val_end)
    val_codes = select_stocks(val_data, max_stocks=100, min_days=60)

    logger.info("Building train dataset...")
    X_tr, Y_tr = _build_dataset(train_data, train_codes, max_samples=max_samples)
    train_ds = _LRDataset(X_tr, Y_tr)

    logger.info("Building val dataset...")
    X_val, Y_val = _build_dataset(val_data, val_codes, max_samples=100_000)
    val_ds = _LRDataset(X_val, Y_val)

    model = DirectionClassifier(n_features=14, d_model=128).to(_DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_acc = 0.0
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_ok = 0
        train_n = 0
        for x, y in train_loader:
            x, y = x.to(_DEVICE), y.to(_DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_ok += (logits.argmax(dim=1) == y).sum().item()
            train_n += y.size(0)

        model.eval()
        val_ok = 0
        val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(_DEVICE), y.to(_DEVICE)
                logits = model(x)
                val_ok += (logits.argmax(dim=1) == y).sum().item()
                val_n += y.size(0)

        train_acc = train_ok / max(train_n, 1)
        val_acc = val_ok / max(val_n, 1)
        logger.info(
            "Epoch %2d/%d: loss=%.4f train_acc=%.3f val_acc=%.3f",
            epoch + 1,
            n_epochs,
            train_loss / max(len(train_loader), 1),
            train_acc,
            val_acc,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.encoder.state_dict(), output_dir / "pretrained_encoder.pt")
            logger.info("  → saved best encoder (val_acc=%.3f)", best_acc)

    logger.info("Supervised pre-training complete. Best val acc: %.3f", best_acc)
    model.encoder.load_state_dict(
        torch.load(
            output_dir / "pretrained_encoder.pt",
            map_location=_DEVICE,
            weights_only=True,
        )
    )
    return model.encoder
