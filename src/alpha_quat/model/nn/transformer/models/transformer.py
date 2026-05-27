import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class StockTransformer(nn.Module):
    """Transformer that predicts SR distributions from 60-day sequences."""

    def __init__(
        self,
        n_features: int = 14,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 4,
        dim_feed: int = 512,
        dropout: float = 0.1,
        n_bins: int = 101,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feed,
            dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 6 * n_bins)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # (batch, d_model) — global average pooling
        x = self.head(x)  # (batch, 6 * n_bins)
        return x.view(-1, 6, self.head.out_features // 6)  # (batch, 6, n_bins)
