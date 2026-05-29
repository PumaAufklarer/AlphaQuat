import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
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


class KeltnerRegimeTransformer(nn.Module):
    """Transformer that predicts market regime from price/Keltner sequences."""

    def __init__(
        self,
        n_features: int = 17,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 4,
        dim_feed: int = 512,
        dropout: float = 0.1,
        n_heads: int = 3,
        n_regimes: int = 5,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_regimes = n_regimes
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feed, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_heads * n_regimes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x.view(-1, self.n_heads, self.n_regimes)
