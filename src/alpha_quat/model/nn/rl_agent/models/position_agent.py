import math

import torch
import torch.nn as nn
from alpha_quat.model.nn.rl_agent.pretrain import DirectionEncoder


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


class PositionAgent(nn.Module):
    """Transformer-based RL agent for continuous position control.

    Input:  (B, 60, n_features + 2) — market data + position_signal + days_held
    Output: μ (B, 1) — mean of Gaussian policy
    """

    def __init__(
        self,
        n_market_features: int = 14,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 4,
        dim_feed: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        n_input = n_market_features + 2
        self.input_proj = nn.Linear(n_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feed, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.actor_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.log_sigma = nn.Parameter(torch.tensor(-1.2))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        mu = self.actor_head(x)
        return mu

    def get_dist(self, x):
        mu = self.forward(x)
        sigma = torch.exp(self.log_sigma).clamp(min=1e-4, max=10.0)
        return torch.distributions.Normal(mu, sigma)

    def load_pretrained_encoder(self, encoder: DirectionEncoder):
        """Copy transformer encoder weights from pre-trained DirectionEncoder.

        Skips input_proj (different input dimension).
        Copies: pos_encoder, encoder transformer, layer norm.
        """
        self.encoder.load_state_dict(encoder.encoder.state_dict())
        self.norm.load_state_dict(encoder.norm.state_dict())
