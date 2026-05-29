import json
import logging
from pathlib import Path

import numpy as np
import torch

from alpha_quat.model.nn.rl_agent.dataset import (
    build_state,
)
from alpha_quat.model.nn.rl_agent.models.position_agent import PositionAgent

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_SEQ_LEN = 60


class PositionAgentInference:
    """Load a trained PositionAgent and run daily inference."""

    def __init__(self, model_dir: Path) -> None:
        with open(model_dir / "agent_config.json") as f:
            cfg = json.load(f)
        self.seq_length = cfg.get("seq_length", 60)

        self.model = PositionAgent(
            n_market_features=cfg.get("n_market_features", 14),
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            n_layers=cfg.get("n_layers", 4),
            dim_feed=cfg.get("dim_feed", 512),
            dropout=cfg.get("dropout", 0.1),
        )
        state = torch.load(
            model_dir / "best_model.pt", map_location=_DEVICE, weights_only=True
        )
        self.model.load_state_dict(state)
        self.model.to(_DEVICE)
        self.model.eval()

    @torch.no_grad()
    def predict_daily(
        self,
        market_seq: np.ndarray,
        current_position: float = 0.0,
        days_held: int = 0,
    ) -> dict:
        """Predict position signal for today.

        market_seq: (60, 14) market data array
        current_position: previous day's position [-1, 1]
        days_held: days since last position change

        Returns dict with: position, confidence (-σ), raw_score
        """
        state = build_state(market_seq, current_position, days_held, self.seq_length)
        state_t = torch.from_numpy(state).unsqueeze(0).to(_DEVICE)

        dist = self.model.get_dist(state_t)
        mu = float(dist.mean.item())
        sigma = float(dist.stddev.item())
        position = float(torch.tanh(torch.tensor([[mu]], device=state_t.device)).item())
        confidence = 1.0 - min(sigma, 1.0)

        return {
            "position": position,
            "confidence": confidence,
            "raw_mu": mu,
            "raw_sigma": sigma,
            "signal": "long"
            if position > 0.3
            else "short"
            if position < -0.3
            else "neutral",
        }
