import json
import logging
from pathlib import Path

import numpy as np
import torch

from alpha_quat.model.nn.config import TransformerConfig
from alpha_quat.model.nn.transformer.models.transformer import StockTransformer

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_FEATURE_COLS = ["open", "high", "low", "close", "volume", "vwap"]


class SRInference:
    """Load a trained SR Transformer and run inference."""

    def __init__(self, model_dir: Path):
        with open(model_dir / "transformer_config.json") as f:
            cfg_dict = json.load(f)
        self.config = TransformerConfig(**cfg_dict)
        with open(model_dir / "norm_params.json") as f:
            self.norm_params = json.load(f)

        self.model = StockTransformer(
            n_features=self.config.n_features,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            n_layers=self.config.n_layers,
            dim_feed=self.config.dim_feed,
            dropout=self.config.dropout,
            n_bins=self.config.n_bins,
        )
        state = torch.load(
            model_dir / "model.pt", map_location=_DEVICE, weights_only=True
        )
        self.model.load_state_dict(state)
        self.model.to(_DEVICE)
        self.model.eval()

    def predict(self, sequence: np.ndarray) -> dict[str, np.ndarray]:
        """Predict SR distributions for a (60, 6) sequence.

        Args:
            sequence: (60, 6) array [open, high, low, close, volume, vwap]

        Returns:
            dict with keys: resistance_5d, resistance_20d, resistance_60d,
                            support_5d, support_20d, support_60d
            Each value is a (100,) probability array.
        """
        # Normalize
        seq = sequence.copy().astype(np.float32)
        for i, col in enumerate(_FEATURE_COLS):
            mean, std = self.norm_params[col]
            seq[:, i] = (seq[:, i] - mean) / std

        x = torch.from_numpy(seq).unsqueeze(0).to(_DEVICE)  # (1, 60, 6)
        with torch.no_grad():
            logits = self.model(x)  # (1, 6, n_bins)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # (6, n_bins)

        names = [
            "resistance_5d",
            "resistance_20d",
            "resistance_60d",
            "support_5d",
            "support_20d",
            "support_60d",
        ]
        return {name: probs[i] for i, name in enumerate(names)}

    def compute_entry_exit(
        self,
        sequence: np.ndarray,
        close_price: float,
    ) -> dict:
        """Compute entry/exit signals from SR predictions.

        Returns:
            dict with keys: rr_ratio, entry, exit, expected_up, expected_down
        """
        probs = self.predict(sequence)
        n_bins = self.config.n_bins
        price_range = self.config.price_range

        # Expected bin value for each distribution
        bin_centers = np.linspace(-price_range, price_range, n_bins)

        expected_resistance = {
            h: float((probs[f"resistance_{h}"] * bin_centers).sum())
            for h in ["5d", "20d", "60d"]
        }
        expected_support = {
            h: float((probs[f"support_{h}"] * bin_centers).sum())
            for h in ["5d", "20d", "60d"]
        }

        # Use 5d for nearest-term signals
        expected_up = expected_resistance["5d"]
        expected_down = -expected_support["5d"]
        rr_ratio = expected_up / max(expected_down, 1e-6)

        # Entry: good risk/reward + close near support
        near_support_mask = (bin_centers > expected_support["5d"] - 0.01) & (
            bin_centers < expected_support["5d"] + 0.01
        )
        support_confidence = float(probs["support_5d"][near_support_mask].sum())
        entry = rr_ratio > 2.0 and support_confidence > 0.1

        # Exit: price approaching resistance
        near_resistance_mask = (bin_centers > expected_resistance["5d"] - 0.01) & (
            bin_centers < expected_resistance["5d"] + 0.01
        )
        resistance_confidence = float(
            probs["resistance_5d"][near_resistance_mask].sum()
        )
        exit = resistance_confidence > 0.3

        return {
            "entry": entry,
            "exit": exit,
            "rr_ratio": rr_ratio,
            "expected_up": expected_up,
            "expected_down": expected_down,
            "support_confidence": support_confidence,
            "resistance_confidence": resistance_confidence,
        }
