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
        self._log_vol_mean = self.norm_params.get("log_vol_mean", 0.0)
        self._log_vol_std = self.norm_params.get("log_vol_std", 1.0)

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

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Per-sequence normalization: price ratios + log volume."""
        out = x.copy().astype(np.float32)
        close_last = out[-1, 3]
        if close_last <= 0:
            close_last = 1.0

        out[:, 0] = out[:, 0] / close_last - 1
        out[:, 1] = out[:, 1] / close_last - 1
        out[:, 2] = out[:, 2] / close_last - 1
        out[:, 3] = out[:, 3] / close_last - 1
        out[:, 5] = out[:, 5] / close_last - 1
        out[:, 4] = (np.log1p(out[:, 4]) - self._log_vol_mean) / self._log_vol_std
        return out

    def predict(self, sequence: np.ndarray) -> dict[str, np.ndarray]:
        """Predict SR distributions for a (60, 6) sequence."""
        seq = self._normalize(sequence)

        x = torch.from_numpy(seq).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            logits = self.model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        names = [
            "resistance_5d",
            "resistance_20d",
            "resistance_60d",
            "support_5d",
            "support_20d",
            "support_60d",
        ]
        return {name: probs[i] for i, name in enumerate(names)}

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """Batch inference: (B, 60, 6) → (B, 6, 100) probabilities."""
        tensor = torch.from_numpy(sequences).float().to(_DEVICE)
        with torch.no_grad():
            logits = self.model(tensor)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def compute_entry_exit_batch(
        self, sequences: np.ndarray, close_prices: np.ndarray
    ) -> list[dict]:
        """Batch entry/exit: (B, 60, 6) → list of per-stock signal dicts."""
        probs = self.predict_batch(sequences)
        n_bins = self.config.n_bins
        price_range = self.config.price_range
        bin_centers = np.linspace(-price_range, price_range, n_bins)

        results = []
        for i in range(probs.shape[0]):
            p = probs[i]
            expected_up = float((p[0] * bin_centers).sum())  # resistance_5d at idx 0
            expected_down = float(-(p[3] * bin_centers).sum())  # support_5d at idx 3
            rr_ratio = expected_up / max(expected_down, 1e-6)

            near_sup = (bin_centers > -expected_down - 0.01) & (
                bin_centers < -expected_down + 0.01
            )
            support_confidence = float(p[3][near_sup].sum())
            near_res = (bin_centers > expected_up - 0.01) & (
                bin_centers < expected_up + 0.01
            )
            resistance_confidence = float(p[0][near_res].sum())

            results.append(
                {
                    "entry": rr_ratio > 2.0 and support_confidence > 0.1,
                    "exit": resistance_confidence > 0.3,
                    "rr_ratio": rr_ratio,
                    "expected_up": expected_up,
                    "expected_down": -expected_down,
                    "support_confidence": support_confidence,
                    "resistance_confidence": resistance_confidence,
                }
            )
        return results

    def compute_entry_exit(self, sequence: np.ndarray, close_price: float) -> dict:
        """Compute entry/exit signals from SR predictions."""
        probs = self.predict(sequence)
        n_bins = self.config.n_bins
        price_range = self.config.price_range
        bin_centers = np.linspace(-price_range, price_range, n_bins)

        expected_resistance = {
            h: float((probs[f"resistance_{h}"] * bin_centers).sum())
            for h in ["5d", "20d", "60d"]
        }
        expected_support = {
            h: float((probs[f"support_{h}"] * bin_centers).sum())
            for h in ["5d", "20d", "60d"]
        }

        expected_up = expected_resistance["5d"]
        expected_down = -expected_support["5d"]
        rr_ratio = expected_up / max(expected_down, 1e-6)

        near_support_mask = (bin_centers > expected_support["5d"] - 0.01) & (
            bin_centers < expected_support["5d"] + 0.01
        )
        support_confidence = float(probs["support_5d"][near_support_mask].sum())
        entry = rr_ratio > 2.0 and support_confidence > 0.1

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
