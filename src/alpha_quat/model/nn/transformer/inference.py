import json
import logging
from pathlib import Path

import numpy as np
import torch

from alpha_quat.model.nn.config import TransformerConfig
from alpha_quat.model.nn.transformer.models.transformer import StockTransformer

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "volume_ratio",
    "turnover_rate",
    "hl_ratio",
    "ret_5d",
    "close_ma20",
    "atr_ratio",
    "vol_change",
    "amt_change",
]


class SRInference:
    """Load a trained SR Transformer and run inference."""

    def __init__(self, model_dir: Path) -> None:
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
        """Per-sequence normalization: price ratios + log vol + per-seq z-score."""
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

        for j in range(6, x.shape[1]):
            col = out[:, j]
            mean = col.mean()
            std = col.std()
            if std > 1e-8:
                out[:, j] = (col - mean) / std
            else:
                out[:, j] = 0.0
            out[:, j] = np.clip(out[:, j], -5.0, 5.0)

        return out

    def predict(self, sequence: np.ndarray) -> dict[str, np.ndarray]:
        """Predict SR distributions for a (60, n_features) sequence."""
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
        """Batch inference: (B, 60, n_features) → (B, 6, n_bins) probabilities."""
        tensor = torch.from_numpy(sequences).float().to(_DEVICE)
        with torch.no_grad():
            logits = self.model(tensor)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def compute_entry_exit_batch(
        self, sequences: np.ndarray, close_prices: np.ndarray
    ) -> list[dict]:
        """Batch entry/exit: (B, 60, n_features) → list of per-stock signal dicts.

        Bin 0..n_bins-2 = price ratios, bin n_bins-1 = "no peak".
        Expected values computed only from price bins (excl. no-peak bin).
        """
        probs = self.predict_batch(sequences)
        n_bins = self.config.n_bins
        n_price_bins = n_bins - 1
        price_range = self.config.price_range
        bin_centers = np.linspace(-price_range, price_range, n_price_bins)

        results = []
        for i in range(probs.shape[0]):
            p = probs[i]

            def _expected(head_idx: int) -> tuple[float, float, float]:
                h = p[head_idx]
                price_probs = h[:n_price_bins]
                no_peak_prob = float(h[n_price_bins])
                price_sum = price_probs.sum()
                if price_sum > 1e-6:
                    p_norm = price_probs / price_sum
                    exp = float((p_norm * bin_centers).sum())
                else:
                    p_norm = price_probs
                    exp = 0.0
                near = (bin_centers > exp - 0.01) & (bin_centers < exp + 0.01)
                conf = float(price_probs[near].sum() / max(price_sum, 1e-6))
                return exp, conf, no_peak_prob

            exp_up, res_conf, _ = _expected(0)
            exp_down_raw, sup_conf, _ = _expected(3)
            expected_down = -exp_down_raw
            rr_ratio = abs(exp_up) / max(abs(expected_down), 1e-6)

            results.append(
                {
                    "entry": rr_ratio > 2.0 and sup_conf > 0.1,
                    "exit": res_conf > 0.3,
                    "rr_ratio": rr_ratio,
                    "expected_up": exp_up,
                    "expected_down": expected_down,
                    "support_confidence": sup_conf,
                    "resistance_confidence": res_conf,
                }
            )
        return results

    def compute_entry_exit(self, sequence: np.ndarray, close_price: float) -> dict:
        """Compute entry/exit signals from SR predictions.

        Bin 0..n_bins-2 = price ratios, bin n_bins-1 = "no peak".
        Expected values computed only from price bins (excl. no-peak bin).
        """
        probs = self.predict(sequence)
        n_bins = self.config.n_bins
        n_price_bins = n_bins - 1
        price_range = self.config.price_range
        bin_centers = np.linspace(-price_range, price_range, n_price_bins)

        def _expected(name: str) -> tuple[float, float, float]:
            h = probs[name]
            price_probs = h[:n_price_bins]
            no_peak_prob = float(h[n_price_bins])
            price_sum = price_probs.sum()
            if price_sum > 1e-6:
                p_norm = price_probs / price_sum
                exp = float((p_norm * bin_centers).sum())
            else:
                p_norm = price_probs
                exp = 0.0
            near = (bin_centers > exp - 0.01) & (bin_centers < exp + 0.01)
            conf = float(price_probs[near].sum() / max(price_sum, 1e-6))
            return exp, conf, no_peak_prob

        expected_resistance = {}
        expected_support = {}
        for h in ["5d", "20d", "60d"]:
            expected_resistance[h], _, _ = _expected(f"resistance_{h}")
            expected_support[h], _, _ = _expected(f"support_{h}")

        expected_up = expected_resistance["5d"]
        expected_down = -expected_support["5d"]
        rr_ratio = abs(expected_up) / max(abs(expected_down), 1e-6)

        _, sup_conf, _ = _expected("support_5d")
        entry = rr_ratio > 2.0 and sup_conf > 0.1

        _, res_conf, _ = _expected("resistance_5d")
        exit = res_conf > 0.3

        return {
            "entry": entry,
            "exit": exit,
            "rr_ratio": rr_ratio,
            "expected_up": expected_up,
            "expected_down": expected_down,
            "support_confidence": sup_conf,
            "resistance_confidence": res_conf,
        }
