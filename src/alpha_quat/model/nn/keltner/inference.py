import json
import logging
from pathlib import Path

import numpy as np
import torch

from alpha_quat.model.nn.keltner.models.dataset import (
    _HORIZONS,
    _REGIME_NAMES,
    _normalize_sequence,
)
from alpha_quat.model.nn.keltner.models.keltner_transformer import (
    KeltnerRegimeTransformer,
)

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KeltnerInference:
    """Load a trained KeltnerRegimeTransformer and run inference."""

    def __init__(self, model_dir: Path):
        with open(model_dir / "keltner_config.json") as f:
            cfg = json.load(f)
        self.n_features = cfg["n_features"]
        self.n_heads = cfg["n_heads"]
        self.n_regimes = cfg["n_regimes"]
        self.d_model = cfg["d_model"]
        self.nhead = cfg["nhead"]
        self.n_layers = cfg["n_layers"]
        self.dim_feed = cfg["dim_feed"]
        self.dropout = cfg["dropout"]
        self.seq_length = cfg["seq_length"]

        self.model = KeltnerRegimeTransformer(
            n_features=self.n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            n_layers=self.n_layers,
            dim_feed=self.dim_feed,
            dropout=self.dropout,
            n_heads=self.n_heads,
            n_regimes=self.n_regimes,
        )
        state = torch.load(
            model_dir / "best_model.pt", map_location=_DEVICE, weights_only=True
        )
        self.model.load_state_dict(state)
        self.model.to(_DEVICE)
        self.model.eval()

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """Predict regime probabilities for a (60, n_features) sequence.

        Returns (n_heads, n_regimes) probabilities.
        """
        seq = _normalize_sequence(sequence)
        x = torch.from_numpy(seq).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            logits = self.model(x)
        return torch.softmax(logits, dim=-1).cpu().numpy()[0]

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """Batch inference: (B, 60, n_features) → (B, n_heads, n_regimes)."""
        x = torch.from_numpy(sequences).float().to(_DEVICE)
        with torch.no_grad():
            logits = self.model(x)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def compute_regime_signals(self, sequences: np.ndarray) -> list[dict]:
        """Batch regime prediction → per-stock signal dicts.

        Returns list of dicts with keys:
          - regime_5d, regime_20d, regime_60d: predicted regime ID
          - regime_5d_conf, regime_20d_conf, regime_60d_conf: confidence (top-1 prob)
          - regime_5d_name, regime_20d_name, regime_60d_name: human-readable regime name
        """
        probs = self.predict_batch(sequences)

        results = []
        for i in range(probs.shape[0]):
            p = probs[i]
            row = {}
            for h_idx, horizon in enumerate(_HORIZONS):
                dist = p[h_idx]
                regime_id = int(dist.argmax())
                conf = float(dist[regime_id])
                row[f"regime_{horizon}d"] = regime_id
                row[f"regime_{horizon}d_conf"] = conf
                row[f"regime_{horizon}d_name"] = _REGIME_NAMES[regime_id]
            results.append(row)

        return results
