"""MLSignalGenerator — scores stocks using ensemble of 3 LightGBM models."""

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from alpha_quat.strategy.signal import ISignalGenerator
from alpha_quat.strategy.types import SignalResult, StrategyContext

logger = logging.getLogger(__name__)

_WEIGHTS = {
    "ret_5d": 0.35,
    "ret_20d": 0.32,
    "ret_60d": 0.33,
}


class MLSignalGenerator(ISignalGenerator):
    """Scores universe using ensemble of 5d/20d/60d LightGBM models."""

    def __init__(self, model_dir: Path, top_k: int = 5):
        self.top_k = top_k
        self.model_dir = Path(model_dir)
        self.models: dict[str, lgb.Booster] = {}
        for label in ["ret_5d", "ret_20d", "ret_60d"]:
            path = model_dir / f"lightgbm_model_{label.replace('ret_', '')}.txt"
            if path.exists():
                self.models[label] = lgb.Booster(model_file=str(path))
                logger.info("Loaded %s from %s", label, path)
            else:
                logger.warning("Model %s not found at %s", label, path)

        if not self.models:
            raise FileNotFoundError(f"No models found in {model_dir}")

    def generate(self, features: pd.DataFrame, ctx: StrategyContext) -> SignalResult:
        factor_cols = [
            c for c in features.columns if c not in ("ts_code", "trade_date")
        ]
        X = features[factor_cols].fillna(0)

        score = np.zeros(len(X))
        for label, model in self.models.items():
            w = _WEIGHTS.get(label, 0)
            pred = np.asarray(model.predict(X), dtype=float)
            score += w * pred

        df = features[["ts_code"]].copy()
        df["score"] = score
        df["action"] = "buy"

        return SignalResult(signals=df, metadata={"model": "ml_ensemble"})
