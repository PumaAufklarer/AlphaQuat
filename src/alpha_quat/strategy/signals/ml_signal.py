"""MLSignalGenerator — scores stocks using ensemble of 3 LightGBM models.

Supports both point-estimate (regression) and quantile regression models.
Quantile median models (alpha=0.5) are used when available.
"""

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from alpha_quat.strategy.signal import ISignalGenerator
from alpha_quat.strategy.types import SignalResult, StrategyContext

logger = logging.getLogger(__name__)

_WEIGHTS = {"5d": 0.35, "20d": 0.32, "60d": 0.33}


class MLSignalGenerator(ISignalGenerator):
    """Scores universe using ensemble of 5d/20d/60d LightGBM models."""

    def __init__(self, model_dir: Path, top_k: int = 5):
        self.top_k = top_k
        self.model_dir = Path(model_dir)
        self.models: dict[str, lgb.Booster] = {}

        # Detect quantile mode: look for median models
        has_quantile = all(
            (model_dir / f"lightgbm_model_{h}_alpha_0.5.txt").exists()
            for h in ["5d", "20d", "60d"]
        )

        if has_quantile:
            logger.info("Quantile mode: loading median (alpha=0.5) models")
            for h in ["5d", "20d", "60d"]:
                path = model_dir / f"lightgbm_model_{h}_alpha_0.5.txt"
                if path.exists():
                    self.models[h] = lgb.Booster(model_file=str(path))
                    logger.info("Loaded %s from %s", h, path)
        else:
            logger.info("Regression mode: loading point-estimate models")
            for h in ["5d", "20d", "60d"]:
                path = model_dir / f"lightgbm_model_{h}.txt"
                if path.exists():
                    self.models[h] = lgb.Booster(model_file=str(path))
                    logger.info("Loaded %s from %s", h, path)

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
