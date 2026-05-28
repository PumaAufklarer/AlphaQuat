from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from alpha_quat.strategy.types import SignalResult, StrategyContext
from alpha_quat.strategy.signals.variants import register
from alpha_quat.strategy.signals.variants.baseline import BaseMLSignal


@register
class QuantileSignal(BaseMLSignal):
    mode = "quantile"

    def _load_models(self, model_dir: Path) -> dict:
        models = {}
        for h in ["5d", "20d", "60d"]:
            models[h] = lgb.Booster(
                model_file=str(model_dir / f"lightgbm_model_{h}_alpha_0.5.txt")
            )
            for a in [0.1, 0.9]:
                path = model_dir / f"lightgbm_model_{h}_alpha_{a}.txt"
                if path.exists():
                    models[f"{h}_alpha_{a}"] = lgb.Booster(model_file=str(path))
        return models

    def generate(self, features: pd.DataFrame, ctx: StrategyContext) -> SignalResult:
        X, feats = self._prepare_features(features)
        n = len(X)
        score = np.zeros(n)
        extra_low = np.zeros(n)
        extra_high = np.zeros(n)
        for h in ["5d", "20d", "60d"]:
            w = self._WEIGHTS[h]
            med = np.asarray(self.models[h].predict(X), dtype=float)
            score += w * med
            low_key = f"{h}_alpha_0.1"
            high_key = f"{h}_alpha_0.9"
            if low_key in self.models and high_key in self.models:
                extra_low += w * np.asarray(
                    self.models[low_key].predict(X), dtype=float
                )
                extra_high += w * np.asarray(
                    self.models[high_key].predict(X), dtype=float
                )
        ci = (extra_high - extra_low) / 2
        df = feats[["ts_code"]].copy()
        df["score"] = score
        df["action"] = "buy"
        return SignalResult(
            signals=df,
            metadata={
                "model": "quantile",
                "ci_width": ci,
                "confidence": 1.0 - ci / max(np.max(ci), 1e-8),
            },
        )
