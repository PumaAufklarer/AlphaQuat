from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from alpha_quat.strategy.types import SignalResult, StrategyContext
from alpha_quat.strategy.signals.variants import register
from alpha_quat.strategy.signals.variants.baseline import BaseMLSignal


@register
class MetaSignal(BaseMLSignal):
    mode = "meta"

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
            meta_path = model_dir / f"meta_model_{h}.txt"
            if meta_path.exists():
                models[f"meta_{h}"] = lgb.Booster(model_file=str(meta_path))
        return models

    def generate(self, features: pd.DataFrame, ctx: StrategyContext) -> SignalResult:
        X = self._prepare_features(features)
        n = len(X)
        base = {}
        for h in ["5d", "20d", "60d"]:
            base[h] = {}
            base[h][0.5] = np.asarray(self.models[h].predict(X), dtype=float)
            low_key = f"{h}_alpha_0.1"
            high_key = f"{h}_alpha_0.9"
            if low_key in self.models:
                base[h][0.1] = np.asarray(self.models[low_key].predict(X), dtype=float)
            if high_key in self.models:
                base[h][0.9] = np.asarray(self.models[high_key].predict(X), dtype=float)

        score = np.zeros(n)
        extra_low = np.zeros(n)
        extra_high = np.zeros(n)
        for h in ["5d", "20d", "60d"]:
            w = self._WEIGHTS[h]
            meta_key = f"meta_{h}"
            if meta_key in self.models:
                feats = np.column_stack(
                    [
                        base[hor][a]
                        for hor in ["5d", "20d", "60d"]
                        for a in [0.1, 0.5, 0.9]
                    ]
                )
                h_score = np.asarray(self.models[meta_key].predict(feats), dtype=float)
                score += w * h_score
            else:
                score += w * base[h][0.5]
            if 0.1 in base[h] and 0.9 in base[h]:
                extra_low += w * base[h][0.1]
                extra_high += w * base[h][0.9]

        ci = (extra_high - extra_low) / 2
        df = features[["ts_code"]].copy()
        df["score"] = score
        df["action"] = "buy"
        return SignalResult(
            signals=df,
            metadata={
                "model": "meta_stacking",
                "ci_width": ci,
                "confidence": 1.0 - ci / max(np.max(ci), 1e-8),
            },
        )
