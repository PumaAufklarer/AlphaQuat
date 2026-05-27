from pathlib import Path

import lightgbm as lgb
import pandas as pd

from alpha_quat.strategy.types import SignalResult, StrategyContext
from alpha_quat.strategy.signals.variants import register
from alpha_quat.strategy.signals.variants.baseline import BaseMLSignal


@register
class LambdaRankSignal(BaseMLSignal):
    mode = "lambdarank"

    def _load_models(self, model_dir: Path) -> dict:
        models = {}
        for h in ["5d", "20d", "60d"]:
            path = model_dir / f"lightgbm_model_{h}.txt"
            if path.exists():
                models[h] = lgb.Booster(model_file=str(path))
        if not models:
            raise FileNotFoundError(f"No lambdarank models found in {model_dir}")
        return models

    def generate(self, features: pd.DataFrame, ctx: StrategyContext) -> SignalResult:
        X = self._prepare_features(features)
        score = sum(self._WEIGHTS[h] * self.models[h].predict(X) for h in self.models)
        df = features[["ts_code"]].copy()
        df["score"] = score
        df["action"] = "buy"
        return SignalResult(signals=df, metadata={"model": "lambdarank"})
