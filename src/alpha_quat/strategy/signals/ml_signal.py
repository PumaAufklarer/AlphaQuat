"""MLSignalGenerator — scores stocks using ensemble of 3 LightGBM models.

Supports point-estimate (regression) and quantile regression models.
Quantile mode loads all 3 per-horizon models (10%/50%/90%) to compute
confidence intervals alongside the median score.
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
        self.extra: dict[tuple[str, float], lgb.Booster] = {}
        self.quantile_mode = False

        # Detect quantile mode
        has_q = all(
            (model_dir / f"lightgbm_model_{h}_alpha_0.5.txt").exists()
            for h in ["5d", "20d", "60d"]
        )
        has_all = has_q and all(
            (model_dir / f"lightgbm_model_{h}_alpha_{a}.txt").exists()
            for h in ["5d", "20d", "60d"]
            for a in [0.1, 0.9]
        )

        if has_all:
            logger.info("Quantile mode: loading median + 10%/90% models")
            self.quantile_mode = True
            for h in ["5d", "20d", "60d"]:
                self.models[h] = lgb.Booster(
                    model_file=str(model_dir / f"lightgbm_model_{h}_alpha_0.5.txt")
                )
                self.extra[(h, 0.1)] = lgb.Booster(
                    model_file=str(model_dir / f"lightgbm_model_{h}_alpha_0.1.txt")
                )
                self.extra[(h, 0.9)] = lgb.Booster(
                    model_file=str(model_dir / f"lightgbm_model_{h}_alpha_0.9.txt")
                )
                logger.info("Loaded quantile models for %s", h)
        elif has_q:
            logger.info("Quantile mode (median only): no 10%/90% models found")
            self.quantile_mode = True
            for h in ["5d", "20d", "60d"]:
                path = model_dir / f"lightgbm_model_{h}_alpha_0.5.txt"
                if path.exists():
                    self.models[h] = lgb.Booster(model_file=str(path))
        else:
            logger.info("Regression mode: loading point-estimate models")
            for h in ["5d", "20d", "60d"]:
                path = model_dir / f"lightgbm_model_{h}.txt"
                if path.exists():
                    self.models[h] = lgb.Booster(model_file=str(path))

        if not self.models:
            raise FileNotFoundError(f"No models found in {model_dir}")

    def generate(self, features: pd.DataFrame, ctx: StrategyContext) -> SignalResult:
        factor_cols = [
            c for c in features.columns if c not in ("ts_code", "trade_date")
        ]
        X = features[factor_cols].fillna(0)

        n = len(X)
        score = np.zeros(n)
        extra_low = np.zeros(n)
        extra_high = np.zeros(n)
        has_extra = False

        for label, model in self.models.items():
            w = _WEIGHTS.get(label, 0)
            pred = np.asarray(model.predict(X), dtype=float)
            score += w * pred

            if self.quantile_mode and (label, 0.1) in self.extra:
                has_extra = True
                low = np.asarray(self.extra[(label, 0.1)].predict(X), dtype=float)
                hi = np.asarray(self.extra[(label, 0.9)].predict(X), dtype=float)
                extra_low += w * low
                extra_high += w * hi

        df = features[["ts_code"]].copy()
        df["score"] = score
        df["action"] = "buy"

        meta: dict = {"model": "ml_ensemble"}
        if has_extra:
            meta["score_low"] = extra_low
            meta["score_high"] = extra_high
            ci = (extra_high - extra_low) / 2
            meta["ci_width"] = ci
            # confidence: 1 - normalized CI (0 = wide = uncertain, 1 = narrow = certain)
            max_ci = max(np.max(ci), 1e-8)
            meta["confidence"] = 1.0 - ci / max_ci

        return SignalResult(signals=df, metadata=meta)
